"""Frame decoding with NVIDIA hardware acceleration and graceful fallback.

For 4K120 HEVC off an iPhone 17 Pro Max, *decoding is the bottleneck*, not the
tracking maths. NVDEC is a dedicated silicon block separate from the CUDA
cores, so hardware decode runs essentially for free alongside GPU fitting.

We decode straight to 8-bit grayscale where possible: segmentation only needs
luma, and it cuts the pipe bandwidth by 3x. At 4K that is the difference
between ~25 MB and ~8 MB per frame crossing the process boundary.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .ffmpeg import available as ffmpeg_available, ffmpeg, popen as _popen, run as _run
from .ingest import VideoInfo


@dataclass
class DecodeBackend:
    name: str
    args: list[str]
    description: str


def _backends(codec: str) -> list[DecodeBackend]:
    """Decode paths to try, fastest first, for this platform and codec.

    Every candidate is probed by actually decoding a frame, so listing a backend
    that does not exist on the machine costs one failed probe rather than a
    crash. That is what lets one list cover Windows and macOS.
    """
    out: list[DecodeBackend] = []
    if sys.platform == "darwin":
        # Apple Silicon has a dedicated media engine reached through
        # VideoToolbox. It decodes HEVC and H.264 -- including the 10-bit HDR an
        # iPhone records -- without touching the GPU cores, so it runs alongside
        # the MPS fit rather than competing with it.
        out.append(DecodeBackend(
            "videotoolbox", ["-hwaccel", "videotoolbox"],
            "Apple VideoToolbox media engine (fastest on Apple Silicon)",
        ))
    else:
        cuvid = {"hevc": "hevc_cuvid", "h264": "h264_cuvid",
                 "av1": "av1_cuvid"}.get(codec)
        if cuvid:
            out.append(DecodeBackend(
                "nvdec-cuvid", ["-hwaccel", "cuda", "-c:v", cuvid],
                "NVDEC dedicated decoder block (fastest)",
            ))
        out.append(DecodeBackend(
            "nvdec-generic", ["-hwaccel", "cuda"],
            "NVDEC via generic CUDA hwaccel",
        ))
        if sys.platform.startswith("win"):
            out.append(DecodeBackend(
                "d3d11va", ["-hwaccel", "d3d11va"],
                "Direct3D 11 hardware decode (any Windows GPU)",
            ))
        else:
            out.append(DecodeBackend(
                "vaapi", ["-hwaccel", "vaapi"],
                "VA-API hardware decode (Linux)",
            ))
    out.append(DecodeBackend(
        "cpu", ["-threads", "0"],
        "Multithreaded CPU decode (fallback)",
    ))
    return out


def _probe_backend(path: str, be: DecodeBackend) -> bool:
    """Decode a single frame to see whether this backend actually works here."""
    cmd = [ffmpeg(), "-v", "error", *be.args, "-i", path,
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    try:
        r = _run(cmd, capture_output=True, timeout=60)
        return r.returncode == 0 and len(r.stdout) > 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def select_backend(info: VideoInfo, force: str | None = None) -> DecodeBackend:
    """Pick the fastest decode path that works on this machine and this file."""
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg not found. A packaged build ships it alongside the "
            "application; otherwise install it with:\n"
            "  Windows:  winget install Gyan.FFmpeg\n"
            "  macOS:    brew install ffmpeg"
        )
    cands = _backends(info.codec)
    if force:
        cands = [b for b in cands if b.name == force] or cands
    for be in cands:
        if _probe_backend(info.path, be):
            return be
    raise RuntimeError(f"No working decode backend for {info.path}")


class FrameReader:
    """Streams grayscale frames as uint8 numpy arrays.

    Rotation metadata is applied by ffmpeg automatically (autorotate is on by
    default), so ``info.width``/``info.height`` already describe the frames you
    receive here and pixel coordinates match what you see in QuickTime.
    """

    def __init__(self, info: VideoInfo, backend: DecodeBackend | None = None,
                 scale: float = 1.0, color: bool = False):
        self.info = info
        self.backend = backend or select_backend(info)
        self.color = color
        self.channels = 3 if color else 1
        self.scale = scale
        self.width = int(round(info.width * scale)) // 2 * 2
        self.height = int(round(info.height * scale)) // 2 * 2
        self._frame_bytes = self.width * self.height * self.channels

    def _cmd(self, start_s: float | None, n_frames: int | None = None) -> list[str]:
        pre = ["-ss", f"{start_s:.6f}"] if start_s else []
        vf = [] if self.scale == 1.0 else ["-vf", f"scale={self.width}:{self.height}:flags=bilinear"]
        lim = ["-frames:v", str(n_frames)] if n_frames else []
        return [
            ffmpeg(), "-v", "error", "-nostdin",
            *self.backend.args, *pre, "-i", self.info.path, *vf, *lim,
            "-f", "rawvideo", "-pix_fmt", "bgr24" if self.color else "gray", "-",
        ]

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yields ``(index, timestamp_seconds, frame)``.

        The timestamp comes from the container's real PTS table, not from
        ``index / fps`` -- that is what keeps 30/60/120 Hz clips (and any VFR
        wobble within them) on a correct time axis.
        """
        shape = (self.height, self.width, 3) if self.color else (self.height, self.width)
        ts = self.info.timestamps
        t0 = float(ts[0]) if ts is not None and ts.size else 0.0
        proc = _popen(self._cmd(None), stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, bufsize=self._frame_bytes * 4)
        i = 0
        try:
            while True:
                buf = proc.stdout.read(self._frame_bytes)
                if len(buf) < self._frame_bytes:
                    break
                t = float(ts[i]) - t0 if (ts is not None and i < ts.size) else i / self.info.measured_fps
                yield i, t, np.frombuffer(buf, np.uint8).reshape(shape)
                i += 1
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            proc.wait()

    def read_at(self, t_seconds: float) -> np.ndarray | None:
        """Decode a single frame near ``t_seconds``. Used by the GUI scrubber.

        Seeking with ``-ss`` before ``-i`` lets the demuxer jump to a nearby
        keyframe instead of decoding the whole clip, which is what makes
        scrubbing feel immediate on a long 4K120 recording.
        """
        shape = (self.height, self.width, 3) if self.color else (self.height, self.width)
        try:
            r = _run(self._cmd(max(0.0, t_seconds), n_frames=1),
                     capture_output=True, timeout=60)
        except (subprocess.TimeoutExpired, OSError):
            return None
        if len(r.stdout) < self._frame_bytes:
            return None
        return np.frombuffer(r.stdout[: self._frame_bytes], np.uint8).reshape(shape).copy()

    def _keyframes(self, limit_s: float = 90.0) -> np.ndarray:
        """Decode only this clip's keyframes, in one pass."""
        shape = (self.height, self.width, 3) if self.color else (self.height, self.width)
        cmd = [ffmpeg(), "-v", "error", "-nostdin", "-skip_frame", "nokey",
               *self.backend.args, "-i", self.info.path,
               *([] if self.scale == 1.0 else
                 ["-vf", f"scale={self.width}:{self.height}:flags=bilinear"]),
               "-vsync", "0", "-f", "rawvideo",
               "-pix_fmt", "bgr24" if self.color else "gray", "-"]
        try:
            r = _run(cmd, capture_output=True, timeout=limit_s)
        except (subprocess.TimeoutExpired, OSError):
            return np.empty((0, *shape), np.uint8)
        count = len(r.stdout) // self._frame_bytes
        if count == 0:
            return np.empty((0, *shape), np.uint8)
        buf = np.frombuffer(r.stdout[: count * self._frame_bytes], np.uint8)
        return buf.reshape(count, *shape).copy()

    def sample(self, n: int) -> np.ndarray:
        """Grab ``n`` frames spread across the clip.

        By seeking to each one, not by streaming the whole file. The streaming
        version decoded every frame and discarded all but ``n`` of them, so
        opening a clip cost a full decode pass -- ten seconds for a 30 s 480p
        clip, and minutes for the 4K120 recordings this is built for. Seeking
        costs one keyframe jump per sample and is independent of clip length.

        Falls back to streaming if seeking yields nothing, which happens on a
        file with a broken index.
        """
        n = max(int(n), 1)

        # Keyframes first. They are spread across the clip by construction, and
        # decoding only them skips all inter-frame reconstruction: 35 frames in
        # 1.1 s against 16.7 s for a full pass on the reference clip, and the gap
        # widens with length because keyframe count grows with duration, not
        # with frame count. For a background plate or a color estimate, "some
        # frames spread over the clip" is exactly the requirement.
        kf = self._keyframes()
        if len(kf) >= max(8, n // 3):
            if len(kf) > n:
                kf = kf[np.linspace(0, len(kf) - 1, n).astype(int)]
            return kf

        dur = self.info.duration_s or (self.info.n_frames /
                                       max(self.info.measured_fps, 1e-6))
        if dur > 0:
            # The last timestamp often lands past the final frame and decodes
            # nothing, so the range is inset slightly.
            times = np.linspace(0.0, dur * 0.995, n)
            out = [f for f in (self.read_at(float(t)) for t in times) if f is not None]
            if out:
                return np.stack(out)

        idx = np.unique(np.linspace(0, max(self.info.n_frames - 1, 0), n).astype(int))
        want = set(int(x) for x in idx)
        out = [f for i, _, f in self if i in want]
        if out:
            return np.stack(out)
        tail = (3,) if self.color else ()
        return np.empty((0, self.height, self.width, *tail), np.uint8)
