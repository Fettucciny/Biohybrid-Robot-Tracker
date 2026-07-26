"""Video ingest: frame-rate detection, VFR diagnosis, per-frame timestamps.

Design note
-----------
Frame *index* is never used as a time axis anywhere downstream. Everything is
computed against real presentation timestamps in seconds. This is what makes
the 30/60/120 Hz handling automatic rather than a special case, and it is the
single thing that most often silently corrupts velocity and frequency numbers
in phone-recorded video.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from .ffmpeg import ffprobe, run as _run

# Rates an iPhone can actually produce. Detection snaps to the nearest of these
# so that 29.97 and 30.0 are reported as one mode, not two.
NOMINAL_RATES = (24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 100.0, 120.0, 240.0)


def _parse_keys_ilst(blob: bytes) -> dict[str, str]:
    """Decode a QuickTime metadata ``keys``/``ilst`` pair.

    The two atoms are parallel arrays: ``keys`` names the fields in order, and
    each ``ilst`` entry carries a 1-based index into that list followed by a
    ``data`` payload. Matching them by position is the only correct way to read
    it -- scanning for the nearest readable string after a key name returns
    whichever value happens to be first in the ilst, which is a different field.
    """
    out: dict[str, str] = {}
    ki, li = blob.find(b"keys"), blob.find(b"ilst")
    if ki < 0 or li < 0 or li < ki:
        return out

    # keys: [version/flags 4][count 4] then count x [size 4]['mdta'][name]
    pos = ki + 4 + 8
    names: list[str] = []
    while pos + 8 <= li:
        size = int.from_bytes(blob[pos:pos + 4], "big")
        if size < 8 or pos + size > li:
            break
        names.append(blob[pos + 8:pos + size].decode("ascii", "replace"))
        pos += size
    if not names:
        return out

    # ilst: entries of [size 4][index 4] then a 'data' atom
    pos = li + 4
    end = len(blob)
    while pos + 8 <= end:
        size = int.from_bytes(blob[pos:pos + 4], "big")
        idx = int.from_bytes(blob[pos + 4:pos + 8], "big")
        if size < 8 or pos + size > end:
            break
        body = blob[pos + 8:pos + size]
        if body[4:8] == b"data" and 1 <= idx <= len(names):
            dtype = int.from_bytes(body[8:12], "big") & 0xFFFFFF
            payload = body[16:]
            if dtype == 1:                       # UTF-8 text
                # The 4-byte locale is already consumed above, so the payload is
                # the text. Only NULs are stripped -- a regex that also ate
                # leading capitals turned "Apple" into "pple".
                out[names[idx - 1]] = payload.decode("utf-8", "replace").strip("\x00").strip()
            elif dtype in (21, 22) and payload:  # signed / unsigned integer
                out[names[idx - 1]] = str(int.from_bytes(
                    payload, "big", signed=(dtype == 21)))
        pos += size
    return out


def read_optics(path: str | Path) -> dict:
    """Lens metadata from the QuickTime moov atom, for provenance only.

    ffprobe does not surface Apple's ``com.apple.quicktime.camera.*`` keys, so
    the atom is parsed directly. This is recorded in run_info.json and never used
    for calibration: the file carries focal length and f-number but no subject
    distance, and without a working distance those cannot give a subject-side
    scale. Scale comes from the robot's own width instead.
    """
    want = ("camera.lens_model", "camera.focal_length.35mm_equivalent",
            "camera.lens_irisfnumber", "make", "model", "software")
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            # QuickTime recorders put moov at the end; a bounded tail read keeps
            # this cheap even on a multi-gigabyte 4K clip.
            f.seek(max(0, size - 8 * 1024 * 1024))
            blob = f.read()
    except OSError:
        return {}

    found = _parse_keys_ilst(blob)
    out = {}
    for k, v in found.items():
        short = k.replace("com.apple.quicktime.", "")
        if short in want and v:
            out[short] = v
    return out


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    n_frames: int
    nominal_fps: float          # snapped to NOMINAL_RATES, e.g. 120.0
    measured_fps: float         # from actual timestamps
    container_fps: float        # what the container claims
    duration_s: float
    is_vfr: bool
    vfr_jitter_pct: float       # std/mean of inter-frame intervals, percent
    rotation_deg: int
    codec: str
    pix_fmt: str
    optics: dict = field(default_factory=dict)   # lens metadata, provenance only
    timestamps: np.ndarray = field(repr=False, default=None)

    @property
    def dt(self) -> float:
        return 1.0 / self.measured_fps

    @property
    def nyquist_hz(self) -> float:
        """Highest contraction frequency this recording can resolve at all."""
        return self.measured_fps / 2.0

    @property
    def reliable_freq_hz(self) -> float:
        """Practical ceiling. Below Nyquist/2 amplitude estimates stay honest."""
        return self.measured_fps / 4.0

    def summary(self) -> str:
        lines = [
            f"{Path(self.path).name}: {self.width}x{self.height} {self.codec}/{self.pix_fmt}",
            f"  detected rate    : {self.nominal_fps:g} Hz "
            f"(measured {self.measured_fps:.3f} Hz, container claims {self.container_fps:.3f} Hz)",
            f"  frames / duration: {self.n_frames} / {self.duration_s:.2f} s",
            f"  frame interval   : {1000*self.dt:.3f} ms, jitter {self.vfr_jitter_pct:.2f}%",
            f"  resolvable freq  : up to {self.reliable_freq_hz:.1f} Hz reliably "
            f"({self.nyquist_hz:.1f} Hz Nyquist limit)",
        ]
        lens = self.optics.get("camera.lens_model")
        if lens:
            lines.append(f"  lens             : {lens}")
        if self.rotation_deg:
            lines.append(f"  rotation metadata: {self.rotation_deg} deg (will be baked in)")
        if self.is_vfr:
            lines.append(
                "  WARNING: variable frame rate detected. Real timestamps are being used, "
                "so results stay correct, but consider re-recording with more light."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("timestamps")
        d["nyquist_hz"] = self.nyquist_hz
        d["reliable_freq_hz"] = self.reliable_freq_hz
        return d


def _ffprobe(args: list[str]) -> dict:
    out = _run(
        [ffprobe(), "-v", "error", "-of", "json", *args],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def _parse_rate(s: str | None) -> float:
    if not s or s in ("0/0", "N/A"):
        return 0.0
    try:
        return float(Fraction(s))
    except (ZeroDivisionError, ValueError):
        return 0.0


def _snap(fps: float) -> float:
    """Snap a measured rate to the nearest rate a phone actually records at.

    Uses relative distance so 119.88 -> 120 and 29.97 -> 30, but an unusual
    rate like 90 Hz is not silently mangled into 100.
    """
    if fps <= 0:
        return 0.0
    best = min(NOMINAL_RATES, key=lambda r: abs(r - fps) / r)
    return best if abs(best - fps) / best < 0.06 else round(fps, 2)


def probe(path: str | Path, max_ts_frames: int = 20000) -> VideoInfo:
    """Inspect a video and return its true timing.

    Reads real per-frame presentation timestamps rather than trusting the
    container's declared rate.
    """
    path = str(path)
    meta = _ffprobe(["-select_streams", "v:0", "-show_streams", "-show_format", path])
    st = meta["streams"][0]

    rotation = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if rotation == 0 and "rotate" in st.get("tags", {}):
        rotation = int(float(st["tags"]["rotate"]))
    rotation = rotation % 360

    container_fps = _parse_rate(st.get("avg_frame_rate")) or _parse_rate(st.get("r_frame_rate"))

    # Real timestamps. Packet PTS is far cheaper than decoding every frame and
    # is identical to frame PTS for the all-intra / simple-GOP files phones make.
    pk = _ffprobe([
        "-select_streams", "v:0", "-show_entries", "packet=pts_time",
        "-read_intervals", f"%+#{max_ts_frames}", path,
    ])
    ts = np.array(
        sorted(float(p["pts_time"]) for p in pk.get("packets", []) if p.get("pts_time") not in (None, "N/A")),
        dtype=np.float64,
    )

    if ts.size >= 3:
        d = np.diff(ts)
        d = d[(d > 0) & (d < np.median(d) * 10)]   # drop pathological gaps
        measured_fps = 1.0 / float(np.median(d))
        jitter = float(np.std(d) / np.mean(d) * 100.0) if d.size else 0.0
    else:
        measured_fps, jitter = container_fps, 0.0

    n_frames = int(st.get("nb_frames") or ts.size or 0)
    duration = float(st.get("duration") or meta.get("format", {}).get("duration") or 0.0)
    if duration == 0.0 and ts.size:
        duration = float(ts[-1] - ts[0])

    w, h = int(st["width"]), int(st["height"])
    if rotation in (90, 270):
        w, h = h, w

    return VideoInfo(
        path=path, width=w, height=h, n_frames=n_frames,
        nominal_fps=_snap(measured_fps), measured_fps=measured_fps,
        container_fps=container_fps, duration_s=duration,
        # 2% is comfortably above codec timebase rounding and well below the
        # jitter a genuinely rate-varying clip shows.
        is_vfr=jitter > 2.0, vfr_jitter_pct=jitter,
        rotation_deg=rotation, codec=st.get("codec_name", "?"),
        pix_fmt=st.get("pix_fmt", "?"), optics=read_optics(path), timestamps=ts,
    )


def frames_for_duration(info: VideoInfo, milliseconds: float, odd: bool = True) -> int:
    """Convert a physical time window into a frame count for this clip.

    Every filter length in the pipeline is specified in milliseconds and passed
    through here, so a 30 Hz and a 120 Hz recording of the same robot get the
    same *physical* smoothing rather than the same number of frames.
    """
    n = int(round(milliseconds / 1000.0 * info.measured_fps))
    n = max(1, n)
    if odd and n % 2 == 0:
        n += 1
    return n
