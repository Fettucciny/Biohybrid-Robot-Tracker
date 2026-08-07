"""Device selection and small GPU primitives shared by the rest of the package."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Device:
    torch_device: torch.device
    name: str
    cuda: bool
    total_mem_gb: float
    kind: str = "cpu"          # "cuda" | "mps" | "cpu"

    @property
    def accelerated(self) -> bool:
        """True on any GPU backend. ``cuda`` stays for CUDA-only decisions."""
        return self.kind in ("cuda", "mps")

    def __str__(self) -> str:
        label = {"cuda": "CUDA", "mps": "Apple Silicon (MPS)"}.get(self.kind, "CPU")
        mem = f", {self.total_mem_gb:.1f} GB" if self.total_mem_gb else ""
        return f"{label}: {self.name}{mem}"


def get_device(prefer_gpu: bool = True) -> Device:
    """The fastest backend available, preferring CUDA, then Apple Silicon.

    On an M-series Mac the GPU is on the same die as the CPU and shares its
    memory, so MPS is the accelerator even though there is no discrete card.
    It is not a drop-in CUDA: a handful of operations this pipeline uses are
    unimplemented there, and those fall back individually rather than pushing
    the whole run onto the CPU -- see ``otsu_threshold``.
    """
    if not prefer_gpu:
        return Device(torch.device("cpu"), platform.processor() or "CPU", False, 0.0, "cpu")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return Device(torch.device("cuda:0"), props.name, True,
                      props.total_memory / 1024 ** 3, "cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        name = platform.processor() or "Apple Silicon"
        if platform.system() == "Darwin":
            try:
                out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip():
                    name = out.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
        # Unified memory: the GPU can address system RAM, so there is no separate
        # figure to report the way there is for a discrete card.
        return Device(torch.device("mps"), name, False, 0.0, "mps")
    return Device(torch.device("cpu"), platform.processor() or "CPU", False, 0.0, "cpu")


_TRUST_CACHE: dict[str, tuple[bool, str]] = {}


def verify_device(dev: Device) -> tuple[bool, str]:
    """Check that this backend computes what the fitter needs it to.

    Why this exists rather than trusting the driver: the whole tracker rests on
    ``grid_sample`` and on autograd through it, and a backend that gets either
    subtly wrong does not raise. It returns numbers, the optimizer converges on
    them, and the result is a confident fit on the wrong part of the picture --
    indistinguishable from a hard clip, and reported as "it tracks on Windows
    but wanders on the Mac". There is no error to look for because nothing
    errors.

    So: run one small problem whose answer is known, on this device and on the
    CPU, and compare. It costs about a millisecond, once per session.

    Returns ``(trustworthy, note)``. This function never falls back on its own,
    because silently moving work to the CPU without saying so is how a
    five-minute run becomes an hour with no explanation. The caller decides,
    and tells the user.
    """
    if dev.kind == "cpu":
        return True, ""
    cached = _TRUST_CACHE.get(dev.kind)
    if cached is not None:
        return cached

    result: tuple[bool, str] = (True, "")
    try:
        g = torch.Generator().manual_seed(11)
        img = torch.rand(1, 1, 48, 64, generator=g)
        pts = torch.rand(6, 40, 2, generator=g) * 2.0 - 1.0

        def probe(device):
            im = img.to(device)
            p = pts.to(device).clone().requires_grad_(True)
            out = F.grid_sample(im, p.reshape(1, -1, 1, 2), mode="bilinear",
                                padding_mode="border", align_corners=True)
            out = out.reshape(6, 40)
            (out * out).sum().backward()
            return out.detach().cpu(), p.grad.detach().cpu()

        ref_v, ref_g = probe(torch.device("cpu"))
        got_v, got_g = probe(dev.torch_device)
        # Loose thresholds, because a different backend is entitled to a
        # different rounding order. This is looking for wrong answers, not for
        # the last bit.
        dv = float((ref_v - got_v).abs().max())
        dg = float((ref_g - got_g).abs().max())
        if not (dv < 2e-3 and dg < 2e-2):
            result = (False, f"image sampling differs from the CPU by {dv:.3g} "
                             f"(gradient {dg:.3g})")
    except Exception as exc:                # noqa: BLE001 -- any failure is a failure
        result = (False, f"{type(exc).__name__}: {exc}")

    _TRUST_CACHE[dev.kind] = result
    return result


def _otsu_cpu(flat: torch.Tensor, bins: int) -> float:
    lo, hi = float(flat.min()), float(flat.max())
    if hi <= lo:
        return hi
    hist = torch.histc(flat, bins=bins, min=lo, max=hi)
    p = hist / hist.sum()
    centers = torch.linspace(lo, hi, bins)
    w0 = torch.cumsum(p, 0)
    w1 = 1.0 - w0
    m0 = torch.cumsum(p * centers, 0) / w0.clamp_min(1e-12)
    mt = (p * centers).sum()
    m1 = (mt - torch.cumsum(p * centers, 0)) / w1.clamp_min(1e-12)
    return float(centers[int(torch.argmax(w0 * w1 * (m0 - m1) ** 2))])


def dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    """Binary/grayscale dilation as a max-pool. x is (H,W) or (N,1,H,W) float."""
    single = x.dim() == 2
    t = x[None, None] if single else x
    t = F.max_pool2d(t, k, stride=1, padding=k // 2)
    return t[0, 0] if single else t


def erode(x: torch.Tensor, k: int) -> torch.Tensor:
    return -dilate(-x, k)


def opening(x: torch.Tensor, k: int) -> torch.Tensor:
    """Removes speckle smaller than k without shrinking the object."""
    return dilate(erode(x, k), k)


def closing(x: torch.Tensor, k: int) -> torch.Tensor:
    """Fills pinholes and hairline gaps in the mask."""
    return erode(dilate(x, k), k)


def otsu_threshold(x: torch.Tensor, bins: int = 256) -> float:
    """Otsu's method, computed on-device so we never pull the frame back to host.

    ``torch.histc`` has no MPS implementation, so on Apple Silicon the histogram
    -- and only the histogram -- is computed on the CPU. That is one transfer of
    a single frame per call, against pushing every morphology and threshold
    operation back to the CPU if the whole function fell back.
    """
    flat = x.flatten().float()
    if flat.device.type == "mps":
        try:
            torch.histc(flat[:8], bins=4, min=0.0, max=1.0)
        except Exception:
            return _otsu_cpu(flat.detach().cpu(), bins)
    lo, hi = float(flat.min()), float(flat.max())
    if hi <= lo:
        return hi
    hist = torch.histc(flat, bins=bins, min=lo, max=hi)
    p = hist / hist.sum()
    centers = torch.linspace(lo, hi, bins, device=x.device)
    w0 = torch.cumsum(p, 0)
    w1 = 1.0 - w0
    m0 = torch.cumsum(p * centers, 0) / w0.clamp_min(1e-12)
    mt = (p * centers).sum()
    m1 = (mt - torch.cumsum(p * centers, 0)) / w1.clamp_min(1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(torch.argmax(between))])
