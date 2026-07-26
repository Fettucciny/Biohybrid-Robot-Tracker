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
