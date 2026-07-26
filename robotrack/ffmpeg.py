"""Locating the ffmpeg/ffprobe binaries.

The whole pipeline shells out to ffmpeg for decoding, so a frozen build that
does not carry its own copy is not actually self-contained -- it works on the
build machine and fails on a clean one with a confusing "ffmpeg not found".

Resolution order:

1. ``ROBOTRACK_FFMPEG`` environment variable (explicit override wins).
2. Binaries bundled next to the frozen executable, or in PyInstaller's
   extraction directory.
3. The system PATH.

The bundled copy is preferred over PATH so a frozen build behaves identically
regardless of what happens to be installed on the machine.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

_EXT = ".exe" if os.name == "nt" else ""


def _search_dirs() -> list[Path]:
    dirs: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs += [Path(meipass), Path(meipass) / "bin"]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        dirs += [exe_dir, exe_dir / "bin", exe_dir / "_internal" / "bin"]
        # macOS .app layout: the binary sits in Contents/MacOS while bundled
        # data lands in Contents/Frameworks or Contents/Resources.
        for parent in exe_dir.parents:
            if parent.name == "Contents":
                dirs += [parent / "Frameworks" / "bin", parent / "Resources" / "bin",
                         parent / "MacOS" / "bin"]
                break
    here = Path(__file__).resolve().parent
    dirs += [here / "bin", here.parent / "bin", here.parent / "launcher" / "bin"]
    return dirs


@lru_cache(maxsize=8)
def tool(name: str) -> str:
    """Absolute path to ``ffmpeg``/``ffprobe``, or the bare name as a fallback."""
    override = os.environ.get("ROBOTRACK_FFMPEG")
    if override:
        cand = Path(override)
        if cand.is_dir():
            p = cand / f"{name}{_EXT}"
            if p.exists():
                return str(p)
        elif cand.exists() and name in cand.name:
            return str(cand)

    for d in _search_dirs():
        p = d / f"{name}{_EXT}"
        if p.exists():
            return str(p)

    found = shutil.which(name)
    return found or name


def ffmpeg() -> str:
    return tool("ffmpeg")


def ffprobe() -> str:
    return tool("ffprobe")


def available() -> bool:
    return Path(ffmpeg()).exists() or shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Silent subprocesses
#
# Every decode and every probe launches a console application. Windows gives a
# console application a console window, and because the GUI is built windowed
# (no console of its own) there is none to inherit -- so each call allocates a
# new one, flashes it on screen, and tears it down. Scrubbing a slider fires one
# per frame, which is a strobe light.
#
# CREATE_NO_WINDOW suppresses that. It is defined on modern Python for Windows
# only, hence the getattr; on other platforms the flags are simply zero.
# ---------------------------------------------------------------------------

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _silent_kwargs(kwargs: dict) -> dict:
    if os.name == "nt":
        kwargs.setdefault("creationflags", 0)
        kwargs["creationflags"] |= CREATE_NO_WINDOW
        # Belt and braces: a STARTUPINFO with SW_HIDE covers the Python builds
        # that predate CREATE_NO_WINDOW being exposed.
        if kwargs.get("startupinfo") is None:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0          # SW_HIDE
            kwargs["startupinfo"] = si
    return kwargs


def run(cmd: list[str], **kwargs):
    """``subprocess.run`` that never flashes a console window."""
    return subprocess.run(cmd, **_silent_kwargs(kwargs))


def popen(cmd: list[str], **kwargs) -> subprocess.Popen:
    """``subprocess.Popen`` that never flashes a console window."""
    return subprocess.Popen(cmd, **_silent_kwargs(kwargs))
