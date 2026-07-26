"""Short notification sounds, played without adding a dependency.

A run can take twenty minutes on 4K footage, which is long enough to walk away
from. The point of a sound is that you learn the outcome without coming back to
read the log -- so success and failure have to be *distinguishable* by ear, not
merely audible. ``done`` rises, ``stopped`` falls; that is the whole vocabulary,
and it survives being heard from another room.

Playing audio at all is the awkward part. The obvious route is Qt Multimedia,
but ``launcher/robotrack.spec`` deliberately excludes ``PySide6.QtMultimedia``
because it drags a media backend and its helper processes into the bundle for
well over a hundred megabytes. Paying that to play a half-second chime is a poor
trade, so each platform's own mechanism is used instead:

* Windows -- ``winsound``, in the standard library since forever, plays a WAV
  asynchronously from a file path with no third-party anything.
* macOS -- ``afplay``, part of the base system, spawned detached.
* Linux -- ``paplay`` or ``aplay`` if either happens to exist.

Every path is best-effort. A machine with no audio device, a locked-down sound
daemon, or a missing asset must not take an analysis run down with it, so every
failure is swallowed. A missing sound is a missing sound; it is never an error
worth interrupting the user for.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path


def asset(name: str) -> Path:
    return Path(__file__).resolve().parent / "assets" / name


def _play_blocking(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            import winsound
            # SND_FILENAME reads from disk; SND_ASYNC returns immediately, which
            # matters because this may be called from the GUI thread.
            winsound.PlaySound(str(path),
                               winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        if sys.platform == "darwin":
            player = ["/usr/bin/afplay", str(path)]
        else:
            exe = shutil.which("paplay") or shutil.which("aplay")
            if not exe:
                return
            player = [exe, str(path)]
        # start_new_session detaches the player, so it is not killed when the
        # analysis worker thread finishes or the window closes mid-chime.
        subprocess.Popen(player, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        pass


def play(name: str, enabled: bool = True) -> None:
    """Play ``assets/<name>.wav``. Never raises, never blocks.

    ``enabled`` is threaded through rather than checked by the caller so that
    turning sounds off is one setting read in one place.
    """
    if not enabled or os.environ.get("ROBOTRACK_NO_SOUND"):
        return
    p = asset(f"{name}.wav")
    if not p.exists():
        return
    # winsound with SND_ASYNC already returns at once, but afplay does not, and
    # a thread costs nothing next to the run that just finished.
    threading.Thread(target=_play_blocking, args=(p,), daemon=True).start()


def finished(enabled: bool = True) -> None:
    """Rising two-note chime: the run produced results."""
    play("done", enabled)


def stopped(enabled: bool = True) -> None:
    """Falling two-note tone: the run was aborted, or it failed."""
    play("stopped", enabled)
