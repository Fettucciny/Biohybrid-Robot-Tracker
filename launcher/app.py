"""Frozen-application entry point.

This is what PyInstaller freezes. It is deliberately thin: its only jobs beyond
calling into the GUI are to make failures *visible* and to verify that the
bundled ffmpeg came along for the ride.

A windowed (console-less) executable that dies during import shows the user
nothing at all -- the process simply vanishes. Since import-time failures are
exactly what packaging breaks, every one of them is caught and shown in a
dialog here.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Update overlay bootstrap
#
# This has to run *before* ``robotrack`` is imported for the first time, which
# means it cannot use ``robotrack.update`` to do it -- that would import the very
# package it is trying to redirect. So the few lines of state-file reading are
# duplicated here deliberately, in terms of nothing but the standard library.
# ``robotrack/update.py`` owns writing this file; this only ever reads it.
# ---------------------------------------------------------------------------

def _user_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "robotrack"


def _bootstrap_overlay() -> str | None:
    """Put an applied code update ahead of the bundled package on sys.path.

    Returns a warning to show the user if a previous update was quarantined.

    The ordering guard: if the last launch wrote an "unverified" marker and never
    cleared it, that launch died during import. A windowed executable that dies
    during import shows the user nothing at all, so the failure would look like
    the application simply refusing to open. Rather than retry it forever, the
    overlay is disabled here and the bundled version is used instead.
    """
    ud = _user_dir()
    state_path = ud / "update-state.json"
    marker = ud / "overlay-unverified"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    warning = None
    try:
        pending = json.loads(marker.read_text(encoding="utf-8"))
        version, attempts = str(pending.get("version", "?")), int(pending.get("attempts", 0))
    except (OSError, ValueError, TypeError):
        version, attempts = None, 0

    if version is not None and state.get("verified_version") != version:
        if attempts < 1:
            # First launch since the update landed. Record the attempt, then let
            # it run. Reaching the GUI clears the marker; dying before that leaves
            # attempts at 1, and the next launch takes the branch below.
            try:
                marker.write_text(json.dumps({"version": version, "attempts": 1}),
                                  encoding="utf-8")
            except OSError:
                pass
        else:
            state["active_overlay"] = ""
            state["quarantined"] = {"version": version,
                                    "reason": "did not finish starting after updating"}
            try:
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                marker.unlink()
            except OSError:
                pass
            warning = (f"Update {version} failed to start and has been disabled.\n\n"
                       "robotrack has reverted to the version it shipped with.")

    version = state.get("active_overlay")
    if not version:
        return warning
    overlay = ud / "overlay" / str(version)
    if (overlay / "robotrack" / "__init__.py").exists():
        sys.path.insert(0, str(overlay))
    return warning


def _ensure_importable() -> None:
    """When run from source (not frozen), put the project root on sys.path.

    In a frozen build ``robotrack`` is inside the bundle and this is a no-op.
    Running ``python launcher/app.py`` during development would otherwise fail
    to import the package that sits one directory up.
    """
    if getattr(sys, "frozen", False):
        return
    root = Path(__file__).resolve().parent.parent
    if (root / "robotrack" / "__init__.py").exists():
        sys.path.insert(0, str(root))


def _log_dir() -> Path:
    """Somewhere a log can actually be written, on either platform.

    The obvious choice -- next to the executable -- is wrong inside a macOS
    .app installed to /Applications, which is not user-writable; the write
    fails silently and the one artefact that would explain a launch failure
    never appears. Prefer the per-user data directory the updater already owns,
    and fall back to the executable's folder only if that is unavailable.
    """
    try:
        from robotrack.update import user_dir
        d = user_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:
        return Path(sys.executable).resolve().parent


def _crash(title: str, detail: str) -> int:
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("robotrack")
        box.setText(title)
        box.setDetailedText(detail)
        box.exec()
    except Exception:
        # Qt itself is broken; fall back to a log file next to the executable
        # so there is still a trace of what happened.
        try:
            (_log_dir() / "robotrack-error.log").write_text(f"{title}\n\n{detail}")
        except Exception:
            pass
    return 1


def selftest() -> int:
    """Verify the bundle without opening a window. Used by build_exe.ps1.

    Packaging failures are import failures, and a windowed executable that dies
    on import is indistinguishable from one that never launched. Running this
    at the end of the build turns that silent class of bug into a build error.
    """
    _ensure_importable()
    log: list[str] = []
    ok = True

    def check(name: str, fn):
        nonlocal ok
        try:
            log.append(f"[ok]   {name}: {fn()}")
        except Exception as exc:
            ok = False
            log.append(f"[FAIL] {name}: {exc!r}")

    check("numpy", lambda: __import__("numpy").__version__)
    check("opencv", lambda: __import__("cv2").__version__)
    check("scipy", lambda: __import__("scipy").__version__)
    check("pandas", lambda: __import__("pandas").__version__)
    check("ezdxf", lambda: __import__("ezdxf").__version__)
    check("matplotlib", lambda: __import__("matplotlib").__version__)
    check("PySide6", lambda: __import__("PySide6").__version__)

    def _torch():
        import torch
        return (f"{torch.__version__} cuda={torch.version.cuda} "
                f"available={torch.cuda.is_available()}")
    check("torch", _torch)

    # What the build will actually compute on. A Mac bundle reporting
    # cuda=None available=False above is normal and says nothing about whether
    # the Metal backend came through, which is the thing that matters there.
    def _device():
        from robotrack.gpu import get_device
        d = get_device()
        return f"{d.kind} -- {d.name} ({d.total_mem_gb:.1f} GB)"
    check("compute device", _device)

    def _ffmpeg():
        from robotrack.ffmpeg import ffmpeg as ff, run as silent_run
        r = silent_run([ff(), "-version"], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[:200])
        return f"{ff()} -> {r.stdout.splitlines()[0][:60]}"
    check("ffmpeg", _ffmpeg)

    check("robotrack.pipeline", lambda: __import__(
        "robotrack.pipeline", fromlist=["run"]).__name__)
    check("robotrack.settings", lambda: __import__(
        "robotrack.settings", fromlist=["load_settings"]).__name__)

    def _update():
        from robotrack import update
        return f"v{update.current_version()} overlay={update.active_overlay_path()}"
    check("robotrack.update", _update)
    check("robotrack.gui", lambda: __import__(
        "robotrack.gui", fromlist=["MainWindow"]).__name__)

    def _assets():
        from robotrack.splash import asset
        missing = [n for n in ("splash.png", "robotrack.ico") if not asset(n).exists()]
        if missing:
            raise FileNotFoundError(f"missing bundled assets: {missing}")
        return "splash.png, robotrack.ico"
    check("robotrack assets", _assets)

    text = "\n".join(log)
    print(text)
    try:
        (_log_dir() / "robotrack-selftest.log").write_text(text, encoding="utf-8")
    except Exception:
        pass
    return 0 if ok else 1


def main() -> int:
    # Overlay first: an applied code update must win over the bundled package,
    # and the decision has to be made before anything imports robotrack.
    overlay_warning = _bootstrap_overlay()
    _ensure_importable()
    if "--selftest" in sys.argv:
        return selftest()

    # Qt plugin resolution inside a frozen bundle is a common failure point;
    # point it at the bundled plugins explicitly rather than hoping.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        plugins = Path(meipass) / "PySide6" / "plugins"
        if plugins.is_dir():
            os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugins / "platforms"))

    try:
        from robotrack.ffmpeg import ffmpeg, available
    except Exception:
        return _crash("robotrack failed to start.", traceback.format_exc())

    if not available():
        return _crash(
            "ffmpeg is missing from this build.",
            "The packaged application should ship ffmpeg.exe and ffprobe.exe "
            "beside the executable.\n\nRe-run build_exe.ps1, which downloads "
            "them automatically, or install ffmpeg system-wide:\n\n"
            "    winget install Gyan.FFmpeg\n\n"
            f"Searched and resolved to: {ffmpeg()}")

    try:
        # The splash owns the QApplication so it can be on screen before the
        # multi-second PyTorch/Qt/OpenCV import begins -- which is the only
        # window in which a splash is worth anything.
        from robotrack.splash import run_with_splash
        return run_with_splash(sys.argv, startup_warning=overlay_warning)
    except Exception:
        return _crash("robotrack stopped unexpectedly.", traceback.format_exc())


if __name__ == "__main__":
    raise SystemExit(main())
