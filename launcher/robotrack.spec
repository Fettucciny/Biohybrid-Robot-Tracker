# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the self-contained robotrack application.

Produces ``dist/robotrack/robotrack.exe`` plus its dependencies -- Python,
PySide6, PyTorch with the CUDA runtime, OpenCV, and ffmpeg. Nothing needs to be
installed on the target machine.

Two deliberate choices:

**onedir, not onefile.** A onefile build of this stack is several gigabytes, and
onefile works by extracting the entire archive to a temp folder *on every
launch*. That turns a double-click into a long wait, and PyInstaller's own docs
note the temp folder leaks if the program crashes. onedir starts effectively
instantly. If you want a single file to hand around, wrap this folder with the
Inno Setup script (installer.iss) rather than switching to onefile -- that gives
one .exe to distribute *and* a fast-starting application.

**ffmpeg is bundled.** The pipeline shells out to ffmpeg for all decoding, so
without this the "self-contained" build would still fail on any machine that
happens not to have it on PATH. build_exe.ps1 downloads the binaries into
launcher/bin before invoking PyInstaller.
"""

import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

SPEC_DIR = Path(SPECPATH)
PROJECT = SPEC_DIR.parent
IS_MAC = sys.platform == "darwin"

# Read the version from the single source of truth so Info.plist agrees with
# the running application and with the update manifest.
_init = (PROJECT / "robotrack" / "__init__.py").read_text(encoding="utf-8")
VERSION = (re.search(r'__version__\s*=\s*["\']([^"\']+)', _init) or [None, "0.0.0"])[1]

datas, binaries, hiddenimports = [], [], []

# Packages that ship data files or import submodules dynamically, which
# PyInstaller's static analysis cannot follow on its own.
for pkg in ("torch", "cv2", "ezdxf", "scipy", "matplotlib", "pandas"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:            # a missing optional package is not fatal
        print(f"[spec] skipping {pkg}: {exc}")

hiddenimports += collect_submodules("robotrack")
hiddenimports += ["matplotlib.backends.backend_agg", "scipy.signal"]

# Splash artwork and icon. robotrack.splash resolves these relative to the
# package directory, so they must land inside it rather than at the bundle root.
ASSETS = PROJECT / "robotrack" / "assets"
if ASSETS.is_dir():
    for p in sorted(ASSETS.iterdir()):
        if p.is_file():
            datas.append((str(p), "robotrack/assets"))
else:
    print("[spec] WARNING: robotrack/assets is missing -- no splash will be shown.")

# Bundled ffmpeg/ffprobe. robotrack.ffmpeg searches _MEIPASS/bin first.
FFMPEG_DIR = SPEC_DIR / "bin"
for name in ("ffmpeg.exe", "ffprobe.exe", "ffmpeg", "ffprobe"):
    p = FFMPEG_DIR / name
    if p.exists():
        binaries.append((str(p), "bin"))

if not any(dest == "bin" for _, dest in binaries):
    print("[spec] WARNING: no ffmpeg binaries in launcher/bin -- "
          "this build will NOT be self-contained.")

# Qt modules this app never touches. Excluding them trims well over a gigabyte
# and avoids shipping WebEngine, which brings its own sandbox helper process.
EXCLUDES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick3D",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtMultimedia",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning",
    "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner",
    "tkinter", "PyQt5", "PyQt6", "IPython", "notebook", "jupyter",
    "pytest", "sphinx",
]

a = Analysis(
    [str(SPEC_DIR / "app.py")],
    pathex=[str(PROJECT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # onedir
    name="robotrack",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                      # UPX is known to corrupt some CUDA DLLs
    console=False,                  # failures surface as dialogs; see app.py
    icon=str(SPEC_DIR / ("robotrack.icns" if IS_MAC else "robotrack.ico")),
    target_arch=None,               # native: arm64 on Apple Silicon, x86_64 on Intel
    codesign_identity=None,         # ad-hoc; see build_macos.sh for the Gatekeeper note
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="robotrack",
)

# ---------------------------------------------------------------------------
# macOS application bundle.
#
# On Windows the COLLECT folder *is* the deliverable; on macOS a bare folder of
# binaries cannot be double-clicked, associated with an icon, or notarised. The
# BUNDLE step re-homes exactly the same files into robotrack.app, where the
# executable lands in Contents/MacOS and everything COLLECT gathered -- including
# launcher/bin/ffmpeg -- lands under Contents/Frameworks. robotrack.ffmpeg knows
# to look there.
#
# NSHighResolutionCapable is what stops the whole GUI from rendering as a blurry
# 1x image upscaled onto a Retina display, which is what an app without it gets.
# ---------------------------------------------------------------------------
if IS_MAC:
    app = BUNDLE(
        coll,
        name="robotrack.app",
        icon=str(SPEC_DIR / "robotrack.icns"),
        bundle_identifier="org.biohybridlab.robotrack",
        version=VERSION,
        info_plist={
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "LSApplicationCategoryType": "public.app-category.education",
            "NSHumanReadableCopyright": "Copyright (C) 2026 BioHybrid Lab. GPL-3.0-or-later.",
            # The app reads video files the user picks; on recent macOS a file
            # chosen through the standard panel needs no extra entitlement, but
            # a clip sitting in Movies/ or on a removable volume does.
            "NSDesktopFolderUsageDescription": "Open video recordings you select.",
            "NSDocumentsFolderUsageDescription": "Open video recordings you select.",
            "NSDownloadsFolderUsageDescription": "Open video recordings you select.",
            "NSRemovableVolumesUsageDescription": "Open video recordings from a connected drive.",
        },
    )
