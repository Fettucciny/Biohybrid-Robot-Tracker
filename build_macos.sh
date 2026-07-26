#!/usr/bin/env bash
#
# Build a self-contained robotrack.app, and a .dmg to hand around.
#
#   ./build_macos.sh                 full build + DMG
#   ./build_macos.sh --skip-dmg      just the .app
#   ./build_macos.sh --dmg-only      re-wrap an existing dist/robotrack.app
#   ./build_macos.sh --clean         wipe build/, dist/ and .venv first
#
# This is the macOS counterpart to build_exe.ps1 and it is deliberately shaped
# the same way: one script, no arguments needed, and every dependency fetched
# rather than assumed.
#
# ---------------------------------------------------------------------------
# What is different from the Windows build, and why
# ---------------------------------------------------------------------------
#
# **No CUDA.** There is no NVIDIA GPU on an M-series Mac, so the CUDA wheel is
# neither available nor wanted. The default PyPI torch wheel for macOS carries
# the Metal Performance Shaders backend, which is what robotrack.gpu picks up as
# device kind "mps". That backend runs the segmentation and the chamfer fit on
# the M1's GPU cores. It is the reason this build is not simply "the CPU
# fallback on a Mac": on the reference clip the MPS path is several times faster
# than CPU, though still short of a discrete RTX-class card.
#
# **Decoding does not go through the GPU at all.** Apple Silicon has a separate
# media engine reached through VideoToolbox, and robotrack.decode asks for it
# first on Darwin. That is a genuinely good split: the media engine decodes 4K
# HEVC while the GPU cores are busy fitting, so the two stages do not contend.
#
# **The deliverable is a bundle, not a folder.** launcher/robotrack.spec grows a
# BUNDLE step on Darwin, producing dist/robotrack.app. ffmpeg lands in
# Contents/Frameworks/bin and robotrack.ffmpeg knows to look there.
#
# **Signing.** Apple Silicon refuses to execute an arm64 binary with no
# signature at all, so everything here is ad-hoc signed (`codesign -s -`). That
# is enough to *run*, but it is not a Developer ID signature and it is not
# notarised, so a Mac that downloads the DMG from the internet will still show
# the "cannot be opened because the developer cannot be verified" dialog. The
# README explains the one-time right-click-Open, and the xattr command that
# clears it outright. Paying for a Developer ID would remove that step; nothing
# else about the build would change.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_DMG=0
DMG_ONLY=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --skip-dmg) SKIP_DMG=1 ;;
    --dmg-only) DMG_ONLY=1 ;;
    --clean)    CLEAN=1 ;;
    -h|--help)  sed -n '3,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[36m== %s ==\033[0m\n' "$1"; }
warn() { printf '\033[33m%s\033[0m\n' "$1"; }
die()  { printf '\033[31m%s\033[0m\n' "$1" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "build_macos.sh only runs on macOS. Use build_exe.ps1 on Windows."

ARCH="$(uname -m)"                       # arm64 on Apple Silicon, x86_64 on Intel
FF_ARCH="arm64"; [[ "$ARCH" == "x86_64" ]] && FF_ARCH="amd64"

VERSION="$(python3 - <<'PY'
import pathlib, re
src = pathlib.Path("robotrack/__init__.py").read_text(encoding="utf-8")
m = re.search(r'__version__\s*=\s*["\']([^"\']+)', src)
print(m.group(1) if m else "0.0.0")
PY
)"

APP="dist/robotrack.app"
DMG="dist/robotrack-${VERSION}.dmg"

# ---------------------------------------------------------------------------
make_dmg() {
  step "disk image"
  [[ -d "$APP" ]] || die "No $APP - run the full build first (without --dmg-only)."

  # A DMG whose only content is the .app makes the user drag it somewhere by
  # hand. Staging it next to an /Applications symlink turns the window into the
  # familiar drag-to-install layout.
  local stage; stage="$(mktemp -d)"
  cp -R "$APP" "$stage/"
  ln -s /Applications "$stage/Applications"
  cp LICENSE "$stage/LICENSE.txt" 2>/dev/null || true

  rm -f "$DMG"
  hdiutil create -volname "robotrack $VERSION" -srcfolder "$stage" \
                 -ov -format UDZO -quiet "$DMG"
  rm -rf "$stage"
  [[ -f "$DMG" ]] || die "hdiutil reported success but $DMG does not exist."
  printf '\033[32mdisk image written to %s (%s)\033[0m\n' \
         "$DMG" "$(du -h "$DMG" | cut -f1)"
}

if [[ $DMG_ONLY -eq 1 ]]; then
  make_dmg
  echo; printf '\033[32mDone.\033[0m\n'
  exit 0
fi

# ---------------------------------------------------------------- clean
if [[ $CLEAN -eq 1 ]]; then
  step "cleaning"
  rm -rf build dist .venv
fi

# ---------------------------------------------------------------- environment
step "build environment"
PY_BASE="$(command -v python3.12 || command -v python3.11 || command -v python3)" \
  || die "No Python 3 found. Install it with:  brew install python@3.12"
echo "using $PY_BASE ($("$PY_BASE" --version))"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
[[ -x "$PY" ]] || "$PY_BASE" -m venv "$VENV"
[[ -x "$PY" ]] || die "venv creation did not produce $PY"

"$PY" -m pip install --upgrade pip >/dev/null

step "installing robotrack, PyTorch and PyInstaller"
# The default index is correct here: the macOS wheel is the MPS-capable one.
# There is no separate "torch-metal" package and no index-url to choose.
"$PY" -m pip install --upgrade torch >/dev/null
"$PY" -m pip install -e ".[gui]" >/dev/null
"$PY" -m pip install --upgrade pyinstaller >/dev/null

"$PY" - <<'PY'
import torch, platform
mps = torch.backends.mps.is_available()
print(f"torch {torch.__version__} | {platform.machine()} | MPS available: {mps}")
if not mps:
    print("\033[33mWARNING: Metal backend not available. The build will still work "
          "but every run falls back to CPU.\n"
          "         MPS needs macOS 12.3+ and an Apple Silicon or AMD GPU Mac.\033[0m")
PY

# ---------------------------------------------------------------- ffmpeg
step "ffmpeg binaries"
BIN="$ROOT/launcher/bin"
mkdir -p "$BIN"
if [[ ! -x "$BIN/ffmpeg" ]]; then
  # Static builds, so the bundle carries no dylib dependencies on the user's
  # machine. Homebrew's ffmpeg is dynamically linked against a tree of formulae
  # and copying just the binary out of it produces a bundle that dies on any Mac
  # without the same brew installs -- exactly the failure this avoids.
  base="https://ffmpeg.martin-riedl.de/redirect/latest/macos/${FF_ARCH}/release"
  tmp="$(mktemp -d)"
  for t in ffmpeg ffprobe; do
    echo "downloading $base/$t.zip"
    curl -fsSL -o "$tmp/$t.zip" "$base/$t.zip" \
      || die "Could not download $t.
Either set it up by hand:
  brew install ffmpeg && cp \$(command -v ffmpeg) $BIN/
(note the dynamic-linking caveat above), or drop static ffmpeg/ffprobe
binaries into $BIN and re-run."
    ditto -x -k "$tmp/$t.zip" "$tmp/x-$t"
    found="$(find "$tmp/x-$t" -type f -name "$t" -perm -u+x | head -1)"
    [[ -n "$found" ]] || die "no $t binary inside $t.zip"
    cp "$found" "$BIN/$t"
    chmod +x "$BIN/$t"
  done
  rm -rf "$tmp"
fi
[[ -x "$BIN/ffmpeg" ]] || die "Could not obtain ffmpeg - the build would not be self-contained."

# A downloaded binary carries the quarantine flag, which makes the frozen app
# fail to launch it with an opaque error rather than a permissions message.
xattr -dr com.apple.quarantine "$BIN" 2>/dev/null || true
codesign --force --sign - "$BIN/ffmpeg" "$BIN/ffprobe" 2>/dev/null || true

if ! "$BIN/ffmpeg" -hide_banner -hwaccels 2>/dev/null | grep -q videotoolbox; then
  warn "WARNING: this ffmpeg has no VideoToolbox support, so decoding will run on the CPU."
  warn "         Decoding is the bottleneck on 4K footage; replace launcher/bin/ffmpeg"
  warn "         with a build that lists videotoolbox in 'ffmpeg -hwaccels'."
else
  echo "ffmpeg ready in $BIN (VideoToolbox present)"
fi

# ---------------------------------------------------------------- freeze
step "freezing (several minutes; large amounts of output are normal)"
"$PY" -m PyInstaller --clean --noconfirm \
      --distpath "$ROOT/dist" --workpath "$ROOT/build" \
      "$ROOT/launcher/robotrack.spec"

[[ -d "$APP" ]] || die "Build failed - no $APP produced."
printf '\n\033[32mBuilt %s (%s)\033[0m\n' "$APP" "$(du -sh "$APP" | cut -f1)"

# ---------------------------------------------------------------- sign
step "ad-hoc signing"
# PyInstaller signs the pieces it writes, but the ffmpeg binaries and any dylib
# touched afterwards need re-signing, and the seal has to be applied
# outside-in last. Without this the app launches to an immediate SIGKILL on
# Apple Silicon, which shows the user nothing at all.
codesign --force --deep --sign - --timestamp=none "$APP" \
  || warn "codesign failed - the app may refuse to launch on Apple Silicon."
codesign --verify --deep --strict "$APP" 2>/dev/null \
  && echo "signature verifies (ad-hoc)" \
  || warn "signature does not verify; see the Signing note at the top of this script."

# ---------------------------------------------------------------- smoke test
step "self-test"
# Same contract as the Windows build: --selftest imports every dependency and
# runs the bundled ffmpeg, then exits. A windowed app that dies on import shows
# the user nothing, so this is what turns a broken bundle into a build failure.
if "$APP/Contents/MacOS/robotrack" --selftest; then
  printf '\033[32mself-test passed\033[0m\n'
else
  code=$?
  printf '\033[31mself-test FAILED (exit %s)\033[0m\n' "$code"
  log="$HOME/Library/Application Support/robotrack/robotrack-selftest.log"
  [[ -f "$log" ]] || log="$APP/Contents/MacOS/robotrack-selftest.log"
  [[ -f "$log" ]] && sed 's/^/  /' "$log"
  warn "Add any missing module to hiddenimports in launcher/robotrack.spec."
fi

# ---------------------------------------------------------------- dmg
[[ $SKIP_DMG -eq 1 ]] || make_dmg

echo; printf '\033[32mDone.\033[0m\n'
