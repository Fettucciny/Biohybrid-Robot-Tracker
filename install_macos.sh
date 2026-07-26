#!/usr/bin/env bash
# One-shot macOS setup for running from source. Run from the repo folder:
#
#     ./install_macos.sh
#
# The counterpart to install_windows.ps1, with one important difference: there
# is no index-url to choose. The plain PyPI wheel is the right one on macOS --
# it carries the Metal Performance Shaders backend, which is what gives an
# M-series Mac GPU acceleration in the absence of a discrete card. Asking for a
# CUDA index here would just fail to resolve.
#
# For a double-clickable app instead of a source checkout, use ./build_macos.sh.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cyan() { printf '\n\033[36m== %s ==\033[0m\n' "$1"; }

[[ "$(uname -s)" == "Darwin" ]] || { echo "macOS only; use install_windows.ps1" >&2; exit 1; }

cyan "ffmpeg (needed for VideoToolbox hardware decode)"
if ! command -v ffmpeg >/dev/null; then
  command -v brew >/dev/null || {
    echo "Homebrew not found. Install it from https://brew.sh, then re-run." >&2; exit 1; }
  brew install ffmpeg
fi

cyan "python environment"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

cyan "PyTorch (Metal / MPS)"
pip install torch

cyan "robotrack"
pip install -e ".[gui]"

cyan "verifying acceleration"
python - <<'PY'
import platform, torch
print("machine       :", platform.machine())
print("torch         :", torch.__version__)
print("MPS available :", torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    print("  -> every run will fall back to CPU. MPS needs macOS 12.3+ on Apple Silicon.")
from robotrack.gpu import get_device
d = get_device()
print(f"device        : {d.kind} -- {d.name} ({d.total_mem_gb:.1f} GB)")
PY
ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -q videotoolbox \
  && echo "videotoolbox  : present" \
  || echo "videotoolbox  : MISSING -- decoding will run on the CPU, which is the bottleneck at 4K"

echo
echo "Done. Start the app with:  source .venv/bin/activate && robotrack-gui"
