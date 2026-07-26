# One-shot Windows setup. Run in PowerShell from the repo folder.
# CUDA build of PyTorch must come from PyTorch's own index; the default
# PyPI wheel is CPU-only and will silently give you no GPU acceleration.

Write-Host "== ffmpeg (needed for NVDEC hardware decode) ==" -ForegroundColor Cyan
winget install --id Gyan.FFmpeg -e --accept-source-agreements

Write-Host "== python environment ==" -ForegroundColor Cyan
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Write-Host "== PyTorch with CUDA 12.8 ==" -ForegroundColor Cyan
pip install torch --index-url https://download.pytorch.org/whl/cu128

Write-Host "== robotrack ==" -ForegroundColor Cyan
pip install -e ".[gui]"

Write-Host "== verifying GPU ==" -ForegroundColor Cyan
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - check driver')"
ffmpeg -hide_banner -decoders 2>&1 | Select-String cuvid
