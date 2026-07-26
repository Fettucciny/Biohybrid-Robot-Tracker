# Build a self-contained robotrack.exe.
#
#   powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
#
# Produces  dist\robotrack\robotrack.exe  -- roughly 3-4 GB including PyTorch's
# CUDA runtime. Nothing needs installing on the target machine: Python, Qt,
# PyTorch, OpenCV and ffmpeg are all inside.
#
# Flags:
#   -SkipInstaller   don't build the single-file Inno Setup installer
#   -InstallerOnly   skip the freeze entirely and just re-run Inno Setup over an
#                    existing dist\robotrack (seconds, not tens of minutes)
#   -Clean           wipe build caches and the venv first
#   -Cpu             build against CPU-only PyTorch (small, slow; for testing)
#   -Iscc <path>     use a specific ISCC.exe instead of searching for one

[CmdletBinding()]
param(
    [switch]$SkipInstaller,
    [switch]$InstallerOnly,
    [switch]$Clean,
    [switch]$Cpu,
    [string]$Iscc
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

function Step($msg) { Write-Host "`n== $msg ==" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------
# Native-command helpers.
#
# PowerShell treats ANY stderr output from an external program as an error
# record, and with $ErrorActionPreference = "Stop" that becomes a terminating
# NativeCommandError. pip, PyInstaller and ffmpeg all write ordinary progress
# and warnings to stderr, so calling them directly under "Stop" aborts the
# build on output that is not an error at all.
#
# These helpers drop to "Continue" for the duration of the call and decide
# success from the process exit code, which is the only reliable signal.
# ---------------------------------------------------------------------------

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Arguments = @(),
        [switch]$Quiet,
        [switch]$AllowFail
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = 0
    try {
        if ($Quiet) {
            & $Exe @Arguments 2>&1 | Out-Null
        } else {
            & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    if (-not $AllowFail -and $code -ne 0) {
        throw "$Exe failed with exit code $code"
    }
    # Only emit the code when the caller asked for it. Returning it
    # unconditionally would scatter stray "0"s through the build log, since an
    # un-assigned function result goes straight to the output stream.
    if ($AllowFail) { return $code }
}

function Get-NativeOutput {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Arguments = @()
    )
    $out = @()
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = 0
    try {
        $out = & $Exe @Arguments 2>$null
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    return [pscustomobject]@{
        Output   = ($out -join "`n").Trim()
        ExitCode = $code
    }
}

function Find-BasePython {
    # The py launcher is preferred but is absent on Microsoft Store installs.
    foreach ($c in @(@{e = "py"; a = @("-3", "--version") },
                     @{e = "python"; a = @("--version") },
                     @{e = "python3"; a = @("--version") })) {
        $cmd = Get-Command $c.e -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $r = Get-NativeOutput -Exe $c.e -Arguments $c.a
        if ($r.ExitCode -eq 0) {
            Write-Host "using $($c.e)  ($($r.Output))"
            return $c
        }
    }
    throw "No Python 3 found on PATH. Install it from https://python.org and tick 'Add python.exe to PATH'."
}

function Get-ProjectVersion {
    # Single source of truth is robotrack/__init__.py. The installer, the app and
    # the update manifest must all agree, so the version is read rather than
    # duplicated into installer.iss.
    $initPy = Join-Path $root "robotrack\__init__.py"
    if (Test-Path $initPy) {
        $m = [regex]::Match((Get-Content $initPy -Raw), '__version__\s*=\s*["'']([^"'']+)["'']')
        if ($m.Success) { return $m.Groups[1].Value }
    }
    return "0.0.0"
}

function Find-Iscc {
    <#
    Locate the Inno Setup command-line compiler.

    The previous version of this script checked exactly two hardcoded paths under
    Program Files. That misses several perfectly normal installs:

      * winget can install per-user, which lands in %LOCALAPPDATA%\Programs
        instead of Program Files and needs no admin rights -- which is precisely
        why a lab machine would end up that way.
      * The directory is version-stamped ("Inno Setup 6"), so a major-version
        bump silently breaks a hardcoded path.
      * A portable or chocolatey install may only be on PATH.

    So: ask the registry where the installer said it put itself, then check PATH,
    then glob the usual roots. Report everything that was searched on failure,
    because "not found" with no detail is what made this hard to diagnose.
    #>
    if ($Iscc) {
        if (Test-Path $Iscc) { return (Resolve-Path $Iscc).Path }
        throw "-Iscc was given '$Iscc' but no file exists there."
    }

    $searched = New-Object System.Collections.Generic.List[string]

    # 1. Registry. Inno Setup's own uninstall entry records InstallLocation, and
    #    it also registers an "Inno Setup Script" file association.
    $regKeys = @(
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*_is1",
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*_is1",
        "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*_is1"
    )
    foreach ($k in $regKeys) {
        $searched.Add("registry $k") | Out-Null
        # A plain foreach, not ForEach-Object: `return` inside a pipeline
        # scriptblock exits only that iteration, so it would not stop the search
        # and its value would leak into the function's output stream.
        $entries = @(Get-ItemProperty $k -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -like "Inno Setup*" -and $_.InstallLocation })
        foreach ($e in $entries) {
            $cand = Join-Path $e.InstallLocation "ISCC.exe"
            if (Test-Path $cand) { return (Resolve-Path $cand).Path }
        }
    }

    # 2. Already on PATH (chocolatey, scoop, a manually extended PATH).
    $searched.Add("PATH") | Out-Null
    $onPath = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    # 3. Glob the usual roots. Version-agnostic, and includes the per-user
    #    location winget uses when it installs without elevation.
    $roots = @(
        ${env:ProgramFiles(x86)},
        $env:ProgramFiles,
        (Join-Path $env:LOCALAPPDATA "Programs"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"),
        "C:\"
    ) | Where-Object { $_ -and (Test-Path $_) }

    foreach ($r in $roots) {
        $pattern = Join-Path $r "Inno Setup*"
        $searched.Add($pattern) | Out-Null
        $hit = Get-ChildItem -Path $pattern -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "ISCC.exe" } |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1
        if ($hit) { return (Resolve-Path $hit).Path }
    }

    # 4. winget's package store nests one level deeper.
    $wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $wingetRoot) {
        $searched.Add("$wingetRoot (recursive)") | Out-Null
        $hit = Get-ChildItem -Path $wingetRoot -Filter "ISCC.exe" -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }

    Write-Host "Searched for ISCC.exe in:" -ForegroundColor Yellow
    $searched | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    return $null
}

function Build-Installer {
    Step "single-file installer"

    $distDir = Join-Path $root "dist\robotrack"
    if (-not (Test-Path (Join-Path $distDir "robotrack.exe"))) {
        throw "No frozen application at $distDir - run the full build first (without -InstallerOnly)."
    }

    $iscc = Find-Iscc
    if (-not $iscc) {
        Write-Host "`nInno Setup not found - skipping the single-file installer." -ForegroundColor Yellow
        Write-Host "  Install it with:   winget install JRSoftware.InnoSetup"
        Write-Host "  Then re-run just this step:   .\build_exe.ps1 -InstallerOnly"
        Write-Host "  Or point at it directly:      .\build_exe.ps1 -InstallerOnly -Iscc 'C:\path\to\ISCC.exe'"
        Write-Host "  Or zip dist\robotrack and distribute that."
        return
    }
    Write-Host "using $iscc"

    $version = Get-ProjectVersion
    $outFile = Join-Path $root "dist\robotrack-setup.exe"
    Remove-Item $outFile -Force -ErrorAction SilentlyContinue

    # Not -Quiet. Inno Setup's diagnostics are the only way to tell a missing
    # file from a bad script, and swallowing them is how a failed compile looks
    # identical to a skipped one.
    $code = Invoke-Native -Exe $iscc -Arguments @(
        "/DAppVersion=$version",
        (Join-Path $root "launcher\installer.iss")
    ) -AllowFail

    if ($code -ne 0) {
        throw "Inno Setup failed with exit code $code (see its output above)."
    }
    if (-not (Test-Path $outFile)) {
        throw "Inno Setup reported success but $outFile does not exist. Check OutputDir in launcher\installer.iss."
    }

    $mb = (Get-Item $outFile).Length / 1MB
    Write-Host ("installer written to {0}  ({1:N0} MB)" -f $outFile, $mb) -ForegroundColor Green
}

# ------------------------------------------------- installer-only fast path
if ($InstallerOnly) {
    Build-Installer
    Write-Host "`nDone." -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------- clean
if ($Clean) {
    Step "cleaning"
    foreach ($d in @("build", "dist", ".venv")) {
        Remove-Item (Join-Path $root $d) -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------- environment
Step "build environment"
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "creating .venv"
    $base = Find-BasePython
    $mkArgs = @()
    if ($base.e -eq "py") { $mkArgs += "-3" }
    $mkArgs += @("-m", "venv", $venv)
    Invoke-Native -Exe $base.e -Arguments $mkArgs
}
if (-not (Test-Path $py)) { throw "venv creation did not produce $py" }

Invoke-Native -Exe $py -Arguments @("-m", "pip", "install", "--upgrade", "pip") -Quiet

# Probe for CUDA-enabled torch. The probe script catches its own ImportError and
# always exits 0, so a missing torch is a normal answer rather than a failure.
$probe = @"
try:
    import torch
    print('cuda' if torch.version.cuda else 'cpu')
except Exception:
    print('none')
"@
$torchState = (Get-NativeOutput -Exe $py -Arguments @("-c", $probe)).Output
Write-Host "existing torch: $torchState"

$wantCpu = $Cpu.IsPresent
$needTorch = ($wantCpu -and $torchState -ne "cpu") -or (-not $wantCpu -and $torchState -ne "cuda")
if ($needTorch) {
    if ($wantCpu) {
        Step "installing PyTorch (CPU-only)"
        Invoke-Native -Exe $py -Arguments @("-m", "pip", "install", "--force-reinstall", "torch",
            "--index-url", "https://download.pytorch.org/whl/cpu")
    } else {
        Step "installing PyTorch with CUDA (large download, several minutes)"
        Invoke-Native -Exe $py -Arguments @("-m", "pip", "install", "--force-reinstall", "torch",
            "--index-url", "https://download.pytorch.org/whl/cu128")
    }
}

Step "installing robotrack and PyInstaller"
Invoke-Native -Exe $py -Arguments @("-m", "pip", "install", "-e", ".[gui]") -Quiet
Invoke-Native -Exe $py -Arguments @("-m", "pip", "install", "--upgrade", "pyinstaller") -Quiet

$report = (Get-NativeOutput -Exe $py -Arguments @("-c",
        "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu visible', torch.cuda.is_available())")).Output
Write-Host $report -ForegroundColor Yellow
if (-not $wantCpu -and $report -notmatch "gpu visible True") {
    Write-Host "WARNING: no CUDA GPU visible to PyTorch. The build will still work but run on CPU." -ForegroundColor Yellow
    Write-Host "         Check your NVIDIA driver, then re-run." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- ffmpeg
Step "ffmpeg binaries"
$bin = Join-Path $root "launcher\bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
if (-not (Test-Path (Join-Path $bin "ffmpeg.exe"))) {
    $zip = Join-Path $env:TEMP "ffmpeg-essentials.zip"
    $url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    Write-Host "downloading $url"
    $oldProgress = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"      # the progress bar makes this ~10x slower
    try {
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    } finally {
        $ProgressPreference = $oldProgress
    }
    $tmp = Join-Path $env:TEMP "ffmpeg-extract"
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    Get-ChildItem $tmp -Recurse -Include ffmpeg.exe, ffprobe.exe |
        ForEach-Object { Copy-Item $_.FullName -Destination $bin -Force }
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
if (-not (Test-Path (Join-Path $bin "ffmpeg.exe"))) {
    throw "Could not obtain ffmpeg.exe - the build would not be self-contained."
}
Write-Host "ffmpeg ready in $bin"

# ---------------------------------------------------------------- freeze
Step "freezing (several minutes; large amounts of output are normal)"
Invoke-Native -Exe $py -Arguments @(
    "-m", "PyInstaller", "--clean", "--noconfirm",
    "--distpath", (Join-Path $root "dist"),
    "--workpath", (Join-Path $root "build"),
    (Join-Path $root "launcher\robotrack.spec")
)

$exe = Join-Path $root "dist\robotrack\robotrack.exe"
if (-not (Test-Path $exe)) { throw "Build failed - no executable produced." }

$measured = Get-ChildItem (Split-Path $exe) -Recurse -File | Measure-Object -Property Length -Sum
$bytes = if ($measured.Sum) { $measured.Sum } else { 0 }
Write-Host ("`nBuilt {0}  (folder is {1:N2} GB)" -f $exe, ($bytes / 1GB)) -ForegroundColor Green

# ---------------------------------------------------------------- smoke test
Step "self-test"
# --selftest imports every dependency and runs the bundled ffmpeg, then exits.
# A windowed exe that dies on import shows the user nothing, so this is what
# turns a broken bundle into a visible build failure.
$code = Invoke-Native -Exe $exe -Arguments @("--selftest") -AllowFail
if ($code -eq 0) {
    Write-Host "self-test passed" -ForegroundColor Green
} else {
    Write-Host "self-test FAILED (exit $code)" -ForegroundColor Red
    $log = Join-Path (Split-Path $exe) "robotrack-selftest.log"
    if (Test-Path $log) { Get-Content $log | ForEach-Object { Write-Host "  $_" } }
    Write-Host "`nAdd any missing module to hiddenimports in launcher\robotrack.spec." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- installer
if (-not $SkipInstaller) {
    Build-Installer
}

Write-Host "`nDone." -ForegroundColor Green
