# BioHybrid RoboTracker

GPU-accelerated tracking for muscle-driven soft robots. Windows and macOS, offline analysis of iPhone footage.

> The Python package, the executable and the release files are all still named
> `robotrack`. That is deliberate: it is the name every installed copy already
> has on `sys.path` and the name the code-patch updater ships a folder called,
> so renaming it would strand existing installs. Only the name you see changed.

Measures, per frame: **outline**, **width**, **length**, **centroid trajectory** and
**cumulative path length** — with a fit that survives partial occlusion and an
explicit confidence score so you can tell which numbers to trust.

---

## Install

### From a release

Every tagged version publishes three files on the repository's
[Releases](../../releases) page:

| file | what it is |
| --- | --- |
| `robotrack-<version>-setup.exe` | Windows installer, everything inside |
| `robotrack-<version>-macos.dmg` | macOS bundle, Apple Silicon |
| `robotrack-<version>-code.zip` | the small patch the **Update** button uses |

Download the one for your platform and run it. Neither needs Python, CUDA,
ffmpeg or anything else present beforehand.

**Windows.** The installer is per-user by default, so it needs no administrator
rights — which is the point on a managed university machine. SmartScreen will
show a blue *"Windows protected your PC"* panel the first time, because the
installer is not signed with a paid code-signing certificate: click **More
info**, then **Run anyway**. That warning is about the absence of a certificate,
not about anything found in the file, and it stops appearing once enough people
have run it.

**macOS.** See **[MACOS-INSTALL.md](MACOS-INSTALL.md)** for a step-by-step
version of this written for someone who has never used GitHub, Python or the
Terminal. The short form: open the `.dmg` and drag **BioHybrid RoboTracker** to
Applications. The first
launch will be refused with *"BioHybrid RoboTracker cannot be opened because the developer
cannot be verified"*: the build is ad-hoc signed, which is enough to execute on
Apple Silicon, but it is not notarised with a paid Apple Developer ID.
Right-click the app and choose **Open** once, then **Open** again in the dialog,
and macOS remembers the decision permanently. If you would rather clear it
outright:

```bash
xattr -dr com.apple.quarantine "/Applications/BioHybrid RoboTracker.app"
```

Nothing about the application changes either way; this is Gatekeeper's response
to any unsigned download.

### From a flash drive or a shared cloud folder

The installers are ordinary files with no network dependency, so this works with
no GitHub access at all — which is what you want for an offline rig, a
conference laptop, or a lab machine behind a restrictive proxy.

Put `robotrack-<version>-setup.exe` and `robotrack-<version>-macos.dmg` in a
folder on the drive or in the synced share. Anyone plugs it in, runs the one for
their platform, and is done. Keeping the matching `robotrack-<version>-code.zip`
alongside them is worth doing: it lets the same folder double as an update
channel (see [Updates](#updates)), so a machine that never reaches GitHub can
still be patched from the drive.

The download is large — 1–2 GB for Windows, less for macOS — because PyTorch's
CUDA runtime is inside it. That is a one-time cost; from then on updates are a
few hundred kilobytes.

---

## Build the application (Windows, .exe)

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1 -InstallerOnly   # re-wrap an existing build
```

Produces `dist\robotrack\robotrack.exe` — a self-contained application of
roughly 3–4 GB. Python, PySide6, PyTorch with the CUDA runtime, OpenCV and
**ffmpeg** are all inside it; nothing needs installing on the machine that runs
it. If Inno Setup is present the script also emits
`dist\robotrack-setup.exe`, a single installer file you can hand to a lab
member.

The build script downloads ffmpeg automatically and bundles it. This matters:
the pipeline shells out to ffmpeg for every decode, so a build without it works
on the machine that made it and fails confusingly everywhere else.

It is **onedir, not onefile**, on purpose. A onefile build of this stack would
re-extract several gigabytes to a temp folder on every launch. The installer is
the right way to get a single distributable file — it unpacks once, then starts
instantly forever after.

The build finishes with a self-test (`robotrack.exe --selftest`) that imports
every dependency and runs the bundled ffmpeg, writing
`robotrack-selftest.log`. Packaging bugs are import bugs, and a windowed
executable that dies on import shows the user nothing at all — this turns that
silent failure into a build error.

Requirements on the *build* machine only: Python 3.11+, and optionally
[Inno Setup](https://jrsoftware.org/isinfo.php)
(`winget install JRSoftware.InnoSetup`).

If the installer step is skipped even though Inno Setup is installed, the build
script could not find `ISCC.exe`. It now looks in the registry, on PATH, and
under both Program Files and the per-user location winget uses when it installs
without elevation — and prints every path it tried when it still comes up empty.
`-InstallerOnly` re-runs just that step in seconds rather than repeating the
freeze, and `-Iscc <path>` points it at a specific compiler. The version stamped
into the installer is read from `robotrack/__init__.py`, so it cannot drift from
what the app reports or what the update manifest says.

## Build the application (macOS, .app)

```bash
./build_macos.sh                 # full build + robotrack-<version>.dmg
./build_macos.sh --dmg-only      # re-wrap an existing dist/"BioHybrid RoboTracker.app"
```

Produces `dist/"BioHybrid RoboTracker.app"` and a drag-to-Applications disk image. Same
contract as the Windows build — Python, Qt, PyTorch and ffmpeg are all inside,
and it finishes with `--selftest` so a packaging bug becomes a build error
rather than an app that silently refuses to open.

Three things differ from the Windows build, and all three are about what
hardware is actually there:

**No CUDA, and none wanted.** An M-series Mac has no NVIDIA GPU, so the CUDA
wheel does not exist for it. The plain PyPI `torch` wheel is the correct one on
macOS: it carries Apple's Metal Performance Shaders backend, which `robotrack.gpu`
picks up as device kind `mps` and uses for segmentation and the chamfer fit.
This is not the CPU fallback wearing a different label — the fit really does run
on the GPU cores. It is still slower than a discrete RTX-class card, which is
what the hardware is.

**Decoding does not touch the GPU at all.** Apple Silicon puts video decode on a
separate media engine reached through VideoToolbox, and `robotrack.decode` asks
for it first on Darwin. That split is genuinely favorable here: the media engine
chews through 4K HEVC while the GPU cores are busy fitting, so the two stages do
not contend for the same silicon the way NVDEC-plus-CUDA very nearly does.

**The deliverable is a bundle.** `launcher/robotrack.spec` grows a `BUNDLE` step
on Darwin, so the same frozen files land in `robotrack.app` instead of a bare
folder — a folder of binaries cannot be double-clicked, carry an icon or be
signed. ffmpeg ends up in `Contents/Frameworks/bin`, and `robotrack.ffmpeg`
knows to look there. The build ad-hoc signs the result, which Apple Silicon
requires before it will execute an arm64 binary at all; see the Gatekeeper note
under [Install](#install) for what ad-hoc signing does *not* buy you.

Requirements on the *build* machine only: macOS 12+ and Python 3.11+
(`brew install python@3.12`). ffmpeg is downloaded as a static binary, not taken
from Homebrew — a Homebrew ffmpeg is dynamically linked against a tree of
formulae, and copying just the executable into the bundle produces something
that works on the build Mac and dies on every other one.

## Run from source instead

For development, skip the freeze:

```powershell
winget install Gyan.FFmpeg
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NOT plain `pip install torch`
pip install -e ".[gui]"
robotrack-gui
```

The PyTorch index URL matters. The default PyPI wheel is CPU-only and will run
without complaint — just slowly. Verify with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

On macOS the situation inverts — the plain wheel is the right one, and there is
no index to choose:

```bash
./install_macos.sh          # or, by hand:
python3 -m venv .venv && source .venv/bin/activate
pip install torch           # NOT an index-url; this wheel carries Metal/MPS
pip install -e ".[gui]"
robotrack-gui
```

Verify with:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Either way, the app's header shows the device it actually selected, so a run
that quietly fell back to CPU is visible rather than merely slow.

## Use

### Desktop app

Double-click `robotrack.exe` (or run `robotrack-gui` from a source checkout).

Open a video (and optionally a DXF), scrub to a representative frame, and check
the mask and fitted outline before committing to a full run. Every control
re-renders the current frame through the real pipeline code, so the preview is
exactly what the analysis will do. Tuning the threshold on one frame here takes
seconds; discovering it was wrong after processing a ten-minute 4K clip does not.

Live readout shows mask area, fitted width and length in pixels, and the fit
confidence for the displayed frame. The progress bar during a run reports frames
finished against the total, the current rate, and an estimate of time remaining.

### Command line

```powershell
BioHybrid RoboTracker myclip.MOV --dxf robot.dxf -o results
BioHybrid RoboTracker myclip.MOV --probe-only               # just report frame rate and timing
BioHybrid RoboTracker myclip.MOV                            # markerless, no CAD
BioHybrid RoboTracker x --dxf robot.dxf --list-outlines     # which curve in the drawing is the robot
```

Outputs in `results/`: `tracking.csv` (per-frame table), `summary.png` (time
series), `overlay.mp4` (fitted outline drawn on the video), `run_info.json`.

Useful flags: `--px-per-mm` (ruler calibration), `--scale 0.5` (downscale 4K for
speed), `--tau-px` (robust kernel width), `--restarts` (multi-start count),
`--smooth-ms` (filter window in physical time), `--dxf-loop` (which outline in the
drawing), `--cpu`.

---

## How the three requested features work

### 1. Frame-rate detection and matched time resolution

`ingest.py` reads real per-frame presentation timestamps with `ffprobe` rather
than trusting the container's declared rate, snaps the measured rate to the
nearest rate a phone actually records at (24/25/30/60/120/240), and reports
jitter so variable-frame-rate clips are flagged.

Frame *index* is never used as a time axis anywhere downstream. Every filter
window in the pipeline is specified in **milliseconds** and converted through
the measured rate, so a 30 Hz and a 120 Hz recording of the same robot get the
same physical smoothing and produce comparable numbers. The same applies to the
temporal prior in the fitter, which is expressed as a strain *rate* per second.
Nyquist and practical (fs/4) frequency ceilings are reported, and the analysis
warns when a detected contraction frequency is too close to them.

### 2. Imperfect matching under occlusion

Three mechanisms, each covering a failure the others miss:

**Segmentation.** A static obstacle is part of the median background plate, so
it removes robot pixels rather than being mistaken for robot. When an obstacle
bisects the body, fragments are regrouped by a *spatial* rule — a fragment is
kept if it clears the noise floor and sits within about one body length of the
main blob, where body length is learned from the least-occluded frames in the
clip. An area-ratio rule instead discards the small sliver that pokes past an
obstacle, and that sliver is often the only thing pinning down the far end.

**Bounded robust kernel.** The chamfer cost uses Geman-McClure,
`ρ(d) = d²/(d²+τ²)`. A template point stranded behind an obstacle contributes at
most 1 and its *gradient goes to zero*, so occluded points stop pulling on the
solution instead of dragging the fit off the robot. τ is annealed geometrically
from wide to narrow (graduated non-convexity) so the search keeps a large basin
of attraction early and sub-pixel precision at convergence.

**Asymmetric coverage term.** Chamfer distance alone cannot distinguish
"correctly fitted" from "collapsed onto a subset of the edges" — both put
template points on real edges. So the cost also asks the reverse question: *is
there observed robot outside my fitted outline?* Observed pixels are
inverse-transformed into template space and read against a precomputed template
signed-distance field. This is one-sided by construction, which is the point:
occlusion only ever removes pixels, so it can never trigger the penalty, but a
collapsed fit always does. In testing this cut length error from 15.9% to 5.6%.

Confidence per frame is the product of edge-inlier fraction, containment, and
coverage. Frames below threshold are gated out; short gaps are bridged linearly,
and gaps longer than `max_gap_ms` are left as NaN rather than fabricated.

### 3. CAD-driven tracking

`--dxf robot.dxf` loads a 2D DXF (`ezdxf`), flattens arcs/splines to a chord
tolerance of 0.05 mm, resamples the outer loop to evenly spaced points, and
rotates it so the long axis is local +y — which makes the two fitted scale
factors directly interpretable as width and length. Header `$INSUNITS` is
honoured, so a drawing in inches converts correctly.

The fit then solves for `(tx, ty, θ, sx, sy)`. Because the drawing is to scale,
`sx` and `sy` are px/mm and give an approximate pixel→mm calibration for free.
Supply `--px-per-mm` from a ruler in the plane of motion when you need the
absolute scale to be trustworthy — self-calibration assumes the drawing depicts
the *resting* size, and carried a ~3% bias in testing. Strain (relative to the
median size) needs no calibration at all and is the more reliable output.


### Choosing the outline in a real drawing

A production DXF is a sheet, not a bare outline: a page border, a title block,
dimension lines, and often three views of the part. It is also usually not
*closed* — the boundary is drawn as dozens of independent LINE and ARC entities
that form a loop only geometrically.

Both of those break "use the largest closed curve". `Legs.DXF` is the worked
example: the largest single entity in it is a 2.5 mm fillet arc, and the largest
closed loop is the 254 × 190.5 mm page border. Either answer produces a template
that fits confidently and measures the wrong thing.

So `cad.py` stitches segments into loops by matching endpoints, discards
rectangles that enclose most of the other geometry (sheet furniture is
recognized structurally, not by size), and ranks what is left. Where more than
one candidate survives — as with `Legs.DXF`, which offers a 63 × 26 mm and a
58 × 21 mm outline — the GUI shows an **Outline** chooser listing each with its
real millimeter dimensions, and the CLI has `--list-outlines`. Checking the pick
is immediate: the chosen outline is what the placement overlay draws on the video.

### Placing the target by hand

The starting pose is normally derived from the mask's own moments — centroid,
principal axis, extents. That is right when the robot is the only moving thing
in view and wrong when it is not: a second robot, a reflection, a bubble or a
tether all join the mask, the moments describe their union, and the optimiser
converges onto that with a perfectly respectable confidence score.

Tick **Place the outline by hand** and the real template is drawn over the frame
to drag onto the robot: the round handle sets the long axis and rotation, the
square handle sets width, the wheel scales both, Ctrl-drag rotates alone. Two
things make it cheap — the overlay is painted by Qt over a cached frame, so
dragging does not re-decode or re-fit, and a re-fit is triggered only on release.

It is a *seed*, not a constraint; the fit is free to move away from it, so
placement within a body width is enough. It is kept as a standing candidate
rather than used only on frame one, because the moment that needs it most is
recovery after a lost stretch — exactly when the tracker would otherwise re-seed
from the mask and jump to the decoy. It also resolves the head/tail ambiguity a
near-symmetric body gives the automatic seed no way to settle.

---

## Settings, configs and updates

### Everything is remembered

Every control, the last video, DXF, chosen outline, output folder, window
geometry and any manual placement are written to `settings.json` in the per-user
data directory and restored on launch. **Save config…** writes the same
dictionary to a `.rtcfg` file: that is what makes a run reproducible months
later, and what you keep next to the data. Loading one offers to reopen the clip
it was saved with. Older and newer files both load — the state is merged over the
defaults, so a missing key gets its default and an unknown key is ignored rather
than failing the launch.

### Updates

The **Update** button in the header checks a channel and installs what it finds.
Publish with:

```powershell
python tools/publish_update.py "C:\Users\you\Nextcloud2\robotrack-updates" --notes "..."
```

That zips the `robotrack` package, hashes it, and writes a manifest into the
folder. Point the app's channel at the same folder and everyone's copy sees it.

A **channel** is just a place to look, and four kinds work:

| channel string | reads from |
| --- | --- |
| `github:owner/repo` | the repository's GitHub Releases |
| `C:\Users\you\Nextcloud2\robotrack-updates` | a synced cloud folder |
| `\\lab-nas\share\robotrack-updates` | a UNC network share |
| `E:\robotrack-updates` | a flash drive |
| `https://example.org/robotrack-updates/` | any static web host |

Nothing else in the app knows which kind is in use, so moving the lab from a
Nextcloud folder to GitHub is one string in Settings and no republishing.

**`github:owner/repo` is the one to use once the repository exists.** It needs
no manifest at all — the app reads the Releases API and recognizes assets by
filename: `*-code.zip` is a patch, `*setup*.exe` and `*.dmg` are full installers
for their platforms. That is exactly what `.github/workflows/release.yml`
attaches, so pushing a tag is the whole publishing step. Every machine that has
ever installed BioHybrid RoboTracker, on either platform, then updates itself from the same
place with no shared drive between them.

The offline channels are not a lesser fallback — they are the answer for a rig
with no network. Copy the release's three files into a folder on a flash drive,
run `python tools/publish_update.py E:\robotrack-updates` once to write the
manifest beside them, and point those machines at the drive.

Updates come in two sizes, because the bundle is 3–4 GB and almost none of it
changes. A **code** update ships only BioHybrid RoboTracker's own Python (~75 KB) and is
extracted into a per-user overlay that `launcher/app.py` puts ahead of the
bundled package on `sys.path` — applied in about a second, with the frozen
interpreter and every heavy dependency untouched. A **full** update runs the Inno
Setup installer silently, and is needed only when a dependency, the Python
version or the bundled ffmpeg changes. The manifest records which, so that is
decided at publish time rather than guessed.

Two things guard against the obvious failure. Payloads are checked against the
SHA-256 in the manifest, which catches a truncated download or a half-finished
sync (it does not make an untrusted channel safe — whoever can write the manifest
can write the hash, so point this at a location only you can write to). And an
overlay is not trusted until a launch proves it imports: a marker is written when
it is applied and cleared once the window is up, so an update that dies during
import is quarantined on the next launch and the bundled version is used instead.
Without that, a bad patch would be indistinguishable from the application simply
refusing to open, which is exactly how a windowed executable fails.

`robotrack/__init__.py` resolves its public names lazily for the same reason: an
update is what fixes a broken CUDA DLL, so the updater must not need torch to
import before it can run.

### Putting the repository on GitHub, once

```powershell
cd "path\to\BioHybrid Tracker"

# The release workflow ships as tools\release-workflow.yml; GitHub only runs
# workflows from .github\workflows, so put it there once, before the first commit.
New-Item -ItemType Directory -Force .github\workflows | Out-Null
Copy-Item tools\release-workflow.yml .github\workflows\release.yml

git init -b main
git add .
git commit -m "BioHybrid RoboTracker: GPU tracking for muscle-driven soft robots"

# create the repo and push (needs the gh CLI: https://cli.github.com)
gh repo create BioHybrid RoboTracker --public --source=. --remote=origin --push
```

Without `gh`, make an empty repository on github.com first and then:

```bash
git remote add origin https://github.com/<you>/robotrack.git
git push -u origin main
```

Check `git status` before that first commit. `.gitignore` already excludes the
things that must not go in — `.venv/`, `build/`, `dist/`, `launcher/bin/`
(ffmpeg is ~200 MB and GitHub rejects any file over 100 MB), `updates/`, footage
and `TestRun/`. What should be there is source, the build scripts, the launcher,
`tests/`, `LICENSE` and this README: a few hundred kilobytes.

### Cutting a release

The version in `robotrack/__init__.py` is the single source of truth; the
installer, the `.app`'s `Info.plist`, the manifest and the release title are all
read from it. So:

```bash
# 1. bump __version__ in robotrack/__init__.py, then
git commit -am "v0.10.0"
git push

# 2. tag it
git tag v0.10.0
git push origin v0.10.0
```

That tag starts `.github/workflows/release.yml`, which builds the Windows
installer on a Windows runner, the `.app` and `.dmg` on an Apple Silicon runner,
packs the code patch, and attaches all three to a new Release. It takes roughly
half an hour, mostly PyTorch downloading.

The first job it runs does nothing but check that the tag matches
`__version__`, and fails the whole workflow if they disagree. That guard is
there because the failure it prevents is nasty and quiet: a release labeled
0.10.0 whose contents report 0.9.1 leaves every installed copy convinced it is
already up to date, and no one notices until someone asks why a fix never
arrived.

`workflow_dispatch` is enabled too, so you can run the same build from the
Actions tab without tagging, to check both platforms still compile.

The macOS runner is `macos-14` or later on purpose. PyInstaller freezes for the
architecture it runs on, so an Intel runner would produce an x86_64 bundle that
executes on an M1 only under Rosetta — and Rosetta gives no MPS at all, which
would quietly undo the entire reason for the macOS build.

---

## Working through a folder of clips

A session is a folder of recordings, not one recording, and most of the friction
in a day's analysis comes from that mismatch rather than from any single run.
Three things address it.

**Results go in their own folder.** Each run writes to
`<output folder>/<video name>/` rather than straight into the output folder.
Before, the second clip silently overwrote the first — same four filenames,
nothing in them naming the video they came from, so there was no way to notice
afterwards. Now a session accumulates.

**The clips are listed beside the plots.** Opening one video lists every video
in the same folder, sorted the way a file manager sorts: `IMG_2` before `IMG_10`,
which is the order a camera actually produced them in. Click any of them to load
it.

**A green dot means "already run".** The dot is not a stored list that could go
stale — it is read from the filesystem, and asks exactly one question: does
`<output folder>/<clip name>/` exist? Delete a results folder to redo a clip and
it goes hollow again immediately. What the dot cannot tell you is whether the
run was any *good*; that is what the fit confidence is for.

**Next video** below the plots walks down the list. It keeps everything that
describes the experiment — thresholds, the drawing, the force model, the output
folder — and clears everything that describes the clip: the fit, the plots and
any manual placement. Manual placement is deliberately not carried over. A pose
measured in one clip's frame is a *worse* starting guess in the next one than no
guess at all, because the fit would begin confidently in the wrong place.

### A sound when it finishes

A 4K run takes long enough to walk away from, so one finishes with a short
rising chime and one that fails or is aborted with a falling one. The two are
distinguishable from another room, which is the whole point — otherwise you have
to come back and read the log to find out which happened.

No dependency was added for this. The obvious route is Qt Multimedia, but the
frozen build deliberately excludes it — it pulls in a media backend and its
helper processes for well over a hundred megabytes, which is a poor trade for
half a second of audio. Instead each platform's own mechanism is used:
`winsound` on Windows, `afplay` on macOS, `paplay` or `aplay` on Linux. Every
path is best-effort and silent on failure; a machine with no audio device must
not take a run down with it. Turn it off by unsetting **sound** in the settings
file, or set `ROBOTRACK_NO_SOUND=1`.

### Small things that stop mistakes

The scroll wheel no longer changes parameter values. Scrolling down a long
sidebar used to pass the pointer over a dozen spin boxes, and whichever one it
was over took the wheel and changed — quietly, with nothing on screen to say a
number had moved. The wheel now scrolls the panel; a control still takes the
wheel once you have deliberately clicked into it.

The **Update** button breathes when a new version is available. The launch check
runs in the background and no longer opens the update window to tell you there
is nothing to do — which is what it used to do on the very first launch after an
update, reporting that you were up to date in the window you had just closed.

---

## Working with real footage

### Color, not brightness

The reference clip is orange limbs in a magenta medium. Measured on it:

| | medium | limbs | separation |
|---|---|---|---|
| luma | 148 | 174 | 26 levels |
| CIELAB a* | +41 | +24 | |
| CIELAB b* | −24 | +58 | **84 a\*b\* units** |

Brightness barely separates them, and the robot's pale interior reads *brighter*
than its limbs — which is why placing a luma threshold felt impossible. Color
separates them by more than three times as much.

So segmentation defaults to **color keying**: the densest cell of the a\*b\*
histogram is the medium (it fills most of every frame, so this needs no
assumption about what color it is), the far mode with real mass is the robot,
and the cut sits a fraction of the way between them. `Auto` measures the
separation in your clip and falls back to luma below 20 units; the header chip
says which it chose and the log says why.

Two consequences beyond the cleaner mask. There is **no background plate** in
color mode, so the assumption that the robot vacates every pixel for more than
half the clip disappears — a robot that mostly sits still no longer leaves a
ghost of itself in the median. And opening a clip got much faster, because the
plate was the slow part.

The cut is a fraction rather than Otsu deliberately. Otsu assumes two populations
of comparable size; the robot is a few percent of the frame, so it lands high and
slices the body in half — 62–67% of the mask in the largest fragment against
92–93% for the fractional cut.

### Fragment grouping needs an envelope

Regrouping fragments across an occlusion gap by proximity alone was tuned for a
luma mask, where stray foreground is rare. A color-keyed mask of a textured
medium has specks everywhere, and "within one body length of the main blob" then
sweeps up the whole frame — measured as a mask spanning all 438×778 px from 13
fragments, and a fitted length of 950 px on a 778 px frame. Fragments are now
accepted nearest-first and only while the union still fits inside a plausible
body envelope. Mean fit confidence on the reference clip went from **0.11 to
0.87**.

### Matching the drawing automatically

**Match drawing to video** segments a few frames from across the clip, fits every
candidate outline to each, and keeps whichever agrees best — then hands its pose
straight to the placement overlay. Geometry alone cannot say which curve in a
drawing is the part; the video can. On `Legs.DXF` it separated the true body
outline at 0.75 confidence from the runner-up at 0.46.

It judges outlines by how well they fit the mask, so run it once the mask looks
right. A best score below about 0.35 is reported as a warning, because that is a
segmentation problem wearing an outline problem's clothes.

### Speed

| | before | after |
|---|---|---|
| opening a clip | ~11 s | ~4 s |
| preview after a parameter change | 2–3 s | 0.8 s |
| re-showing a visited frame | 2–3 s | 0.04 s |

Three causes, three fixes. Sampling decoded the entire clip to keep 60 frames; it
now decodes only keyframes in one pass — 35 frames in 1.1 s against 16.7 s, and
the gap widens with clip length because keyframe count tracks duration rather
than frame count. Every preview rebuilt the `ShapeFitter`, rasterising a
signed-distance grid each time; it is now rebuilt only when something it depends
on changes. And decoded frames are cached, so stepping back and forth over a
contraction costs nothing.

The preview also runs a lighter fit than the analysis — 24 restarts against 64,
which lands within a pixel on a frame that is going to be re-measured properly
anyway. Runs always use the full settings.

### Playback

Space, or the transport under the scrubber, plays the clip with live
segmentation at 0.25× to 4×. It holds one sequential decode open rather than
seeking per frame, and skips shape fitting — that is the expensive stage, and the
fit returns the moment you pause. Frames are dropped rather than queued if the
machine cannot keep up, so the clock stays honest.

### No more console flashes

Every decode and probe launches a console application, and a windowed Qt process
has no console to inherit — so Windows allocated a new one, flashed it, and tore
it down, once per frame while scrubbing. All ffmpeg and ffprobe calls now go
through `ffmpeg.run` / `ffmpeg.popen`, which set `CREATE_NO_WINDOW`.


---

## Scale, micrometers and force

### The robot's own width is the ruler

The frame is rigid across its short axis while the long axis is what contracts,
so the width is a constant of known length present in every frame — a ruler
always in the plane of motion, always in focus, impossible to forget at capture
time. Paired with the width from the drawing it gives µm/px directly, per clip.

The assumption is checkable, so it is checked. Width CV is computed and reported;
on the reference clip it is **0.3%**, which is a rigid width. If it ever climbs,
that line says so rather than quietly biasing every micrometer in the output.

Width is therefore *not* plotted and not written to `tracking.csv` — it is the
instrument, not a result, and a flat trace beside the thing that varies reads as
a measurement that did nothing. Its median and CV go to `run_info.json`.

**No drawing? Type the width.** The ruler only needs the robot's true width in
millimeters, and the drawing is just the usual way to know it. The **True width**
field takes it directly, so markerless tracking calibrates too. Watch the width
CV afterwards: the fitted width from a DXF measured 0.3% on the reference clip,
while the markerless silhouette width measured 17.7% — same robot, and the CV is
what tells them apart.

### Getting a usable DXF out of SOLIDWORKS

The reference drawing is a **drawing sheet** exported at 5:1 — a page border, a
title block, three views, and every outline five times its true size. That is the
default outcome of `File > Save As > DXF` from a drawing, and it is why both the
scale field and the sheet-border rejection exist.

The clean route skips the sheet entirely: **right-click the flat top face in the
part and choose `Export to DXF/DWG`**. The face fixes the projection, geometry
comes out at true size, and there is no border or title block to filter. Set
`Tools > Options > Document Properties > Units` to MMGS first so the DXF header
records millimeters.

If you must go via a drawing, three things have to be right: sheet scale 1:1
(`Sheet Properties`), view scale 1:1 (`View Properties > Use custom scale`), and
`File > Save As > DXF > Options >` **`Scale output 1:1`** with the base scale
checked. Miss any one and the geometry is scaled by that factor.

Either way, the scale is recoverable after the fact: measure the real width,
enter it under **True width**, and press **Set scale from true width**. A drawing
at 5:1 and one drawn in centimetres are indistinguishable from inside the file,
so one measured dimension is what settles it. `Legs.DXF` resolves to exactly
0.2000.

**Drawing scale matters more than it looks.** `Legs.DXF` is at 5:1 detail scale:
it measures 26.25 × 63.00 mm as drawn against a true 5.25 × 12.60 mm, so it needs
**× 0.2**. Get that wrong and every micrometer is off by the same factor with
nothing else looking wrong. The Drawing scale field shows the resulting
millimeters as you type, so you can match it against the bench. At × 0.2 the
reference clip calibrates to **26.45 µm/px**.

Video metadata cannot supply this. The file does carry the optics — `iPhone 17
Pro Max back camera 6.765mm f/1.78`, 35 mm-equivalent 50, implying a 4.68 × 3.51
mm sensor and 10.69 µm per pixel *on the sensor* — but subject-side scale needs
magnification, and that needs a working distance Apple does not record. Between a
60 mm and a 300 mm standoff the answer spans 84 to 464 µm/px. The lens data is
parsed out of the QuickTime `keys`/`ilst` atoms (ffprobe does not surface it) and
recorded in `run_info.json` as provenance only.

### Force: a measured curve, or beam mechanics

Two methods, selectable under **Force**.

**Measured LUT.** A two-column calibration you performed on a rig:

```
Length (mm),Force (mN)
12.60,0.00
12.20,0.85
```

Units come from the header text — `Length (um)`, `Force (mN)`, `Load (gf)` are
all understood, column order does not matter, and an unlabelled file is treated
as mm/mN with the assumption written to the log rather than left silent. A
thousand-fold unit error is the kind that survives review because every number
still looks plausible. Between points the curve is interpolated linearly; beyond
either end it is **clamped, not extrapolated**, and the number of frames outside
is reported. A length–force curve has a real domain, and continuing its end slope
into lengths you never tested invents stiffness data.

**Beam model.** No calibration run at all — force from the robot's own geometry,
as in `SampleForce.m`. The muscle pulls the leg tips together, the legs rotate
like rigid links, and the beam's bending stiffness resists:

```
I     = t³·w / 12                     second moment of area
θ     = asin( (δ/2) / L_leg )         rotation of each leg
M     = 2·E·I·θ / L                   Euler–Bernoulli, end-rotation form
F     = M / l = 2·E·I·θ / (l·L)
```

where δ is the shortening from rest. Cvetkovic et al. (PNAS 2014) write the same
physics as `P = 8·E·I·δ_max/(l·L²)` in terms of the beam's transverse mid-span
deflection. The two are algebraically identical — substituting `δ_max = θL/4`
turns one into the other, checked numerically to **0.00%** across pull-ins from
0.05 to 1.5 mm. The end-rotation form is used because it takes the quantity a
camera can actually measure: video from above gives the change in leg separation,
not the out-of-plane bow of the beam.

With E in pascals and every length in millimeters, `E·I/(l·L)` is Pa·mm² = 10⁻⁶ N,
so **force comes out in micronewtons with no conversion factor** — the unit the
literature reports (395 µN active, 534–1147 µN passive), and why `SampleForce.m`
uses a peak prominence of 200 unscaled. Defaults are that file's values, and the
implementation reproduces it to 1e-13 µN.

What the model assumes, and where it will bite:

- **Force scales linearly with E**, and a cast hydrogel's modulus varies batch to
  batch and drifts in culture. It is the least certain number here by a wide
  margin. It is also a pure scale factor, so a force can be rescaled afterwards
  without re-running the tracking.
- **I goes as thickness cubed**, so a 10% error in a callipered thickness is a
  33% error in force. Measure the fabricated part, not the drawing.
- **Deflection is the tracked length change**, taken as the leg-tip pull-in. That
  holds when the tracked outline spans the legs, which is what the top-down view
  of these designs gives.
- **Resting length** defaults to the maximum over the clip, matching the MATLAB.
  That is the single most extreme sample in the recording, so one over-long frame
  sets the baseline for the whole clip — in testing, one bad frame moved the mean
  force by 830 µN. A robust option (median of the upper quartile) is offered; the
  value actually used is printed in the summary and recorded in `run_info.json`.

Running both against each other is worthwhile. Agreement is real evidence; a
systematic gap usually means the modulus is off, since force scales linearly
with it.

## The plot panel

A third column beside the video, live during a run and interactive after it.

Rows arrive frame by frame and are drawn at a fixed refresh rate — redrawing the
figure once per row would take longer than the analysis. Live values are raw,
before gating and smoothing; when the run finishes the gated, calibrated table
replaces them.

**Scroll zooms time, shift-scroll zooms the value axis** under the cursor,
double-click resets. Time is shared across panels, because comparing force
against strain at different time ranges would be actively misleading; a value
zoom applies only to the panel you are pointing at.

Whatever range is on screen when a run starts is written into `summary.png` and
recorded under `axis_ranges` in `run_info.json`, and the figure title says
`zoomed`. A figure cropped to the interesting three seconds is worth keeping; one
that silently reverted to the full clip is not.

**Run doubles as Abort.** During a run the button becomes Abort; it sets a flag
the pipeline checks between frames rather than killing the thread, since a torch
graph mid-backward and a half-written video are not things to interrupt at an
arbitrary instruction. An aborted run writes nothing, so there is never a partial
CSV claiming to be a complete one.

**Scroll zooms, right-drag pans, left-drag measures.** Scroll zooms time on
every panel, shift-scroll zooms one panel's value axis, right-drag pans, and
left-drag marks a stretch of time — which is measured the moment you release.
Double-click clears both. The pan anchor is held in pixels rather than data
coordinates; the data-coordinate version drifts, because once the first motion
moves the view the same cursor position means a different value and the plot
accelerates away.

The selected region reports, per panel:

| panel | measured |
|---|---|
| length | average Δ length, and the cycle count behind it |
| force | average Δ force in µN |
| movement | speed along the path *and* the net rate, both in mm/min |
| fit confidence | nothing — it describes the tracker, not the robot |

The delta is a **mean peak-to-trough swing, not max minus min**. Max minus min
reports the single largest excursion in the window, which is the noisiest sample
available and grows the longer you select; pairing successive turning points
gives the average contraction amplitude, which is what "delta length" is meant to
mean. Turning points need a prominence floor scaled to the trace's own range, or
ordinary measurement noise registers as thousands of tiny cycles. The cycle count
is printed beside each delta: two cycles is an anecdote, twenty is a measurement.

**Two speeds are reported, and the gap between them is the point.** The path
slope is regressed on the cumulative path; the net rate is straight-line, start
to finish, over the same window. Cumulative path only ever increases, so every
wobble of the centroid adds distance the robot never traveled — with 0.6 px of
centroid noise at 30 Hz the path slope measured **50.4 mm/min against a true 3.18
mm/min**. Net displacement cannot do that. On the reference clip the ratio is a
mild 1.2–1.3×; above about 2× it is flagged, because at that point the centroid
is wandering more than the robot is walking and the net rate is the one to quote.

The x and y curves are drawn for direction; a slope through either is a
component, not a speed.

Whatever is selected when a run starts is shaded and annotated on the exported
figure and recorded under `region_analysis` in `run_info.json`, so the number and
the picture it came from stay together.

The movement panel carries three curves: cumulative path, plus net x and net y.
Path says how far the robot traveled; x and y say where it went, and the two
differ whenever it doubles back. Fit confidence reads as a percentage.

The **View** control above the video switches between the photograph and the
color-distance surface the segmenter actually thresholds. A threshold problem is
obvious there and nearly invisible on the photograph. The tracked outline is drawn
at all times — while scrubbing, during playback, and on every sixth frame during
the analysis itself, so a tracker that lets go shows it long before the numbers do.

---

## Fitting the drawing's interior

The silhouette fixes position, rotation and the two scales, and says nothing
about the inside of the body — so a boundary that segmentation renders slightly
wrong has nothing to correct it. **Fit interior features** samples the closed
loops inside the outer boundary (`Legs.DXF` has two, at 74.5% and 27.8% of the
body area) as extra points, making the fit over-determined. That is the point: it
is what lets the threshold be tight.

Interior edges come from the **color-distance image, not the mask**. A binary
mask has exactly one edge — its silhouette — and an aggressively thresholded one
is a solid blob: measured on the reference clip, *zero* enclosed holes at any
morphology setting. Canny edges of the chroma surface, restricted to the mask's
neighbourhood so the dish rim cannot attract the template, recover what the
threshold discarded.

Only the silhouette carries the containment term. "Just outside this edge is
background" is true for the outer boundary and false for an internal edge with
material on both sides, so applying it there would push a correct fit away.
Confidence likewise stays silhouette-based, so turning features on never changes
what the confidence gate means; interior agreement is reported separately.

Measured on the reference clip, width CV (the calibration's own stability):

| color cut | silhouette only | + interior | interior points matched |
|---|---|---|---|
| 0.30 | 3.79% | 5.41% | 28% |
| 0.45 | 2.17% | **1.90%** | 19% |
| 0.60 | 2.58% | **1.90%** | 23% |

It helps at a tight cut and hurts at a loose one, which is the regime it was
built for — raise the cut and enable features together. Around 20% of interior
points find an edge on this drawing; that number is reported so you can tell
when a drawing's inner loops do not correspond to anything the camera resolves.

### A tight cut is not free

Thresholding harder shrinks the mask, and past a point the fit settles on a
*sub-region* of the robot rather than the robot. This failure looks excellent by
every other measure — at color cut 0.50 the reference clip reported **185/185
frames tracked and a width CV of 1.5%** while measuring one limb.

Proportions are what give it away, so they are now checked automatically: the
ratio of the two fitted scales is compared against the drawing's, and a
disagreement over 15% is reported as a warning. At cut 0.30 the clip reads 2.27
against 2.40 drawn; at 0.50 it reads 1.42, and says so.

---

## Validation

`tests/make_synthetic.py` generates a clip with exactly prescribed ground truth:
a tapered body contracting at 1.5 Hz (length ±12%, width ∓8%, as an
incompressible muscle would), translating along a curved path, passing behind a
static occluder that hides part of it. `tests/evaluate.py` scores the recovery.

The fixture is colored, not gray, and that is load-bearing. The analysis keys
on chroma, so a neutral-gray clip would exercise a path the software no longer
takes by default. The generator therefore reproduces the real separation — about
27 CIELAB units of luma against 73 of chroma — and makes the occluder a *shadow
of the medium* rather than a gray bar: same hue, 46 units darker. A luma
threshold sees that shadow as the largest object in the frame; the chroma key
does not see it at all, which is precisely the asymmetry the color path exists
to exploit. Its background texture is uneven illumination for the same reason.

Measured on a 5 s clip, **76% of frames partially occluded**, CPU build:

| | 30 Hz clip | 120 Hz clip |
|---|---|---|
| length | 4.7% MAE | 5.7% MAE |
| centroid x | 5.4% MAE | 5.4% MAE |
| centroid y | 4.8% MAE | 4.7% MAE |
| scale recovered from the width ruler | +5.5% | +5.5% |
| net displacement | −4.5% | −4.4% |
| contraction frequency | **1.500 Hz** (truth 1.5) | **1.500 Hz** (truth 1.5) |
| frames tracked | 100% | 100% |

Both frame rates recover the same physical answer — and the same calibration to
two decimal places — which is the check that matters for feature 1.

Read that table as one error, not six. Every millimeter figure is a pixel
measurement divided by the recovered scale, and the scale is 5.5% high, so the
centroid errors are almost exactly the calibration error and carry no
independent information. The cause is edge placement: the color cut sits 30% of
the way from the medium to the robot, which includes the blurred boundary pixels
and hands back a body a few percent too wide. Running the same fit on a neutral
fixture in luma mode returns 2.1% length MAE and −2.1% on scale, so this is the
threshold's edge bias rather than anything in the fit.

What follows from that is a practical point rather than a caveat: **strain,
frequency and force are unaffected, absolute size is not.** A uniform scale
error divides out of every ratio, which is why length strain is right while
length in micrometers is a few percent high. If you need absolute size to better
than a few percent, calibrate against a ruler in the frame
(`--px-per-mm`) rather than against the robot's own width, or raise the color
fraction until the fitted proportions match the drawing's.

Cumulative path length reads 10–12% high, because tracking jitter adds distance
the robot never traveled — net displacement is the more robust locomotion
metric, and increasing `--smooth-ms` trades jitter for temporal resolution.

These are synthetic-data numbers. Treat them as an upper bound on real-footage
accuracy: the fixture has no specular highlights, no depth of field, no rolling
shutter and a perfectly known outline.

Reproduce:

```bash
python tests/make_synthetic.py --fps 120 --seconds 5 --outdir syn
BioHybrid RoboTracker syn/synthetic_120fps.mp4 --dxf syn/robot.dxf -o out
python tests/evaluate.py out/tracking.csv syn/truth_120.npy --seconds 5
```

On a CPU-only machine this takes a few minutes; on CUDA or MPS, seconds.

---

## Interface and theme

### Every control explains itself

Each adjustable field carries a **(?)** badge. Hovering gives a one-line summary
with the valid range and default; clicking opens the full explanation — what the
parameter does, and concretely what going too far in each direction looks like.
All 22 entries live in `robotrack/paramhelp.py` as editable prose, and the same
text is available from Python via `paramhelp.describe("coverage")`.

The guidance is specific rather than generic. The coverage-weight entry, for
instance, records that raising it from 0 to 3 cut length error from 15.9% to
5.6% in validation — so you know which dial actually matters when occlusion is
hurting you.

### Visual identity

Built on the shared **MEA Suite theme kit**, which is vendored unmodified as
`robotrack/mea_theme.py`. The suite's rule is that neutrals, the indigo→teal
window gradient, radii (card 12 px · button 8 px · field 6 px), typography and
the parula colormap stay identical across programs, and **only the accent
changes**. BioHybrid RoboTracker follows that, so it reads as a member of the family.

Its accent is **`#FF5470`, crimson-rose**, registered as `myo`. Chosen by
measurement rather than by eye:

| Check | Result |
|---|---|
| Hue distance to nearest existing accent | 53° (to `solar` amber); 67° to `pattern` magenta |
| Contrast with on-accent near-black `#0A0E16` | 6.21:1 — above the 4.5:1 AA floor, and above `analyzer` blue's 5.25:1 |
| Contrast against `PANEL` | 5.82:1 |

Thematically it is oxygenated muscle rather than the blues and teals that
electrophysiology tooling has standardised on — this program measures
contraction, not spikes.

One caution that comes with a red accent: red reads as "danger" in most
interfaces, and the primary button is filled with it. Warnings and errors
therefore use amber and emerald, **never a second red**, so the accent never has
to compete with an alarm. Exported figures use the suite's plot styling and draw
from the shared parula map, so panels look like they came from the same
instrument.

To register the accent in your master kit, add one line to `ACCENTS` in
`config\mea_theme.py`:

```python
"myo": "#FF5470",     # BioHybrid RoboTracker - biohybrid muscle robots
```

Nothing in BioHybrid RoboTracker hardcodes a palette hex; `robotrack/theme.py` pulls every
neutral from the kit and defines only the accent.

## Capture protocol

Your rig (tripod, stabilization off, fixed 2x lens) already avoids the worst
problems. To keep it that way:

- **Lock AE/AF** (long-press on the subject) so exposure drift doesn't move the
  segmentation threshold mid-clip.
- **Stay on the 2x lens, no digital zoom.** A mid-record lens switch causes a
  discontinuous jump in pixel scale that nothing downstream can undo.
- **Put a ruler in the plane of motion**, in frame, in focus. Pass the result via
  `--px-per-mm`.
- **Keep stabilization off.** EIS warps and crops frames, making a static
  background appear to move, which breaks the fixed-camera assumption the median
  background plate depends on.
- **Record at ≥4× your contraction frequency**, ideally ≥8×. At 1 Hz contraction,
  30 Hz is plenty; use 120 Hz for fast twitch dynamics.
- **Good even lighting** — it keeps the iPhone at a constant frame rate. Low light
  is what triggers VFR.

## Layout

```
robotrack/
  ingest.py      frame-rate detection, PTS timestamps, VFR diagnosis
  decode.py      NVDEC -> D3D11VA -> CPU decode chain, auto-probed
  gpu.py         device selection, GPU morphology, Otsu
  segment.py     median background plate, mask, occlusion-aware fragment grouping
  cad.py         DXF -> template polygon, normals, signed-distance grid
  register.py    batched multi-start robust chamfer fitting (the core)
  shape.py       markerless PCA measurement (no-CAD path)
  kinematics.py  time-aware smoothing, path length, frequency analysis
  pipeline.py    orchestration, CSV/plots/overlay
  cli.py         command line
  gui.py         PySide6 desktop app with live segmentation preview
  placement.py   draggable CAD outline over the frame, for picking the target
  automatch.py   scores each candidate outline against real frames
  forcelut.py    Length,Force calibration curve -> force per frame
  plotpanel.py   live, zoomable plots beside the video
  splash.py      launch splash with real load progress
  assets/        splash artwork, application icon, completion sounds
  settings.py    remembered session state and .rtcfg configuration files
  update.py      update channels, code-overlay patching, rollback
  updater_ui.py  the Update dialog
  ffmpeg.py      locates bundled or system ffmpeg/ffprobe
  sound.py       completion / abort chimes, no extra dependency
  forcelut.py    Length,Force calibration curve -> force per frame
  forcemodel.py  Cvetkovic beam model: delta length -> force, in closed form
  theme.py       widget styling, the self-explaining help popups
  mea_theme.py   the MEA Suite palette and type scale
  paramhelp.py   the text behind every control's help button
launcher/
  app.py         frozen entry point, crash dialogs, --selftest
  robotrack.spec PyInstaller build definition (Windows folder / macOS .app)
  installer.iss  Inno Setup script -> single-file installer
  robotrack.ico  Windows icon
  robotrack.icns macOS icon
build_exe.ps1        one-command Windows build
build_macos.sh       one-command macOS build (.app + .dmg)
install_windows.ps1  source checkout, CUDA wheel
install_macos.sh     source checkout, Metal/MPS wheel
.github/workflows/
  release.yml    tag -> build both platforms -> GitHub Release
tools/
  publish_update.py   package and publish an update to a channel
tests/
  make_synthetic.py   ground-truth clip generator
  evaluate.py         accuracy scoring
MACOS-INSTALL.md click-by-click Mac install guide for non-technical users
PUBLISHING.md    first-time git/GitHub walkthrough and release procedure
LICENSE          GPL-3.0-or-later
```

## When a run is slow

Every run now reports where its time went, because a clip that takes twenty
seconds one day and ten minutes the next is almost always one stage rather than a
general slowdown. Measured on the reference clip:

```
time per frame : fit 210 ms (91%)  decode+segment 20 ms (9%)  live preview 0 ms (0%)
```

Two stages can dominate, and they fail differently.

**The overlay video** is a second full pass — decode the whole clip again in
color, composite, re-encode — and it used to run *after* the reported elapsed
time with no progress, so it looked like a hang. At full 4K it measured about 3.4
minutes per 930 frames against 13 seconds at 960 px. It is now written at 960 px
on the longest side by default (`overlay_max_px`), reports progress, and is
included in both the elapsed time and the per-stage breakdown. An overlay is for
confirming the fit followed the robot, not for measuring from.

**The fit** is the other one, and the thing to check is whether it is on the GPU. On the CPU it measures 400–570 ms per frame against
roughly 27 ms on CUDA — a 930-frame clip goes from about 25 seconds to over ten
minutes, and nothing else in the pipeline comes close to mattering by comparison.
The run summary says so explicitly when CUDA is absent, the log prints the device
and settings before starting, and the app asks for confirmation rather than
quietly spending ten minutes.

The usual cause is a source install where `pip install -e ".[gui]"` pulled the
CPU-only wheel from PyPI. Check with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

After the device, the dials that matter are **restarts** (16 → 128 roughly
triples the fit), **iterations**, **decode scale**, and interior features (about
+11%). Everything else is noise next to those.

**Early stop** ends each fit once the best hypothesis stalls rather than always
running the full schedule, then jumps the kernel to its final width so the result
is not left converged under a wider one. On the reference clip that is 463 → 246
ms per frame, a 1.9× speedup, with median length unchanged to 0.1%. It is not
free — confidence fell from 0.85 to 0.82 and width CV from 0.16% to 0.34%, both
still far inside tolerance — so it can be turned off for a final run.

## Performance notes

Decoding dominates for 4K120 HEVC, which is why NVDEC matters — it is a
dedicated silicon block, so hardware decode runs alongside GPU fitting for free.
Frames are decoded straight to 8-bit grayscale (segmentation only needs luma),
cutting pipe bandwidth 3×.

The fit evaluates all restarts as one batched tensor op, so 64 restarts cost
little more than one — which is what makes the multi-start search affordable.
Once tracking is locked on, a shorter warm-start schedule is used. If throughput
is short, `--scale 0.5` and `--restarts 16` are the first two dials to turn.

The same shape holds on Apple Silicon with different hardware underneath: the
media engine decodes through VideoToolbox while MPS runs the fit, so the two
still overlap. Expect an M1/M2 to sit between a CPU-only machine and a discrete
RTX card — the unified-memory GPU is real acceleration, and the batched restart
trick is exactly the kind of work it is good at, but the memory bandwidth and
core count are not in the same class. One MPS-specific detail: `torch.histc` has
no Metal kernel, so `gpu.otsu_threshold` computes just the histogram on the CPU
and leaves the rest of the pipeline on the GPU. Falling back wholesale for one
missing op would have cost far more than the histogram does.

## Extending

`register.ShapeFitter` only needs a `Template` (points + normals + SDF), so
swapping DXF for SVG, STL silhouettes, or a hand-traced outline means writing
one loader. Allowing non-rigid deformation means adding parameters to the pose
vector — the cost function and its gradients need no changes.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

The choice follows from what this links against. PyQt-free though it is, the
stack includes an ffmpeg build compiled with GPL components, and the GPL is
also what most of the academic tooling in this space uses, so it keeps the
project compatible with the things people will want to combine it with. In
practice it means: use it, modify it, publish papers with it freely; if you
distribute a modified version, ship the source of your modifications too.
