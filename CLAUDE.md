# BioHybrid RoboTracker — working notes for Claude Code

GPU video tracker for muscle-driven hydrogel soft robots. Measures per-frame
outline, length, leg closure, trajectory, contraction frequency and muscle
force from phone video. Windows + macOS desktop app (PySide6) plus a CLI.

**Current version: 0.40.0.** The version lives in exactly one place,
`robotrack/__init__.py`, and the installer, the `.app` Info.plist, the update
manifest and the release workflow all read it from there.

`README.md` in this folder is a 100 KB engineering log with the full reasoning
behind most design decisions. This file is the orientation and the rules; when
something here says "see the README", it means there is a longer argument there
worth reading before changing that area.

---

## Naming — do not "finish the rename"

Display name is **BioHybrid RoboTracker** (`APP_NAME`). The Python package, the
executable, the install folder, the Inno `AppId` and the release filenames all
stay **`robotrack`** on purpose: the code-patch updater ships a folder named
after the package and every installed copy already has it on `sys.path`.
Renaming it strands every existing install on whatever version it is running.

---

## Commands

```bash
# run from source
python -m robotrack.gui          # the app
python -m robotrack.cli --help   # headless

# publish a code update (the normal release path)
#   1. bump __version__ in robotrack/__init__.py
#   2. then:
python tools/publish_update.py updates --notes "what changed and why"
```

`publish_update.py` zips the package, hashes it, writes it into `updates/` and
rewrites `updates/robotrack-updates.json`. It refuses to overwrite a version
that is already published unless you pass `--force`. A **code patch carries
pure Python only** — if a release changes a dependency, the Python version or
the bundled ffmpeg, it needs a full installer (`--full`), because an overlay
cannot swap a compiled wheel that is already loaded.

`updates/`, `tests/` and `tools/` are **gitignored**. The release artifacts are
not in git; the publish script writes into an ignored folder by design.

Installers: `build_exe.ps1` (Windows, Inno) and `build_macos.sh` (macOS, .dmg).

### git

- `git add -A`, **never** `git commit -am`. The latter has silently skipped a
  new module twice (`sound.py` at 0.11.0, `appearance.py` at 0.16.0), producing
  a tag whose build could not import.
- A tag that already exists on the remote will not re-fire CI. Delete and
  recreate it; pushing an unchanged ref is a no-op and reports
  "Everything up-to-date".
- **Outstanding:** `git rm -r --cached tests tools` has never been run. They are
  in `.gitignore` but still tracked.

---

## Architecture

```
ingest   probe the file, measure the real frame rate from timestamps
decode   per-platform hwaccel, probed in order (NVDEC/VideoToolbox/D3D11VA/CPU)
segment  CIELAB chroma keying; fragment regrouping under a body envelope
cad      DXF → template points, normals, signed-distance grid
register 5-DOF chamfer fit, Geman-McClure annealed kernel, batched restarts
legpoints two marked points tracked independently by ECC   ← the force input
kinematics gating, smoothing, path length, dominant frequency
forcemodel Euler-Bernoulli beam  |  forcelut  COMSOL curve
pipeline  orchestrates all of the above, writes tracking.csv + run_info.json
plotpanel live plots, region selection, contraction counting
gui       PySide6 window; placement.py is the interactive overlay widget
```

`appearance.py` is a third tracker (ECC affine lock) for footage where colour
keying cannot work at all.

### Three ways to measure, and what each is for

| | needs | speed | use |
|---|---|---|---|
| markerless | nothing | ~29 fps | the default |
| CAD fit | a DXF | 1.5–3 fps | occlusion tolerance only |
| appearance lock | a reference patch | — | when colour keying fails outright |

**CAD fit is no longer the main path.** Force comes from leg points now, so the
fit's only unique contribution is measuring through an occluder. On the
reference clip the fit was 89% of a 646 s run and produced a length column the
force did not read.

---

## Invariants that are easy to break

### Coordinate spaces

Three of them, and mixing them has caused two separate bugs:

- **Full-resolution image px** — what is *stored*: `roi`, `leg_marks`,
  `manual_pose`. Stored this way so they survive a change of decode scale.
- **Decoded px** — what the pipeline *measures* in: `width_px`, `length_px`,
  `leg_sep_px`, and the calibration derived from them. All consistent.
- **Preview px** — what the view *draws* in. Anything handed to `PreviewView`
  must be converted: `_pose_to_preview`, `_roi_to_preview`, `_legs_to_preview`,
  and the inverses on the way back.

`SegmentConfig.roi_scale` carries the decode scale into the three places that
convert the region. It is set from the reader (which rounds dimensions to even
numbers, so the effective scale is not exactly the requested one), not assumed.

### The force model

```
I = t³ · w / 12          t = 1.100 mm, w = 5.06 mm → I = 0.56124 mm⁴
θ = asin((δ/2) / L_leg)  L_leg = 3.7125 mm
F = 2·E·I·θ / (l·L)      E = 293 kPa, l = 1.238 mm, L = 8.030 mm
```

- The section is **solid**. Two earlier answers were wrong and both produced
  plausible forces: a solid 3.025 mm section (40% low) and 5.06 mm minus a
  0.9625 mm channel (19% low). The beam width is a **fixed 5.06 mm**, not tied
  to the Input card's True width.
- **δ is leg closure, not length change.** Whole-body measurements report a
  fraction of it: measured against manual tracking, the silhouette gave
  0.759 mm of length change per 1 mm of real closure, at r = 0.993 — following
  the motion perfectly and reporting the wrong quantity.
- Marking the legs **in the DXF would not work**: under a fitted pose, two
  template-fixed points are a constant multiple of the fitted length apart.
  They must be tracked in the image, independently of the body.
- At rest the two marks should sit `L` (8.030 mm) apart. The run checks this and
  warns, because marks further out along the legs read high and marks inboard
  read low, and neither looks like anything but a plausible force.

### Calibration

`px_per_mm = width_px(first usable frame) / true_width_mm`, taken from the raw
(unsmoothed) width on one frame — not a median over the clip. A ruler must not
change because the frame range or the confidence floor moved. The clip-wide
median and its CV are still computed as a *check*, and the summary warns when
the calibration frame sits >3% off it.

This means **segmentation is still load-bearing for force**, even though the
deflection comes from the marks: both series divide by the same `px_per_mm`,
which comes from the segmented width. Typing a value into Calibration (px/mm)
takes segmentation out of the force entirely.

---

## Failure modes that have actually happened

Each of these cost real time. The signature matters more than the fix.

**A windowed build has no stderr.** An exception in a Qt slot prints to a
console that does not exist, and the program carries on looking merely stuck.
An attribute assigned in `__init__` shadowed the `_loaded` *method* (the
LoadWorker's done handler), so `connect(self._loaded)` raised `TypeError`,
`.start()` was never reached, and for four releases **no video could be opened
at all** — with no symptom but a progress line that never advanced. There is now
a `sys.excepthook` + `threading.excepthook` writing to `robotrack.log`, and an
AST scan for attribute/method shadowing. **Run that scan after adding state to
`MainWindow`:**

```python
# any attribute that shadows a method on the same class
import ast, pathlib
tree = ast.parse(pathlib.Path("robotrack/gui.py").read_text(encoding="utf-8"))
for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assigned = {t.attr for n in ast.walk(cls)
                for t in (n.targets if isinstance(n, ast.Assign)
                          else [n.target] if isinstance(n, ast.AnnAssign) else [])
                if isinstance(t, ast.Attribute) and getattr(t.value, "id", "") == "self"}
    assert not (methods & assigned), (cls.name, methods & assigned)
```

**Rebinding a running QThread aborts the process.** Qt calls `abort()` — SIGABRT,
no traceback — when a running QThread is destroyed, and Python destroys one the
moment the last reference goes. `self._loader = LoadWorker(...)` during a load
was a hard crash. `MainWindow._retire(attr)` disconnects the old worker's
signals and parks it in `self._retired` until it finishes. **Call it before
every worker reassignment** (`_sizer`, `_player`, `_matcher`, `_loader`,
`_preview`, `_runner`).

**Never ship an ffmpeg command-line change tested only on one decoder.** Adding
a `-vf scale` filter to the keyframe pass was correct and nearly free on CPU
decode, and on NVDEC cuvid the filter graph rejected the frames, `_keyframes`
returned empty, and `sample()` fell through to its last resort — a full
sequential decode of the entire clip, no timeout, nothing logged. A 4K clip
looked hung for three hours. `FrameReader.sample_path` now records which of the
three routes was taken and the open line reports it; anything other than
`keyframes` is a fault.

**A silent fallback that is 100× slower is worse than an error.** That is the
general form of the one above. If a fast path can fail into a slow one, say so.

**`_fit_signal` (the colour-distance surface) is zero outside the ROI.** The leg
tracker matches against it, so a mark outside the region is matched against flat
nothing, holds position all clip, and produces a perfectly smooth trace of two
stationary points. The run warns about this now.

**Gating couples things that should not be coupled.** `leg_sep_px` used to be
gated on the *fit's* confidence; `gate_and_fill` does not drop rejected frames,
it interpolates them, so good leg data came back as an invented smooth curve on
exactly the clips where segmentation was failing. It is gated on `leg_ok` now.

**Slow runs are usually a fit that never locks.** Below ~0.75 median confidence
the warm-start path is not trusted and every frame re-runs the cold schedule
(64 restarts × 150 iters vs 10 × 30) — ~32× the work. On the reference clip
that is 0.574 s/frame against 0.26 s/frame. The summary says so now, along with
a warning when interior features match under 15% of observed edges (7% was
costing most of a 646 s run).

---

## Platforms

| | Windows | macOS |
|---|---|---|
| compute | CUDA | Metal / MPS |
| decode | NVDEC cuvid → generic → D3D11VA → CPU | VideoToolbox → CPU |
| build | `build_exe.ps1` | `build_macos.sh` |

The Windows + CUDA path is well tested; the Mac path is not. Known MPS traps,
all already handled — **do not "simplify" these back**:

- `torch.histc` can be missing or wrong on MPS. `gpu.otsu_threshold` guards the
  **real** call (not a probe) and remembers the verdict, because Otsu runs per
  frame.
- A stride-0 `expand()` into `grid_sample` makes the fit wander on Metal.
  `register.py` flattens to one batch instead; verified bit-identical.
- `stack.median(dim=0)` falls back to numpy on MPS.
- `verify_device()` catches a backend that computes `grid_sample` *subtly wrong*
  — it returns numbers, the optimiser converges, and the fit lands confidently
  on the wrong part of the picture. It runs at startup and in `pipeline.run`.
- Every tensor that reaches a device is explicitly float32. MPS has no float64.
- `build_macos.sh` must stay **bash 3.2** compatible (macOS ships 3.2.57): no
  heredoc inside `$( )`, and `"$@"` under `set -u` needs a `$# -gt 0` guard.

**Open on Mac:** the UI is sized in points against `Segoe UI` / `Cascadia Mono`,
neither of which exists on macOS, and Qt maps points through logical DPI — 72 on
macOS against Windows' 96, so everything renders ~25% smaller inside chrome full
of fixed pixel widths. `self.resize(1480, 940)` also has no
`availableGeometry()` clamp and is taller than a 13" MacBook Air's usable area.
Needs eyes on a real Mac, not a code review.

---

## Verifying a change

There is **no pytest suite**. `tests/` holds `evaluate.py` and
`make_synthetic.py` only. What has actually caught regressions:

1. **Headless GUI smoke test** — builds the whole window, exercises the widget:
   ```bash
   QT_QPA_PLATFORM=offscreen python -c "
   from PySide6.QtWidgets import QApplication; app=QApplication([])
   from robotrack.gui import MainWindow; w=MainWindow(); print('ok'); w.close()"
   ```
2. **A real clip end to end**, checking the numbers rather than the exit code —
   `calibration_px_per_mm`, `force_length_source`, median `leg_a_conf`, fps.
3. **Round-trip assertions** for anything coordinate-related: convert and
   convert back, and assert equality on the actual stored values.
4. **Compare against ground truth** when touching the force path. A colleague's
   manual tracking of `Ring1_1HzTet` is in `TestRun/Tracking_Raw.xlsx`: per-leg
   (t, x, y) for 1/2/3/4 Hz × twitch/tetany. The median peak-to-trough leg
   closure through the beam model reproduces her eight published forces to
   −4.1%…+2.6%. That file is the regression test for force.

Reference numbers for `Ring1_1HzTet.MOV` (724×724, 930 frames, markerless +
leg marks): ~29.5 fps, force 5550 µN against her 5589, `leg_a_conf` 0.978,
`leg_b_conf` 0.970.

---

## Open items

- **GitHub repo description** still says the old name. Set it to "BioHybrid
  RoboTracker: GPU accelerated tracking for muscle-driven soft robots".
  Outstanding since 0.11.0.
- **Latest published GitHub release is still v0.14.0**, though the code channel
  is at 0.40.0. Releases and the local update channel have drifted apart.
- **Installer rebuild needed** for the embedded `.exe` icon and the macOS
  certifi/SSL fix — a code patch cannot change either.
- `git rm -r --cached tests tools`.
- **The ×1.1 DXF scale / `known_width_mm = 5.78`** may be wrong — possibly 5.25.
  Interior features match only 7% of observed edges on the reference clip, which
  is the signature of a drawing and a robot that are not the same size. This
  affects fit quality and therefore run speed, and is worth resolving on its own
  merits.
- **macOS font/DPI/layout pass** (above).
- Watch the Windows installer against GitHub's 2 GiB asset limit (0.10.0 was
  1.9 GB).

---

## Working style that paid off here

- **Measure before optimising.** Every speed claim in this project came from a
  stage profile, and two of three intuitions about where the time went were
  wrong. `run_info.json` carries `stage_seconds`.
- **When a symptom is confusing, make the program say what it is doing** rather
  than reasoning from the description. Two wrong diagnoses and a shipped bad fix
  preceded the one-line instrumentation that found the real bug in a minute.
- **State the quantity, not just the number.** The costliest bug in this project
  was not a wrong number, it was the right number for the wrong measurement
  (`force_length_source` exists because of it).
