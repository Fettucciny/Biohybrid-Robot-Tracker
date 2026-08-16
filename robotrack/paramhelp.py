"""Explanations for every adjustable parameter in the interface.

Kept separate from the widget code so the guidance can be reviewed and edited as
prose, and so the CLI can print the same text.

Each entry provides:
  title     - human name
  short     - one line, shown on hover
  what      - what the parameter actually does
  range     - valid range with units
  default   - shipped default
  guidance  - concrete symptoms of setting it wrong in each direction
"""

from __future__ import annotations

HELP: dict[str, dict] = {

    # ---------------------------------------------------------------- input
    "video": {
        "title": "Video file",
        "short": "The recording to analyze.",
        "what": "An iPhone .MOV or any file ffmpeg can read. Frame rate is measured "
                "from the file's real timestamps rather than the declared rate, so "
                "30, 60 and 120 Hz clips are handled automatically and no setting "
                "needs changing between them.",
        "range": "any ffmpeg-readable video",
        "default": "—",
        "guidance": [
            "Record on a tripod with stabilisation off; the background model assumes "
            "the camera does not move.",
            "Keep the lens fixed at 2x. A mid-recording lens switch causes a sudden "
            "jump in pixel scale that cannot be corrected afterwards.",
            "Good lighting keeps the phone at a constant frame rate. Low light is what "
            "triggers variable frame rate.",
        ],
    },
    "dxf": {
        "title": "CAD outline (DXF)",
        "short": "Optional 2D drawing used as the tracking template.",
        "what": "A 2D DXF of the robot's outline. Supplying one switches from "
                "markerless measurement to model-based fitting: the known shape is "
                "fitted to each frame, so width and length can still be measured "
                "where part of the robot is hidden behind an obstacle. Without it, "
                "measurements come from the visible silhouette only.",
        "range": "2D DXF; arcs, splines and polylines all supported",
        "default": "none (markerless)",
        "guidance": [
            "Drawing units are read from the file header, so an inch drawing converts "
            "correctly.",
            "Draw the robot at its resting size — the pixel-per-mm self-calibration "
            "assumes this.",
            "If geometry lives inside a block, explode it first; only model space is read.",
        ],
    },

    # --------------------------------------------------------- segmentation
    "background_frames": {
        "title": "Background frames",
        "short": "How many frames are sampled to build the static background plate.",
        "what": "The robot is found by comparing each frame against a per-pixel median "
                "of this many frames spread across the clip. A median is used rather "
                "than an average so the robot, which appears in every frame, is seen "
                "straight through instead of leaving a ghost.",
        "range": "10 – 400 frames",
        "default": "60",
        "guidance": [
            "Raise it if the background plate shows a faint ghost of the robot — this "
            "happens when the robot dwells in one spot for much of the clip.",
            "Raise it for long clips with drifting illumination.",
            "Lowering it below about 30 makes the plate noisy and the mask ragged.",
            "Cost is one extra decode pass, so very high values slow loading.",
        ],
    },
    "threshold_mode": {
        "title": "Threshold mode",
        "short": "Automatic (Otsu) or a fixed manual value.",
        "what": "Sets how different from the background a pixel must be to count as "
                "robot. Automatic uses Otsu's method on the first frames and then "
                "freezes the value; freezing matters because a per-frame threshold "
                "drifts when the robot is occluded, which would make measured size "
                "depend on the occlusion.",
        "range": "Auto or Manual",
        "default": "Auto",
        "guidance": [
            "Auto works when the robot contrasts clearly with the dish.",
            "Switch to Manual for low-contrast tissue, strong reflections, or when the "
            "mask visibly flickers between frames.",
            "Scrub to a hard frame and tune Manual there, not on an easy one.",
        ],
    },
    "threshold": {
        "title": "Manual threshold",
        "short": "Gray-level difference from background required to count as robot.",
        "what": "A pixel joins the mask when it differs from the background plate by "
                "more than this many gray levels (0–255 scale).",
        "range": "1 – 254 gray levels",
        "default": "30",
        "guidance": [
            "Too low: background texture, reflections and shadows leak into the mask; "
            "the measured size inflates and jitters.",
            "Too high: faint edges of the robot are lost; width and length read "
            "systematically small.",
            "Tune with the mask overlay on — aim for the mask edge sitting on the "
            "visible tissue boundary.",
        ],
    },
    "despeckle": {
        "title": "Despeckle",
        "short": "Removes isolated noise blobs smaller than this size.",
        "what": "Morphological opening. Erodes then dilates the mask, which deletes "
                "specks narrower than the kernel while leaving the robot's overall "
                "size unchanged.",
        "range": "1 – 31 px (odd values; 1 disables)",
        "default": "3",
        "guidance": [
            "Raise it if the mask has scattered dots from sensor noise or debris.",
            "Too high erases genuinely thin structures — thin tails and the narrow "
            "slivers that show past an obstacle, which are exactly the evidence the "
            "occlusion fitting depends on.",
            "Rarely needs to exceed 5 on well-lit footage.",
        ],
    },
    "fill_holes": {
        "title": "Fill holes",
        "short": "Closes pinholes and hairline gaps inside the mask.",
        "what": "Morphological closing. Dilates then erodes, sealing small interior "
                "holes and gaps caused by specular highlights on wet tissue.",
        "range": "1 – 41 px (odd values; 1 disables)",
        "default": "7",
        "guidance": [
            "Raise it when bright reflections punch holes through the middle of the robot.",
            "Too high bridges the gap between genuinely separate parts and can merge "
            "the robot with a nearby object.",
            "It also slightly rounds sharp concave features.",
        ],
    },
    "min_area": {
        "title": "Minimum blob size",
        "short": "Noise floor, as a fraction of total frame area.",
        "what": "Connected components smaller than this fraction of the frame are "
                "discarded as noise before any measurement.",
        "range": "0.00001 – 0.01 (i.e. 0.001% – 1% of the frame)",
        "default": "0.0001 (0.01%)",
        "guidance": [
            "Too high discards the small sliver of robot that pokes past an obstacle — "
            "and that sliver is often the only thing pinning down the hidden end.",
            "Too low admits background texture as if it were robot.",
            "If tracking fails only during occlusion, try lowering this first.",
        ],
    },
    "gap_factor": {
        "title": "Occlusion gap tolerance",
        "short": "How far apart fragments can be and still be treated as one robot.",
        "what": "When an obstacle cuts the robot in two, the pieces are regrouped if "
                "they sit within this multiple of the body length of each other. Body "
                "length is learned from the least-occluded frames in the clip.",
        "range": "0 – 3 body lengths",
        "default": "1.0",
        "guidance": [
            "Raise it if a wide obstacle splits the robot and only one half is being "
            "tracked.",
            "Too high merges genuinely separate objects — another robot, a bubble, or "
            "debris — into one measurement.",
            "Set to 0 to keep only the single largest blob.",
        ],
    },

    # -------------------------------------------------------- shape fitting
    "tau": {
        "title": "Robust kernel τ (start)",
        "short": "How far an edge may stray and still attract the outline.",
        "what": "Scale of the bounded robust cost. Template points further than about "
                "τ from any real edge stop pulling on the fit, which is what makes "
                "occluded regions harmless instead of dragging the outline off the "
                "robot. τ is annealed from this value down to the final value during "
                "fitting, giving a wide capture range early and precision at the end.",
        "range": "1 – 60 px",
        "default": "12",
        "guidance": [
            "Raise it if the outline fails to lock on, or if the robot moves a long way "
            "between frames.",
            "Too high lets the fit settle loosely, softening real contraction amplitude.",
            "Scale it with your robot: roughly 5–10% of body length is a good start.",
        ],
    },
    "tau_final": {
        "title": "Robust kernel τ (final)",
        "short": "Kernel width at convergence; sets final precision.",
        "what": "The annealing endpoint. Small values make the last iterations care "
                "only about edges very close to the outline, which is where sub-pixel "
                "accuracy comes from.",
        "range": "0.5 – 10 px",
        "default": "2.5",
        "guidance": [
            "Lower for sharp, high-contrast edges and clean footage.",
            "Raise for blurred or noisy edges, where an over-tight kernel makes the fit "
            "chase segmentation noise and adds jitter.",
            "Must stay below the starting τ.",
        ],
    },
    "early_stop": {
        "title": "Early stop",
        "short": "End the fit once it stops improving, instead of always running every iteration.",
        "what": "The iteration count is sized for the worst case — a cold start from a "
                "bad seed. A warm frame in a locked-on sequence converges in a "
                "fraction of it, and the remaining iterations only re-confirm the same "
                "answer. This checks the best hypothesis periodically and stops when it "
                "stalls, then jumps the kernel to its final width so the result is not "
                "left converged under a wider one.",
        "range": "on / off",
        "default": "on",
        "guidance": [
            "Measured on the reference clip: 463 → 246 ms per frame, a 1.9× speedup, "
            "with median length unchanged to 0.1%.",
            "It is not free. Confidence fell from 0.85 to 0.82 and width CV rose from "
            "0.16% to 0.34% — both still far inside tolerance, but turn it off for a "
            "final measurement run if you want every iteration.",
            "The check forces a GPU→CPU sync, so it runs every tenth iteration rather "
            "than every one; checking constantly would cost more than it saves.",
        ],
    },
    "restarts": {
        "title": "Restarts",
        "short": "Independent starting poses tried in parallel each frame.",
        "what": "The fit is a non-convex search, so several starting guesses are "
                "optimized simultaneously and the best is kept. All restarts are "
                "evaluated as one batched GPU operation, so 64 costs little more than "
                "1 — which is what makes this affordable.",
        "range": "1 – 256",
        "default": "64",
        "guidance": [
            "Raise it if tracking occasionally jumps to a wrong pose or loses the robot.",
            "Lower it to about 16 for faster preview scrubbing while tuning.",
            "On CPU this scales cost almost linearly; on GPU the effect is modest.",
        ],
    },
    "coverage": {
        "title": "Occlusion coverage weight",
        "short": "Penalty for observed robot pixels left outside the fitted outline.",
        "what": "Edge-distance alone cannot tell a correct fit from one that has "
                "collapsed onto part of the robot — both put template points on real "
                "edges. This term asks the reverse question: is there visible robot "
                "outside my outline? It is one-sided by design, since occlusion only "
                "ever removes pixels and so can never trigger it falsely.",
        "range": "0 – 20",
        "default": "3.0",
        "guidance": [
            "This is the single most important control for occlusion. Raising it from "
            "0 to 3 cut length error from 15.9% to 5.6% in validation.",
            "Raise it if the outline shrinks onto part of the robot.",
            "Too high makes the outline reluctant to shrink at all, suppressing real "
            "contraction.",
        ],
    },
    "scale_prior": {
        "title": "Temporal smoothness prior",
        "short": "Resistance to sudden frame-to-frame size changes.",
        "what": "Muscle contraction is continuous, so a large jump in fitted size "
                "between consecutive frames is more likely a fitting artifact than "
                "real biology. Expressed as a strain rate per second, so it means the "
                "same physical thing at 30 and at 120 Hz.",
        "range": "0 – 2 (0 disables)",
        "default": "0.35",
        "guidance": [
            "Raise it if size readings are noisy or spike during occlusion.",
            "Too high damps genuine contraction and underestimates amplitude — if your "
            "measured strain looks suspiciously small, lower this first.",
            "Set to 0 when studying fast twitch dynamics.",
        ],
    },
    "max_scale": {
        "title": "Maximum size change",
        "short": "Limit on how far the outline may stretch or shrink from nominal.",
        "what": "Bounds the fitted width and length scale factors relative to the "
                "drawing's nominal size, preventing the optimizer from wandering into "
                "physically impossible solutions.",
        "range": "0.1 – 0.95 (fraction)",
        "default": "0.60 (±60%)",
        "guidance": [
            "Raise it if your robot contracts more than 60% and readings appear clipped "
            "at the limit.",
            "Lower it to stabilise tracking on difficult footage.",
            "Check the CSV: values pinned exactly at the bound mean this is too tight.",
        ],
    },

    "width_weight": {
        "title": "Width hold",
        "short": "How hard the fitted width is held to the drawing's width.",
        "what": "The robot's width and its length are not the same kind of quantity. "
                "Width is set by the mould: it is whatever the drawing says, it takes "
                "no part in contraction, and a frame where the fitted width departs "
                "from the drawing is a frame where segmentation went wrong — not one "
                "where the robot got wider. Length is the opposite; it is the thing "
                "being measured. This weight applies a quadratic penalty pulling the "
                "width back to nominal, in the same units as the outline-matching "
                "cost, which saturates at 1. So 2.0 means one tolerance of width "
                "error costs twice as much as a completely wrong outline.",
        "range": "0 (off) – 20",
        "default": "2.0",
        "guidance": [
            "Raise it if the fitted width in the CSV wanders while the robot "
            "visibly does not.",
            "Lower it, or turn it off, if you are tracking something whose width "
            "genuinely changes — a bending sheet rather than a walker.",
            "It is deliberately weak near nominal and strong far from it: at 1% "
            "off it barely acts, at 10% off it dominates everything else.",
        ],
    },
    "width_tol": {
        "title": "Width tolerance",
        "short": "The width error at which the hold reaches full strength.",
        "what": "The sigma of the width penalty, as a percentage of the drawing's "
                "width. Error inside this costs almost nothing; error well outside "
                "it costs the square. A separate hard clamp stops the width leaving "
                "±15% of nominal whatever this and the weight are set to.",
        "range": "0.5 – 30 %",
        "default": "4 %",
        "guidance": [
            "4% is about the frame-to-frame width noise of a clean color key at "
            "1080p, so this leaves ordinary noise alone and catches real failures.",
            "Widen it if the drawing's width is only approximately right — a "
            "penalty centred on the wrong value is worse than no penalty at all.",
        ],
    },
    "length_overshoot": {
        "title": "Length overshoot",
        "short": "How far the fitted length may exceed the drawing's, in pixels.",
        "what": "Relaxed length is bounded above by the drawing: a hydrogel robot "
                "contracts from its moulded length, it does not grow past it. An "
                "outline that has stretched beyond the drawing has latched onto "
                "something that is not the robot — a tether, a shadow, a second "
                "body keying in — and that failure is otherwise invisible, because "
                "an over-long fit still puts most of its points on real edges and "
                "still reports high confidence. This is the noise allowance on that "
                "ceiling.\n\n"
                "The ceiling itself is the hand-placed length if you placed the "
                "outline, and otherwise the 90th percentile of the first 45 "
                "confident fits — the clip's own answer to how long this robot is "
                "when relaxed.",
        "range": "0 – 60 px",
        "default": "3 px",
        "guidance": [
            "Not zero: a real mask edge moves about a pixel frame to frame, and a "
            "hard equality would clip genuine relaxation to whichever frame "
            "happened to segment tightest.",
            "Raise it if relaxed length in the CSV sits pinned at one value for "
            "long stretches — that means the ceiling is binding on real data.",
            "Raise it a lot, or place the outline by hand, if the clip never shows "
            "the robot fully relaxed: the learned ceiling is then too low.",
        ],
    },

    # ------------------------------------------------------------- analysis
    "smoothing": {
        "title": "Smoothing window",
        "short": "Savitzky-Golay filter width, in milliseconds of real time.",
        "what": "A Savitzky-Golay filter over the measured series: every frame keeps "
                "its own timestamp and its own value, and each value is replaced by "
                "a local polynomial fit through its neighbours. Nothing is dropped "
                "and nothing is re-timed. It removes noise; it does not change how "
                "often you measured.\n\n"
                "This is *not* the same control as trajectory sampling, and the two "
                "do different jobs on purpose:\n\n"
                "• Smoothing keeps every frame and pulls each one toward its "
                "neighbours. Use it for length, and for anything read off length.\n"
                "• Trajectory sampling, under the plots, throws frames away — it "
                "takes the centroid at, say, 5 Hz and ignores the rest. Use it for "
                "path length and speed.\n\n"
                "They are not interchangeable because path length is a *sum*, and "
                "summing noise adds distance that was never travelled. Smoothing "
                "reduces the size of each spurious step but still adds up all of "
                "them; sampling removes the steps entirely. Measured on real "
                "footage, jitter at 60 fps inflated path speed by a factor of 260, "
                "which no amount of smoothing recovers.\n\n"
                "Why it is set in milliseconds: the *number of frames* the window "
                "covers is what determines how much real signal it removes, and "
                "that number depends on the frame rate. A 5-frame window is 167 ms "
                "at 30 fps and 42 ms at 120 fps — the same setting, four times the "
                "smoothing. You are right that each frame already carries its own "
                "timestamp; the filter uses those. What the timestamps cannot do is "
                "decide how wide the window should be, and expressing the width in "
                "time rather than frames is what makes the same number mean the same "
                "thing on both recordings.",
        "range": "0 – 2000 ms (0 disables)",
        "default": "100 ms",
        "guidance": [
            "Keep it well under one contraction period, or you will flatten the "
            "contraction you are trying to measure. At 1.5 Hz (667 ms period), stay "
            "below roughly 150 ms.",
            "For path length and speed, reach for trajectory sampling instead — "
            "that is the control that fixes jitter-inflated distance.",
            "Applies to position and size, and to the velocities derived from them. "
            "The unsmoothed values stay in the CSV as the _raw columns.",
        ],
    },
    "min_confidence": {
        "title": "Minimum confidence",
        "short": "Fits scoring below this are treated as unobserved.",
        "what": "Every frame gets a confidence combining how much of the outline sits "
                "on real edges, whether anything spills outside the fit, and whether "
                "observed robot was left uncovered. Frames below the threshold are "
                "blanked before smoothing rather than contributing bad numbers.",
        "range": "0 – 1",
        "default": "0.5",
        "guidance": [
            "Raise it for a conservative dataset with more gaps but higher quality.",
            "Lower it if heavy occlusion is discarding frames you believe are usable.",
            "The per-frame confidence is written to the CSV, so you can re-filter "
            "afterwards without re-running.",
        ],
    },
    "max_gap": {
        "title": "Maximum bridged gap",
        "short": "Longest dropout that gets interpolated instead of left blank.",
        "what": "Low-confidence stretches shorter than this are filled by linear "
                "interpolation; longer ones stay as blank values in the output. "
                "Interpolating across a long dropout would invent data.",
        "range": "0 – 5000 ms",
        "default": "400 ms",
        "guidance": [
            "Keep it below half a contraction period, or interpolation will smooth over "
            "a real contraction cycle.",
            "Set to 0 to never interpolate — every uncertain frame stays visibly blank.",
            "Gaps left blank appear as breaks in the plots and empty cells in the CSV.",
        ],
    },
    "px_per_mm": {
        "title": "Pixels per millimeter",
        "short": "Spatial calibration. Leave at 0 to self-calibrate from the DXF.",
        "what": "Converts pixel measurements to millimeters. Measure it from a ruler "
                "placed in the plane of motion. A value typed here wins over "
                "everything else. At 0, the scale is derived from True width (or the "
                "drawing's width) divided into the robot's pixel width on the first "
                "frame of the run.",
        "range": "0 (auto) or 0.1 – 10000 px/mm",
        "default": "0 (auto)",
        "guidance": [
            "A ruler is more trustworthy. Self-calibration assumes the drawing shows the "
            "resting size and carried about 3% bias in validation.",
            "The ruler must be at the same distance from the camera as the robot, or "
            "perspective introduces error.",
            "Strain outputs are ratios and need no calibration at all — prefer them when "
            "absolute size is not essential.",
        ],
    },

    # --------------------------------------------------------------- output
    "decode_scale": {
        "title": "Decode scale",
        "short": "Downscale frames before analysis, trading accuracy for speed.",
        "what": "Frames are resized during decoding. Halving resolution cuts decode "
                "and segmentation cost roughly fourfold, at the price of coarser "
                "measurement.",
        "range": "1.0, 0.5 or 0.25",
        "default": "1.0 (full)",
        "guidance": [
            "0.5 is a reasonable default for 4K footage of a large robot.",
            "Stay at 1.0 when the robot is small in frame, or when measuring small "
            "contraction amplitudes.",
            "All outputs are reported in original-resolution units, so results stay "
            "comparable across settings.",
        ],
    },
    "overlay": {
        "title": "Write overlay video",
        "short": "Render the fitted outline onto a copy of the video.",
        "what": "Produces overlay.mp4 with the outline, centroid and per-frame "
                "measurements drawn on each frame. Green marks confident frames, "
                "amber marks low-confidence ones.",
        "range": "on / off",
        "default": "on",
        "guidance": [
            "The fastest way to sanity-check a run — watch it before trusting the numbers.",
            "Turn it off for batch processing; it requires a second decode pass and "
            "roughly doubles the total time.",
        ],
    },
    "gpu": {
        "title": "GPU acceleration",
        "short": "Use CUDA for segmentation and fitting.",
        "what": "Runs morphology, thresholding and the batched multi-start fit on the "
                "GPU, and uses NVDEC hardware video decoding where available.",
        "range": "on / off",
        "default": "on",
        "guidance": [
            "Turn off only to diagnose a suspected GPU problem — CPU is far slower.",
            "If this shows as unavailable, the PyTorch build is CPU-only or the NVIDIA "
            "driver needs attention.",
        ],
    },
    "keying": {
        "title": "Keying",
        "short": "Whether the robot is found by color or by brightness.",
        "what": "Brightness compares each frame against a median background plate. "
                "Color ignores brightness entirely and measures how far each pixel's "
                "hue sits from the medium's own, in CIELAB a*b*. Auto measures the "
                "separation in your clip and picks — and reports which it chose.",
        "range": "Auto · Color · Brightness",
        "default": "Auto",
        "guidance": [
            "Color is the right answer for a colored medium. On the reference clip "
            "the pink gel and the orange limbs are 84 a*b* units apart but only 26 "
            "gray levels apart, and a pale interior reads brighter than the limbs — "
            "which is exactly why the brightness threshold was so hard to place.",
            "Color needs no background plate, so it also removes the assumption that "
            "the robot moves far enough for a median to see through it. A robot that "
            "mostly sits still leaves a ghost of itself in the plate.",
            "Auto falls back to brightness when the colors are within 20 a*b* units, "
            "where there is nothing reliable to key on.",
            "Brightness remains the better choice for a genuinely monochrome scene.",
        ],
    },
    "color_frac": {
        "title": "Color cut",
        "short": "Where between medium and robot color the boundary sits.",
        "what": "0 puts the cut on the medium's own color and 1 puts it on the "
                "robot's, so lower includes more and higher includes less. The cut is "
                "placed as a fraction rather than by Otsu because Otsu assumes two "
                "populations of comparable size, and the robot is a few percent of "
                "the frame — it lands high and slices the body in half.",
        "range": "0.05 – 0.90 of the separation",
        "default": "0.30",
        "guidance": [
            "Raise it if medium texture or highlights are joining the mask.",
            "Lower it if the body is breaking into pieces — at 0.45 the reference clip "
            "kept only 65% of the mask in the largest fragment, against 92% at 0.30.",
            "Only applies to color keying; brightness uses the threshold below.",
        ],
    },
    "envelope": {
        "title": "Body envelope",
        "short": "How much larger than the learned body the mask may grow.",
        "what": "Fragments are regrouped across occlusion gaps by proximity. This caps "
                "how far that can go: the grouped mask may not exceed this multiple of "
                "the body size learned from the clip's clearest frames.",
        "range": "1.0 – 4.0 × body extent",
        "default": "1.10",
        "guidance": [
            "Without a cap, proximity alone sweeps up every speck within one body "
            "length — on the reference clip that produced a mask spanning the whole "
            "frame from 13 fragments, and a fit that ran to twice the frame height.",
            "Raise it if a genuinely long occlusion is splitting the robot further "
            "apart than the cap allows.",
            "Lower it toward 1.0 when the scene is busy and nothing occludes the robot.",
        ],
    },
    "known_width": {
        "title": "True width",
        "short": "The robot's real width, used as the measurement ruler.",
        "what": "Scale comes from the robot's own width: it is rigid while the "
                "length contracts, so it is a known length present in every frame. "
                "Normally the width is taken from the drawing; type a value here to "
                "override it, or to get micrometers with no DXF at all.\n\n"
                "The pixel side of that ratio is read off the <b>first frame of the "
                "run</b> — one frame, fixed the moment Run is pressed — not averaged "
                "over the clip. A ruler should not change because you analysed more "
                "of the video or moved the confidence floor. If the opening frame "
                "has no usable width the search walks forward to the first that "
                "does, and the run summary names which frame it landed on.",
        "range": "0 (use the drawing) – 10000 mm",
        "default": "from drawing",
        "guidance": [
            "Without either a drawing or a value here there is no scale, and every "
            "plot and column stays in pixels.",
            "Overriding is the right move when the fabricated part differs from the "
            "drawing — measure one under a microscope and type that.",
            "Check the width CV in the run summary afterwards. Under about 3% the "
            "rigid-width assumption holds; well above it, the scale is approximate.",
        ],
    },
    "dxf_scale": {
        "title": "Drawing scale",
        "short": "Multiplier on the drawing's dimensions.",
        "what": "Scales the outline read from the DXF. Use it when the drawing is "
                "at a detail scale, in the wrong units, or of a design that was "
                "fabricated slightly larger or smaller than drawn. The millimeter "
                "readout underneath shows the result, so you can dial it until it "
                "matches what you measured on the bench.",
        "range": "0.0001 – 1000 ×",
        "default": "1",
        "guidance": [
            "This is not cosmetic. The robot's width is the calibration ruler, so a "
            "drawing at 5:1 makes every micrometer in the output five times too large "
            "— and nothing else in the run would look wrong.",
            "Legs.DXF measures 26.25 × 63.00 mm as drawn against a true 5.25 × 12.60 mm, "
            "so it needs × 0.2.",
            "Do not work the number out by hand: enter the measured width under True "
            "width and press 'Set scale from true width'. A drawing at 5:1 and one in "
            "centimetres are indistinguishable from inside the file, so one measured "
            "dimension is what settles it.",
            "The Outline list is labeled at this scale, so the dimensions there are "
            "the real ones once the scale is right.",
            "Check the aspect ratio too: if the drawn ratio and the tracked ratio "
            "disagree, the problem is the chosen outline, not the scale.",
        ],
    },
    "force_method": {
        "title": "Force method",
        "short": "How the tracked length is turned into force.",
        "what": "Two routes to the same number. A <b>simulated LUT</b> interpolates a Length–Force curve exported from a COMSOL model of your device: it carries whatever that simulation captured, including behavior no closed-form expression describes, and its domain is whatever was simulated — lengths outside it are clamped rather than extrapolated. The <b>Cvetkovic Model</b> needs no simulation and no calibration run, computing force analytically from the robot's geometry and material.<br><br>The Cvetkovic model computes force from the robot's own mechanics rather than from a calibration. A muscle ring anchored below the beam contracts and pulls the leg tips together; the legs rotate as rigid links about their bases, that rotation is imposed on the ends of the compliant beam, and the beam's bending stiffness is what resists. Read backwards, the pull-in measured on video gives the force that caused it.<br><br><b>&nbsp;&nbsp;I = t³·w / 12</b><br><b>&nbsp;&nbsp;θ = asin( (δ/2) / L_leg )</b><br><b>&nbsp;&nbsp;M = 2·E·I·θ / L</b><br><b>&nbsp;&nbsp;F = M / l = 2·E·I·θ / (l·L)</b><br><br><b>δ</b> is the shortening from rest — the resting leg separation minus the separation now — and is the only quantity that comes from the video; everything else is fixed geometry or material. <b>L_leg</b> is the average leg length, and dividing the pull-in between the two legs and taking the arcsine converts a horizontal displacement into the angle each leg has swung through. <b>t</b> is the beam thickness in the bending direction and <b>w</b> its width; together they give <b>I</b>, the second moment of area, which is how strongly the cross-section resists bending — note it goes as the <i>cube</i> of thickness. <b>E</b> is Young's modulus of the hydrogel the beam is cast from, the material's intrinsic stiffness. <b>L</b> is the span that bends, measured leg center to leg center, and appears because a longer beam rotates further for the same moment. <b>M</b> is that moment: the beam's elastic reaction to being rotated by θ at both ends. <b>l</b> is the moment arm — the perpendicular distance from the beam's neutral axis down to the muscle's line of action — and converts the moment back into the force <b>F</b> that produced it.<br><br>Cvetkovic et al. (PNAS 2014) write the same physics as P = 8·E·I·δ_max/(l·L²), in terms of the beam's transverse mid-span deflection δ_max rather than the leg rotation. The two are algebraically identical — substituting δ_max = θ·L/4 turns one into the other, verified numerically to 0.00% across pull-ins from 0.05 to 1.5 mm. The rotation form is used here because it takes the quantity a top-down camera can actually measure.<br><br>With E in pascals and every length in millimeters, E·I/(l·L) evaluates to Pa·mm² = 10⁻⁶ N, so <b>force emerges in micronewtons with no conversion factor</b> — the unit this literature reports, against roughly 395 µN of active tension and 534–1147 µN of passive tension in that paper.<br><br>Two sensitivities are worth carrying in mind. Force is <i>directly proportional to E</i>, and a cast hydrogel's modulus varies between batches and drifts in culture, which makes it the least certain quantity in the calculation; because it enters linearly, a force can be rescaled afterwards without re-tracking. And because I depends on t³, a ten percent error in a callipered thickness becomes a thirty-three percent error in force.<br><br>Running the two against each other is worthwhile: agreement is real evidence, while a systematic offset almost always points at the modulus, since force scales linearly with it.",
    },
    "beam_E": {
        "title": "Young's modulus",
        "short": "Elastic modulus of the beam material, the E in the force equation.",
        "what": "The intrinsic stiffness of the hydrogel the beam is cast from, entering the force equation <b>F = 2·E·I·θ / (l·L)</b> as a direct multiplier: double it and every force in the output doubles. That linearity is also the convenient part — a force computed with one modulus can be rescaled to another afterwards without re-running the tracking. It is nonetheless the least certain quantity in the whole calculation, because a cast hydrogel's modulus varies from batch to batch and drifts over time in culture, so a force figure is worth recording alongside the batch and the measurement it came from.",
    },
    "beam_geom": {
        "title": "Beam geometry",
        "short": "The dimensions the bending calculation is built from.",
        "what": "These are the fixed terms of the force equation, all measured on the fabricated part rather than taken from the drawing.<br><br>The Cvetkovic model computes force from the robot's own mechanics rather than from a calibration. A muscle ring anchored below the beam contracts and pulls the leg tips together; the legs rotate as rigid links about their bases, that rotation is imposed on the ends of the compliant beam, and the beam's bending stiffness is what resists. Read backwards, the pull-in measured on video gives the force that caused it.<br><br><b>&nbsp;&nbsp;I = t³·w / 12</b><br><b>&nbsp;&nbsp;θ = asin( (δ/2) / L_leg )</b><br><b>&nbsp;&nbsp;M = 2·E·I·θ / L</b><br><b>&nbsp;&nbsp;F = M / l = 2·E·I·θ / (l·L)</b><br><br><b>δ</b> is the shortening from rest — the resting leg separation minus the separation now — and is the only quantity that comes from the video; everything else is fixed geometry or material. <b>L_leg</b> is the average leg length, and dividing the pull-in between the two legs and taking the arcsine converts a horizontal displacement into the angle each leg has swung through. <b>t</b> is the beam thickness in the bending direction and <b>w</b> its width; together they give <b>I</b>, the second moment of area, which is how strongly the cross-section resists bending — note it goes as the <i>cube</i> of thickness. <b>E</b> is Young's modulus of the hydrogel the beam is cast from, the material's intrinsic stiffness. <b>L</b> is the span that bends, measured leg center to leg center, and appears because a longer beam rotates further for the same moment. <b>M</b> is that moment: the beam's elastic reaction to being rotated by θ at both ends. <b>l</b> is the moment arm — the perpendicular distance from the beam's neutral axis down to the muscle's line of action — and converts the moment back into the force <b>F</b> that produced it.<br><br>Cvetkovic et al. (PNAS 2014) write the same physics as P = 8·E·I·δ_max/(l·L²), in terms of the beam's transverse mid-span deflection δ_max rather than the leg rotation. The two are algebraically identical — substituting δ_max = θ·L/4 turns one into the other, verified numerically to 0.00% across pull-ins from 0.05 to 1.5 mm. The rotation form is used here because it takes the quantity a top-down camera can actually measure.<br><br>With E in pascals and every length in millimeters, E·I/(l·L) evaluates to Pa·mm² = 10⁻⁶ N, so <b>force emerges in micronewtons with no conversion factor</b> — the unit this literature reports, against roughly 395 µN of active tension and 534–1147 µN of passive tension in that paper.<br><br>Two sensitivities are worth carrying in mind. Force is <i>directly proportional to E</i>, and a cast hydrogel's modulus varies between batches and drifts in culture, which makes it the least certain quantity in the calculation; because it enters linearly, a force can be rescaled afterwards without re-tracking. And because I depends on t³, a ten percent error in a callipered thickness becomes a thirty-three percent error in force.",
    },
    "leg_marks": {
        "title": "Leg points",
        "short": "Measure the force from how far two marked points close, not "
                 "from the robot's overall length.",
        "what": "Click two points on the video — one on each leg — and the "
                "distance between them is tracked frame by frame and becomes "
                "the length the force is computed from. Drag either mark to "
                "adjust it. Untick to go back to measuring the whole body.<br><br>"
                "<b>Why this exists.</b> The beam model's δ is how much closer "
                "the two legs are than at rest, and that is not the same as how "
                "much shorter the robot got. Both of the other measurements — "
                "the silhouette's long-axis extent, and the fitted CAD outline — "
                "describe the robot as one shape that translates, rotates and "
                "scales. A muscle-driven robot does not deform that way: the "
                "beam does not shorten, the legs pivot inward and the rest of "
                "the body comes along. A whole-body measurement spreads the leg "
                "closure over the entire outline and reports a fraction of it.<br><br>"
                "Against a colleague's frame-by-frame manual tracking of the "
                "same clip, the silhouette reported <b>0.759 mm of length change "
                "per 1 mm of real leg closure</b>, with a correlation of 0.993 — "
                "following the motion perfectly and reporting the wrong "
                "quantity, which is the failure that does not look like one. "
                "Two marked points recover a slope of 0.998 and a force within "
                "1% of the manual result.<br><br>"
                "Marking the legs in the DXF instead would not work, and it is "
                "worth knowing why: under a fitted pose, two template-fixed "
                "points end up a <i>constant multiple</i> of the fitted length "
                "apart. They carry no information the length column did not "
                "already have. The marks have to be tracked in the image, "
                "independently of the body, which is what this does.",
        "range": "two points on the frame",
        "default": "off — force from the overall length",
        "guidance": [
            "Tick the box, then press 'Mark on video' and click the two points. "
            "While marking is on, the region and the placed outline stop "
            "responding to clicks, so nothing competes for them; press 'Done "
            "marking' to hand the picture back.",
            "Mark the two ends of the beam, where it meets the legs. That is the "
            "pair every other term in the model is written for: 'Leg to leg' is "
            "their span and the leg angle is what carries them together. The run "
            "checks this for you — at rest the marks should sit 'Leg to leg' "
            "apart, and the summary says so if they do not.",
            "Marks further out along the legs swing further for the same "
            "rotation and read high; marks inboard read low. Neither looks like "
            "anything but a plausible force, so the resting span is the check "
            "worth doing.",
            "Pick features with structure — a pad, a corner, a printed edge. A "
            "patch of flat tissue has nothing to lock onto and will wander.",
            "Marks are pixel coordinates in one recording and are cleared when a "
            "new clip is loaded, exactly like the region.",
            "Frames where a mark could not be solved hold their last position "
            "and are counted in the summary. A held frame is a flat spot, so it "
            "shallows any contraction it lands in.",
        ],
    },

    "beam_rest": {
        "title": "Resting length",
        "short": "The length that the shortening δ is measured from.",
        "what": 'Deflection is <b>δ = resting length − current length</b>, so this control sets the zero of the force scale; every force in the run is measured relative to it. <b>Maximum (Cvetkovic Model)</b> takes the longest length seen anywhere in the clip, which is correct when the muscle only ever shortens the robot and is what the reference implementation does. It is also, by construction, the single most extreme sample in the recording, so one over-long tracked frame sets the baseline for everything — in testing, a single bad frame moved the mean force by 830 µN. <b>Robust</b> takes the median of the upper quartile instead, which ignores a handful of outliers at the cost of no longer reproducing a reference analysis exactly. Whichever is used, the resting length actually applied is printed in the run summary and recorded in run_info.json.',
    },
    "force_lut": {
        "title": "Simulated LUT",
        "short": "A Length–Force curve from a COMSOL simulation of the device.",
        "what": 'A two-column CSV of lengths and the forces your simulation predicts for them. Units are read from the header text, so <i>Length (um)</i>, <i>Force (mN)</i> and <i>Load (gf)</i> are all understood and column order does not matter; a file that states no units is treated as millimeters and millinewtons, with that assumption written into the log rather than left silent, because a thousand-fold unit error is exactly the kind that survives review with every number still looking plausible. Between tabulated points the curve is interpolated linearly. Beyond either end it is clamped rather than extrapolated, and the number of frames that fell outside is reported: a simulated curve has a real domain, and continuing its end slope into lengths that were never simulated invents stiffness data, most confidently in exactly the regime where the device is least linear.',
    },
    "view_mode": {
        "title": "View",
        "short": "The photograph, or the color distance the segmenter works on.",
        "what": "Color distance shows how far each pixel's hue sits from the medium's, "
                "which is the surface the threshold cuts through. The robot appears "
                "bright against a dark background regardless of how similar the "
                "brightnesses are.",
        "range": "Video · Color distance (b*)",
        "default": "Color distance (b*)",
        "guidance": [
            "Switch to it when the mask looks wrong. A threshold problem is obvious "
            "here and near-invisible on the photograph.",
            "If the robot is dim in this view, the colors are genuinely close and "
            "brightness keying may do better — check the header chip for which mode "
            "was chosen.",
            "Only meaningful in color keying; in brightness mode the video is shown.",
        ],
    },
    "video_queue": {
        "title": "Videos in this folder",
        "short": "Every clip sitting next to the one you opened, and which of them "
                 "you have already analyzed.",
        "what": "Lists the video files in the same folder as the current clip, in "
                "the order a file manager would show them — IMG_2 before IMG_10, "
                "not after it. Click one to load it. A filled green dot means a "
                "results folder for that clip already exists inside the output "
                "folder, so it has been run; a hollow dot means it has not.",
        "guidance": [
            "The dot is read from the filesystem each time the list refreshes, so "
            "deleting a results folder to redo a clip immediately shows it as "
            "unanalyzed again.",
            "The dot only knows a folder exists. It cannot tell a good run from a "
            "bad one — check the fit confidence for that.",
            "Next video below the plots walks down this list, keeping your "
            "parameters and drawing and clearing the fit.",
            "The line underneath estimates how long the unanalyzed clips will "
            "take. It multiplies each clip's frames by its pixel count and "
            "applies the median rate measured from your own finished runs on "
            "this machine, so it is silent until at least one run has been "
            "timed and it gets sharper as you work.",
            "That estimate assumes the remaining clips resemble the ones "
            "already done. A clip that needs far more restarts, or that keeps "
            "losing the fit, will overrun it.",
        ],
    },
    "roi": {
        "title": "Region of interest",
        "short": "Track only inside a rectangle you draw on the video.",
        "what": "Restricts both the color model and the mask to the rectangle. "
                "Everything outside it is dimmed on screen and ignored entirely "
                "by the analysis. The rectangle is stored in full-resolution "
                "pixels, so it stays put when you change the decode scale.",
        "guidance": [
            "Draw one whenever something outside the dish is brighter or more "
            "saturated than the robot. The color model picks the background as "
            "the commonest hue and the robot as the farthest populated hue from "
            "it — a lamp reflection, a dish rim or the bench beyond the well "
            "will win that contest simply for being far away and numerous, and "
            "the fit then locks onto it.",
            "It narrows the color *estimate*, not just the mask. That is the "
            "part that matters: clipping a bad mask leaves a bad threshold, "
            "while estimating inside the region gives the medium and the robot "
            "their real separation.",
            "Leave a margin. A region drawn tight against the robot clips it "
            "the moment it moves, and a clipped silhouette measures short.",
            "The shading shows what you excluded rather than hiding it, so a "
            "region that has cut the robot in half is obvious instead of "
            "looking like a tracking failure.",
        ],
    },
    "appearance": {
        "title": "Appearance lock",
        "short": "Track the robot by aligning how it looks, instead of by "
                 "thresholding its color.",
        "what": "Takes a patch of the frame you are currently looking at, "
                "inside the region you drew, and finds the rotation, "
                "translation and two scales that best align it to every "
                "later frame. The length scale it recovers is the "
                "measurement. It optimises correlation, so a change in "
                "lighting or exposure moves it hardly at all.",
        "guidance": [
            "Use it when color keying cannot work — when the robot and the "
            "medium overlap in every channel, or something outside the dish "
            "is brighter than the robot. On such footage the per-pixel "
            "separability measures around 1.4 where a threshold needs above "
            "2, while the spatial pattern is unmistakable.",
            "Draw a region first, and scrub to a frame where the robot is "
            "clearly visible and the fit is good. That frame is what gets "
            "learned; everything after is measured relative to it.",
            "With a DXF placed on the reference frame the results are "
            "absolute — micrometers, force, the lot. Without one the scales "
            "are ratios, so strain, frequency and the trajectory are right "
            "but absolute size is withheld rather than guessed.",
            "It follows whatever you pointed it at. Aim it at a bubble and it "
            "will track the bubble faithfully and report high confidence "
            "doing it — the confidence is the correlation, which says the "
            "alignment is good, not that the target was the right one.",
        ],
    },
    "features": {
        "title": "Fit interior features",
        "short": "Also match the drawing's holes, windows and beams, not just its silhouette.",
        "what": "Samples the closed loops inside the outer boundary as extra points "
                "and matches them against edges in the image. The silhouette alone "
                "fixes position, rotation and the two scales but says nothing about "
                "the inside of the body, so a boundary that segmentation renders "
                "slightly wrong has nothing to correct it. Interior structure makes "
                "the fit over-determined — which is the point: it is what lets you "
                "threshold tightly without the fit becoming unstable.",
        "range": "on / off (needs a DXF with interior loops)",
        "default": "on",
        "guidance": [
            "It pairs with a tighter color cut, which is what it is for. Measured on "
            "the reference clip at cut 0.45 and 0.60, width CV improved from 2.17% to "
            "1.90% and from 2.58% to 1.90%.",
            "At a loose cut it can make things worse — at 0.30 the mask is bloated and "
            "the interior edges are unreliable. Raise the cut and turn this on together.",
            "Interior edges are taken from the color-distance image, not the mask: an "
            "aggressively thresholded mask is a solid blob with no interior at all "
            "(zero enclosed holes on the reference clip, at any morphology setting).",
            "Watch the interior percentage in the readout and the run summary. It is "
            "the fraction of interior points landing on a real edge; around 20% on the "
            "reference drawing, and a low number means the drawing's inner loops do "
            "not correspond to anything the camera resolves.",
            "Only the silhouette carries the containment term, so turning this on "
            "never changes what the confidence gate means.",
        ],
    },
    "feature_weight": {
        "title": "Feature weight",
        "short": "How much the interior counts relative to the silhouette.",
        "what": "The two are combined as a weighted mean of their own means, so the "
                "balance does not drift with how many points each carries. At 1.0 the "
                "outer boundary and the interior contribute half each; at 0 the "
                "interior is ignored entirely.",
        "range": "0 – 5",
        "default": "1.0",
        "guidance": [
            "Lower it when the interior percentage is small — you are then weighting "
            "points that mostly are not matching anything.",
            "Raise it when the silhouette is the unreliable part, e.g. a soft or "
            "poorly-lit outer edge with crisp internal structure.",
        ],
    },
    "automatch": {
        "title": "Match drawing to video",
        "short": "Let the fitter decide which outline is the robot, and where.",
        "what": "Segments a few frames from across the clip, fits every candidate "
                "outline in the drawing to each of them, and keeps whichever agrees "
                "with the image best. The winner becomes the active outline and its "
                "pose becomes the manual placement, so the overlay lands on the robot "
                "without dragging anything.",
        "range": "needs a video and a DXF",
        "default": "—",
        "guidance": [
            "Geometry alone cannot tell which curve in a drawing is the part; the "
            "video can. On the reference clip this separated the true body outline at "
            "0.75 confidence from the runner-up at 0.46.",
            "Run it after the mask looks right. It judges outlines by how well they "
            "fit the mask, so a bad mask gives a confident answer to the wrong question.",
            "A best score below about 0.35 is reported as a warning — that is a "
            "segmentation problem, not an outline problem.",
            "Two identical robots in frame is the case it cannot settle. Place that "
            "one by hand.",
        ],
    },
    "plot_region": {
        "title": "Reading the plots",
        "short": "Right-drag to pan, left-drag to select a stretch and measure it.",
        "what": "Scroll zooms time on every panel, shift-scroll zooms one panel's "
                "value axis, right-drag pans, and left-drag marks a stretch of time. "
                "The marked region is measured immediately: average change in length "
                "and in strain, average change in force where a LUT is loaded, and "
                "locomotion speed in units per minute. Double-click clears both the "
                "selection and the zoom.",
        "range": "any stretch of the clip",
        "default": "—",
        "guidance": [
            "The delta is a mean peak-to-trough swing, not max minus min. Max minus "
            "min reports the single largest excursion in the window, which is the "
            "noisiest sample available and grows the longer you select.",
            "The cycle count beside it tells you how much the average rests on. Two "
            "cycles is an anecdote; twenty is a measurement.",
            "Two speeds are reported. The path slope is regressed on the cumulative "
            "path; the net rate is straight-line start-to-finish over the same window.",
            "Cumulative path only ever increases, so every wobble of the centroid adds "
            "distance the robot never traveled — the path slope reads as locomotion "
            "plus jitter. With 0.6 px of centroid noise at 30 Hz it measured 50.4 "
            "mm/min against a true 3.18 mm/min. Net displacement cannot do that.",
            "A path-to-net ratio above about 2 is flagged: at that point the centroid "
            "is wandering more than the robot is traveling, and the net rate is the "
            "one to quote.",
            "Fit confidence is not measured — it describes the tracker, not the robot.",
            "Whatever is selected when a run starts is drawn and annotated on the "
            "exported figure and recorded under region_analysis in run_info.json.",
        ],
    },
    "playback": {
        "title": "Playback",
        "short": "Play the clip with live segmentation. Space toggles it.",
        "what": "Streams the video sequentially with the mask drawn on top, at a "
                "chosen fraction of real time. Sequential decoding is far faster than "
                "the seek-per-frame the scrubber uses, so this is the quick way to "
                "check that the mask holds across a whole clip rather than on the one "
                "frame you happened to stop at.",
        "range": "0.25× to 4× real time",
        "default": "1×",
        "guidance": [
            "Shape fitting is skipped while playing — it is the expensive stage, and "
            "the fit returns as soon as you pause.",
            "Frames are dropped rather than queued if the machine cannot keep up, so "
            "the clock stays honest.",
            "Manual placement handles are hidden during playback; pause to adjust.",
        ],
    },
    "dxf_outline": {
        "title": "Outline",
        "short": "Which closed curve in the drawing is the robot.",
        "what": "A production DXF is a sheet, not a bare outline: a page border, a "
                "title block, dimension lines and often several views of the part. "
                "robotrack assembles every closed curve it can from the drawing's "
                "individual lines and arcs, discards the ones that look like sheet "
                "furniture, and offers the rest largest first. This control appears "
                "only when more than one candidate survives.",
        "range": "one of the outlines found in the file",
        "default": "the largest non-border closed outline",
        "guidance": [
            "Check it against the video — the placement overlay draws the outline you "
            "have chosen, so a wrong pick is obvious the moment you see it on the robot.",
            "Dimensions listed beside each option are the real millimeters from the "
            "drawing; the one matching the part you measured is the one you want.",
            "An option marked [open] did not close. It is usable but the drawing has a "
            "gap in it, and the fit will be worse than a closed outline.",
            "Changing this clears any manual placement, because the scale factors are "
            "per template millimeter and would otherwise mean a different size.",
        ],
    },

    # -------------------------------------------------------- placement
    "manual_placement": {
        "title": "Place the outline by hand",
        "short": "Say which object in frame is the robot, instead of inferring it.",
        "what": "Normally the starting pose is derived from the mask's own moments: "
                "its centroid, principal axis and extents. That is correct when the "
                "robot is the only moving thing in view. Turning this on lets you drag "
                "the CAD outline onto the target yourself, and that pose is used as the "
                "starting guess for the fit — including after a recovery, so the "
                "tracker returns to the right object rather than re-deciding from "
                "whatever the mask contains.",
        "range": "on / off (requires a DXF)",
        "default": "off",
        "guidance": [
            "Turn it on when two robots, a reflection, a bubble or a tether are in "
            "frame — a moment-based seed describes all of them at once and the fit "
            "then locks onto the wrong thing with high confidence.",
            "Turn it on when the reported orientation is 180° out. A near-symmetric "
            "body gives the automatic seed no way to tell head from tail; placing it "
            "by hand does.",
            "It is a seed, not a constraint. The fit is free to move away from it, so "
            "a rough placement within a body width is enough.",
            "Place it on a frame where the robot is clearly visible, not one where it "
            "is behind the obstacle.",
            "Scale matters as much as position: the placed size sets the bounds the "
            "search is allowed to explore, so place it at roughly the resting size.",
        ],
    },

    # ---------------------------------------------------------- updates
    "update_channel": {
        "title": "Updates",
        "short": "Where robotrack looks for new versions.",
        "what": "A folder every copy can reach — a synced Nextcloud folder or a lab "
                "share — or a GitHub repository written as github:owner/repo. Normal "
                "updates carry only robotrack's own Python, a few hundred kilobytes, "
                "and are applied without reinstalling the multi-gigabyte bundle. A "
                "release that changes a dependency ships the full installer instead, "
                "and says so before anything is downloaded.",
        "range": "folder path, https URL, or github:owner/repo",
        "default": "none",
        "guidance": [
            "Nothing is downloaded until you have seen the version number and its "
            "notes.",
            "Checking at launch only interrupts you when something is actually "
            "available.",
            "Downloads are checked against the checksum in the manifest, which catches "
            "a truncated file or a half-finished sync. Point this only at a location "
            "you control — whoever can write the manifest can write the checksum.",
            "If an update will not start, robotrack disables it on the next launch and "
            "falls back to the version it shipped with, rather than failing to open.",
        ],
    },
}


def describe(key: str) -> str:
    """Plain-text rendering, for CLI help output."""
    s = HELP[key]
    lines = [s["title"], "-" * len(s["title"]), s["what"], "",
             f"Range:   {s['range']}", f"Default: {s['default']}"]
    if s.get("guidance"):
        lines += [""] + [f"  - {g}" for g in s["guidance"]]
    return "\n".join(lines)
