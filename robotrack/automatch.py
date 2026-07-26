"""Match the drawing to the robot in the video, automatically.

Two questions get answered here, and they are the same question asked twice.

**Which outline?** A production DXF offers several closed curves and only one of
them is the part. ``cad.read_loops`` ranks them by geometry alone -- it has never
seen the video, so it can only guess. But the fitter already produces exactly the
right discriminator: fit each candidate to real frames and keep the one that
agrees with the image best. On the reference clip that separates the true body
outline at 0.87 mean confidence from the runner-up at 0.59, which is not a close
call.

**Where is it?** Having chosen the outline, the same machinery returns the pose,
so the placement overlay can be dropped onto the robot without anyone dragging
anything. Manual placement stays for the cases this cannot resolve -- two
identical robots in frame, where no amount of shape evidence says which one you
meant.

Cost is bounded by construction: a handful of frames per candidate, at a reduced
restart schedule, on frames chosen from across the clip so a single unlucky one
cannot decide it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cad import Template, load_dxf, read_loops
from .register import FitConfig, ShapeFitter
from .segment import SegmentConfig, Segmenter
from .decode import FrameReader


@dataclass
class LoopScore:
    index: int
    template: Template
    confidence: float
    width_px: float
    length_px: float
    pose: list[float] | None
    n_frames: int

    def label(self) -> str:
        return (f"{self.index + 1}.  {self.template.width_mm:.1f} × "
                f"{self.template.length_mm:.1f} mm — confidence {self.confidence:.2f}")


@dataclass
class MatchResult:
    scores: list[LoopScore] = field(default_factory=list)
    best: LoopScore | None = None
    frames_used: int = 0

    def summary(self) -> str:
        if not self.best:
            return "no outline could be matched to the video"
        lines = [f"matched outline {self.best.index + 1} "
                 f"({self.best.template.width_mm:.1f} × "
                 f"{self.best.template.length_mm:.1f} mm) "
                 f"at confidence {self.best.confidence:.2f}"]
        runners = [s for s in self.scores[1:4]]
        if runners:
            lines.append("  runners-up: " + ", ".join(
                f"#{s.index + 1} {s.confidence:.2f}" for s in runners))
        return "\n".join(lines)


def collect_masks(reader: FrameReader, seg_cfg: SegmentConfig, dev,
                  n_frames: int = 5) -> list[np.ndarray]:
    """A few masks spread across the clip.

    Spread rather than consecutive: consecutive frames are nearly identical, so
    five of them is one sample with extra cost. Frames from across the clip also
    mean a single moment of heavy occlusion cannot decide the answer.
    """
    seg = Segmenter(reader, seg_cfg, dev)
    total = max(reader.info.n_frames, 1)
    want = set(np.linspace(0, total - 1, n_frames * 3).astype(int).tolist())
    out: list[np.ndarray] = []
    for m in seg:
        if m.index in want and m.mask.any():
            out.append(m.mask)
            if len(out) >= n_frames:
                break
    return out


def score_loops(dxf_path: str, masks: list[np.ndarray], dev,
                fit_cfg: FitConfig | None = None,
                max_candidates: int = 6,
                progress=None) -> MatchResult:
    """Fit every plausible outline to the masks and rank by agreement."""
    if not masks:
        return MatchResult()

    loops, _ = read_loops(dxf_path)
    # Only closed, non-border loops are worth the fit. An open chain cannot
    # bound a body, and the sheet border is already known not to be the part.
    order = [i for i, L in enumerate(loops) if L.closed and not L.is_frame]
    order = order[:max_candidates] or list(range(min(len(loops), max_candidates)))

    cfg = fit_cfg or FitConfig()
    # A cheaper schedule than a real run: this is a comparison between
    # candidates, not a measurement, and every candidate pays the same price.
    cfg = FitConfig(**{**cfg.__dict__, "n_restarts": min(cfg.n_restarts, 24),
                       "iters": min(cfg.iters, 80), "iters_warm": min(cfg.iters_warm, 40)})

    scores: list[LoopScore] = []
    for n, i in enumerate(order):
        if progress:
            progress(n, len(order))
        try:
            tpl = load_dxf(dxf_path, loop_index=i)
        except Exception:
            continue
        fitter = ShapeFitter(tpl, cfg, dev)
        confs, poses = [], []
        for mask in masks:
            fitter.prev = None          # each frame judged on its own merits
            p = fitter.fit(mask)
            if p is not None:
                confs.append(p.confidence)
                poses.append(p.as_array())
        if not confs:
            continue
        k = int(np.argmax(confs))
        best_pose = poses[k]
        scores.append(LoopScore(
            index=i, template=tpl,
            # The median, not the mean: one badly occluded frame should not sink
            # an otherwise correct outline.
            confidence=float(np.median(confs)),
            width_px=float(best_pose[3] * tpl.width_mm),
            length_px=float(best_pose[4] * tpl.length_mm),
            pose=[float(x) for x in best_pose],
            n_frames=len(confs),
        ))

    scores.sort(key=lambda s: s.confidence, reverse=True)
    return MatchResult(scores=scores, best=scores[0] if scores else None,
                       frames_used=len(masks))
