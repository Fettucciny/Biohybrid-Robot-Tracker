"""Command-line entry point:  robotrack video.mov --dxf robot.dxf"""
from __future__ import annotations
import argparse, sys
from .pipeline import RunConfig, run
from .ingest import probe

def _beam(a):
    from .forcemodel import BeamForceModel
    return BeamForceModel(E_pa=a.beam_E_kpa * 1000.0,
                          thickness_mm=a.beam_thickness,
                          beam_width_mm=a.beam_width,
                          L_mm=a.beam_leg_to_leg,
                          l_mm=a.beam_offset,
                          leg_long_mm=a.beam_leg_long,
                          leg_short_mm=a.beam_leg_short)


def main(argv=None) -> int:
    p = argparse.ArgumentParser("robotrack", description=__doc__)
    p.add_argument("video")
    p.add_argument("--dxf", help="2D DXF outline of the robot (enables occlusion-tolerant fitting)")
    p.add_argument("--dxf-scale", type=float, default=1.0, metavar="K",
                   help="multiply the drawing's dimensions (Legs.DXF needs 0.2)")
    p.add_argument("--no-features", action="store_true",
                   help="ignore the drawing's interior structure; silhouette only")
    p.add_argument("--force-lut", metavar="CSV",
                   help="Length,Force calibration curve; adds a force column")
    p.add_argument("--force-beam", action="store_true",
                   help="compute force from beam mechanics instead of a LUT")
    p.add_argument("--beam-E-kpa", type=float, default=293.0)
    p.add_argument("--beam-thickness", type=float, default=1.1, metavar="MM")
    p.add_argument("--beam-width", type=float, default=1.925, metavar="MM")
    p.add_argument("--beam-leg-to-leg", type=float, default=8.25, metavar="MM")
    p.add_argument("--beam-offset", type=float, default=1.642, metavar="MM")
    p.add_argument("--beam-leg-long", type=float, default=4.125, metavar="MM")
    p.add_argument("--beam-leg-short", type=float, default=3.3, metavar="MM")
    p.add_argument("--dxf-loop", type=int, default=0, metavar="N",
                   help="which outline in the drawing is the robot (see --list-outlines)")
    p.add_argument("--list-outlines", action="store_true",
                   help="list the candidate outlines in --dxf and exit")
    p.add_argument("-o", "--outdir", default="results")
    p.add_argument("--px-per-mm", type=float, default=None)
    p.add_argument("--scale", type=float, default=1.0, help="decode downscale, e.g. 0.5")
    p.add_argument("--smooth-ms", type=float, default=100.0)
    p.add_argument("--tau-px", type=float, default=12.0, help="robust kernel scale")
    p.add_argument("--restarts", type=int, default=64)
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    a = p.parse_args(argv)

    if a.list_outlines:
        if not a.dxf:
            print("--list-outlines needs --dxf", file=sys.stderr); return 2
        from .cad import describe_loops
        print(describe_loops(a.dxf)); return 0

    if a.probe_only:
        print(probe(a.video).summary()); return 0

    cfg = RunConfig(video=a.video, dxf=a.dxf, dxf_loop_index=a.dxf_loop,
                    dxf_scale=a.dxf_scale, use_features=not a.no_features,
                    force_lut=a.force_lut,
                    force_method=("beam" if a.force_beam
                                  else "lut" if a.force_lut else "none"),
                    beam=(_beam(a) if a.force_beam else None),
                    outdir=a.outdir, px_per_mm=a.px_per_mm,
                    scale=a.scale, write_overlay=not a.no_overlay, gpu=not a.cpu)
    cfg.analysis.smooth_ms = a.smooth_ms
    cfg.fit.tau_px = a.tau_px
    cfg.fit.n_restarts = a.restarts

    def prog(i, n):
        print(f"\r  frame {i}/{n or '?'}", end="", file=sys.stderr, flush=True)

    res = run(cfg, progress=prog)
    print(f"\r{' '*40}\r", end="", file=sys.stderr)
    print(res.summary())
    # Report where the files actually landed, not where they were asked for:
    # each clip now gets its own subfolder under --outdir, and printing the
    # parent would send anyone following the message to an empty directory.
    from .pipeline import output_dir
    dest = output_dir(a.outdir, a.video)
    print(f"\nwrote {dest}/tracking.csv, summary.png, run_info.json"
          + ("" if a.no_overlay else ", overlay.mp4"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
