"""Generate a synthetic muscle-robot clip with exactly known ground truth.

Simulates the real measurement problem: a tapered body that contracts
periodically (length down, width up, as an incompressible muscle would),
translates along a curved path, and passes behind a static obstacle for part of
the clip. Because every quantity is prescribed, the recovered values can be
checked against truth rather than merely eyeballed.

The palette is not decorative. The analysis keys on *chroma* -- the a* and b*
channels of CIELAB -- because in the real footage the robot and its culture
medium differ by about 84 units of chroma and only about 26 of luma, which is
what made grayscale thresholding fail. A neutral-grey fixture would therefore
test a path the software no longer takes, and would fail for reasons that say
nothing about it. So the background is the medium's deep pink and the body is
the tissue's orange-yellow, at roughly the separation the camera actually
delivers. The obstacle is a dark neutral, as a real one is.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import ezdxf
import numpy as np

# Ground truth, in the units the analysis should recover.
PX_PER_MM = 10.0
L0_MM, W0_MM = 20.0, 6.0          # resting length / width
FREQ_HZ = 1.5                      # contraction frequency
LEN_AMP, WID_AMP = 0.12, 0.08      # fractional stroke
TRAVEL_MM_S = 4.0                  # net locomotion speed
# Horizontal bar the robot passes behind. Deliberately narrower than the robot
# is long, so a *portion* is hidden and the rest stays visible -- the case the
# robust fit is supposed to handle. A bar wider than the robot hides it
# completely, which no fitting method can recover, only interpolation.
OBSTACLE_Y = (300, 372)            # image px; ~35% of the 200 px body length
OBSTACLE_X0 = 380                  # bar starts here, so the robot enters it mid-clip

# BGR, matching the real domain: deep pink medium, orange body. The values are
# chosen so the CIELAB separation between body and background reproduces what
# the camera actually delivers -- about 27 units of luma against 73 of chroma.
# The low luma figure is the point: it is what makes a grayscale threshold fail
# and a chroma key succeed, so a fixture that got it wrong by being obligingly
# bright would pass whether or not the colour path worked.
BG_BGR = (150, 70, 205)
BODY_BGR = (45, 93, 115)
EDGE_BGR = (34, 70, 92)
# The occluder is a *shadow on the medium*, not a neutral-grey bar: same hue,
# 46 CIELAB units darker. That is deliberately the harder and more honest case.
# A neutral bar is maximally distant from a saturated pink in a*b*, so a chroma
# key would treat it as the largest object in the frame and lock onto it -- a
# failure of the fixture, not of the method. A same-hue shadow is invisible to
# the chroma key while being glaring to a luma threshold, which is exactly the
# asymmetry the colour path exists to exploit. It still removes robot pixels
# wherever it overlaps, so the occlusion test is unchanged.
OBSTACLE_BGR = (98, 43, 135)          # == _shade(BG_BGR, 0.40)


def _shade(bgr, k: float) -> tuple[int, int, int]:
    """Scale an sRGB colour's brightness by ``k`` while holding its hue.

    Scaling the encoded bytes directly does not: sRGB is gamma-encoded, so a
    uniform multiply shifts chromaticity as well as brightness. Decoding to
    linear light, scaling, and re-encoding is the operation that corresponds to
    "the same paint under a dimmer lamp", which is what an occluding shadow and
    an uneven illumination field both are.
    """
    u = np.asarray(bgr, np.float64) / 255.0
    lin = np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4) * k
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return tuple(int(v) for v in np.clip(np.round(srgb * 255), 0, 255))


def body_outline(n: int = 400) -> np.ndarray:
    """Tapered ellipse-like body, in mm, centred, long axis = +y."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    y = (L0_MM / 2) * np.cos(th)
    taper = 1.0 - 0.35 * (np.abs(y) / (L0_MM / 2)) ** 2.2
    x = (W0_MM / 2) * np.sin(th) * taper
    return np.stack([x, y], 1)


def write_dxf(path: Path) -> None:
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4          # millimetres
    doc.modelspace().add_lwpolyline(body_outline(240), close=True)
    doc.saveas(str(path))


def render(outdir: Path, fps: float, seconds: float, size=(1280, 720),
           occlude=True) -> tuple[Path, np.ndarray]:
    W, H = size
    n = int(round(fps * seconds))
    tpl = body_outline()
    truth = []

    # Textured static background so median-plate estimation is exercised properly.
    rng = np.random.default_rng(7)
    bg = (rng.normal(BG_BGR, 5, (H, W, 3))).clip(0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    for _ in range(60):
        # Texture as *uneven illumination*, not as a second colour: each blob is
        # the medium's own hue scaled in linear light, which changes brightness
        # by up to 40% and chromaticity by almost nothing. That is what real
        # lighting does, and it is the case the chroma key claims to be immune
        # to. Randomising the BGR triple instead would scatter a*b* far enough
        # to clear the key's threshold, and the resulting speckle would be a
        # property of the fixture rather than of anything the software sees.
        c = _shade(BG_BGR, float(rng.uniform(0.55, 1.35)))
        cv2.circle(bg, tuple(rng.integers([0, 0], [W, H])), int(rng.integers(8, 40)), c, -1)
    bg = cv2.GaussianBlur(bg, (0, 0), 2)

    raw = outdir / "_frames.raw"
    with open(raw, "wb") as fh:
        for i in range(n):
            t = i / fps
            sl = 1 + LEN_AMP * np.sin(2 * np.pi * FREQ_HZ * t)
            sw = 1 - WID_AMP * np.sin(2 * np.pi * FREQ_HZ * t)
            cx_mm = 30.0 + TRAVEL_MM_S * t
            cy_mm = 36.0 + 3.0 * np.sin(2 * np.pi * 0.25 * t)
            ang = np.deg2rad(8.0)

            p = tpl * np.array([sw, sl])
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            p = p @ R.T + np.array([cx_mm, cy_mm])
            px = (p * PX_PER_MM).astype(np.int32)

            img = bg.copy()
            cv2.fillPoly(img, [px], BODY_BGR)
            cv2.polylines(img, [px], True, EDGE_BGR, 2)
            img = cv2.GaussianBlur(img, (0, 0), 0.8)
            img = np.clip(img + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)

            occluded_now = False
            if occlude:
                # Static, present in every frame: it joins the background
                # plate in luma mode, and shares the medium's hue in colour
                # mode. Either way it removes robot pixels rather than being
                # mistaken for robot.
                cv2.rectangle(img, (OBSTACLE_X0, OBSTACLE_Y[0]), (W, OBSTACLE_Y[1]),
                              OBSTACLE_BGR, -1)
                occluded_now = bool(px[:, 1].max() > OBSTACLE_Y[0]
                                    and px[:, 1].min() < OBSTACLE_Y[1]
                                    and px[:, 0].max() > OBSTACLE_X0)

            fh.write(img.tobytes())
            truth.append((i, t, cx_mm, cy_mm, sl * L0_MM, sw * W0_MM, occluded_now))

    video = outdir / f"synthetic_{int(round(fps))}fps.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", f"{fps}", "-i", str(raw),
         "-c:v", "libx264", "-preset", "medium", "-crf", "16",
         "-pix_fmt", "yuv420p", "-r", f"{fps}", str(video)],
        check=True)
    raw.unlink()
    return video, np.array([r[:6] + (float(r[6]),) for r in truth], float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=120)
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--outdir", default="synthetic")
    ap.add_argument("--no-occlude", action="store_true")
    a = ap.parse_args()
    d = Path(a.outdir); d.mkdir(parents=True, exist_ok=True)
    write_dxf(d / "robot.dxf")
    v, truth = render(d, a.fps, a.seconds, occlude=not a.no_occlude)
    np.save(d / f"truth_{int(a.fps)}.npy", truth)
    print(f"wrote {v}  ({len(truth)} frames, {truth[:,6].mean()*100:.0f}% occluded)")
    print(f"wrote {d/'robot.dxf'}")
