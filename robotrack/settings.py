"""Persisted interface state, and named configuration files.

Two things that look the same and are not:

**Session memory.** Every control's value plus the last video, DXF and output
folder are written to ``settings.json`` in the per-user data directory whenever
they change, and restored on launch. Nobody should have to re-find their clip and
re-dial six parameters because they closed the window.

**Named configs.** The same dictionary can be written to a ``.rtcfg`` file the
user chooses. That is what makes a run reproducible across machines and months:
one file that says exactly how a set of clips was analyzed, alongside the data.

Both use one serialiser, so a config file is exactly the state the app restores.

Forward and backward compatibility is handled by merging over the defaults rather
than replacing them: a config written by an older build simply lacks the newer
keys and gets their defaults, and a config from a newer build has its unknown
keys ignored instead of crashing the load. A settings file that would otherwise
be a hard error is the wrong thing to fail a launch on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .update import user_dir

CONFIG_SUFFIX = ".rtcfg"
CONFIG_FILTER = f"robotrack config (*{CONFIG_SUFFIX});;JSON (*.json);;All files (*)"

# Every persisted key, with the value the app ships with. This is also the
# whitelist: anything not named here is not restored from a file.
DEFAULTS: dict[str, Any] = {
    # --- paths -----------------------------------------------------------
    "video_path": "",
    "dxf_path": "",
    "dxf_loop_index": 0,            # which candidate outline in the drawing
    "dxf_scale": 1.0,               # multiplier on the drawing's dimensions
    "known_width_mm": 0.0,          # 0 = take the ruler width from the drawing
    "force_lut_path": "",
    "force_method_index": 1,        # 0 simulated LUT (COMSOL), 1 Cvetkovic model
    # Beam model, defaulting to the values in SampleForce.m
    "beam_E_kpa": 293.0,
    "beam_thickness_mm": 1.100,
    "beam_width_mm": 1.925,
    "beam_leg_to_leg_mm": 8.250,
    "beam_muscle_offset_mm": 1.243,
    "beam_leg_long_mm": 4.125,
    "beam_leg_short_mm": 3.300,
    "beam_resting_index": 0,
    "last_lut_dir": "",
    "view_mode_index": 1,           # 0 = video, 1 = color distance (b*)
    "output_dir": "",
    "last_video_dir": "",
    "last_dxf_dir": "",
    "last_config_dir": "",

    # --- segmentation ----------------------------------------------------
    "threshold_mode": 0,            # 0 = auto (Otsu), 1 = manual
    "threshold": 30,
    "despeckle_px": 3,
    "fill_holes_px": 7,
    "min_area_pct": 0.0500,
    "gap_factor": 1.0,
    "envelope_factor": 1.10,
    "seg_mode_index": 0,            # 0 = auto, 1 = color, 2 = luma
    "color_frac": 0.30,
    "background_frames": 60,

    # --- fitting ---------------------------------------------------------
    "tau_px": 12.0,
    "tau_final_px": 2.5,
    "restarts": 64,
    "coverage_weight": 3.0,
    "use_features": True,
    "early_stop": True,
    "feature_weight": 1.0,
    "scale_prior_weight": 0.35,
    "max_scale_change": 0.60,

    # --- analysis --------------------------------------------------------
    "smooth_ms": 100.0,
    "min_confidence": 0.50,
    "max_gap_ms": 400.0,
    "px_per_mm": 0.0,               # 0 means "auto"

    # --- output ----------------------------------------------------------
    "decode_scale_index": 0,        # 0 = 1.0, 1 = 0.5, 2 = 0.25
    "write_overlay": True,
    "use_gpu": True,

    # --- viewer ----------------------------------------------------------
    "show_mask": True,
    "show_outline": True,

    # --- manual placement ------------------------------------------------
    # Stored so a lock survives a restart: re-finding the target by hand every
    # session would defeat the point of having placed it once.
    "manual_placement": False,
    "manual_pose": None,            # [tx, ty, theta, sx, sy] in full-resolution px

    # --- updates ---------------------------------------------------------
    # The public repository, so a fresh install updates itself with no setup.
    # Anyone on an offline rig overrides this with a folder path in Settings;
    # nothing in the app cares which kind of channel it is pointed at.
    "update_channel": "github:Fettucciny/Biohybrid-Robot-Tracker",
    "check_updates_on_start": True,

    # --- notifications ---------------------------------------------------
    "sound_enabled": True,

    # --- throughput history ----------------------------------------------
    # [[frames, pixels, seconds], ...] from finished runs, newest last. Used to
    # estimate how long the rest of a folder will take. Kept per machine rather
    # than shipped with a config: an RTX 4090 and an M1 do not share a rate, and
    # a borrowed .rtcfg should not import someone else's timings.
    "run_history": [],

    # --- window ----------------------------------------------------------
    "window_geometry": "",          # base64 QByteArray from saveGeometry()
}


# Keys that have been renamed. A ``.rtcfg`` saved by an older build is a record
# of a real experiment and has to keep loading, so the old spelling is accepted
# and mapped forward rather than silently dropped -- which is what would happen
# otherwise, since DEFAULTS doubles as the whitelist. Old name on the left.
RENAMED: dict[str, str] = {
    "color_frac": "color_frac",
}


# Saved values that were not merely different from the new default, but wrong.
#
# A stored setting normally beats a new default -- that is the entire point of
# remembering it. These are the exception: they were transcribed from a source
# that has since been corrected, so honouring the saved copy means quietly going
# on producing wrong numbers with no sign that anything is off. Each entry is
# (old value, corrected value, why), and it only fires when the stored value is
# still *exactly* the old default, i.e. nobody has deliberately typed anything.
#
# The change is announced in the log rather than made silently. A parameter that
# rescales every force in your results should never move without saying so.
CORRECTED: dict[str, tuple[float, float, str]] = {
    "beam_muscle_offset_mm": (
        1.642, 1.243,
        "SampleForce.m was corrected; the moment arm scales force by 1/l, so "
        "the old value read about 32% low"),
}


def apply_corrections(state: dict) -> list[str]:
    """Fix known-wrong stored values in place. Returns notes worth logging."""
    notes = []
    for key, (old, new, why) in CORRECTED.items():
        try:
            cur = float(state.get(key))
        except (TypeError, ValueError):
            continue
        if abs(cur - old) < 1e-9:
            state[key] = new
            notes.append(f"{key}: {old:g} → {new:g} mm — {why}")
    return notes


def settings_path() -> Path:
    return user_dir() / "settings.json"


def merge(loaded: dict | None) -> dict:
    """Defaults overlaid with whatever of ``loaded`` is recognized and sane."""
    out = dict(DEFAULTS)
    if not isinstance(loaded, dict):
        return out
    loaded = dict(loaded)
    for old, new in RENAMED.items():
        if old in loaded and new not in loaded:
            loaded[new] = loaded.pop(old)
    for k, default in DEFAULTS.items():
        if k not in loaded:
            continue
        v = loaded[k]
        if default is None:
            out[k] = v
            continue
        # Coerce rather than trust: a hand-edited config with "64" where an int
        # belongs should still load, and a genuinely wrong type should fall back
        # to the default rather than reach a Qt setter and raise.
        try:
            if isinstance(default, bool):
                out[k] = bool(v)
            elif isinstance(default, int):
                out[k] = int(v)
            elif isinstance(default, float):
                out[k] = float(v)
            elif isinstance(default, str):
                out[k] = str(v)
            else:
                out[k] = v
        except (TypeError, ValueError):
            out[k] = default
    return out


def load_settings() -> dict:
    try:
        raw = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULTS)
    return merge(raw.get("settings", raw))


def save_settings(state: dict) -> None:
    """Best effort. Failing to persist preferences must never break a session."""
    try:
        settings_path().write_text(_document(state), encoding="utf-8")
    except OSError:
        pass


def _document(state: dict) -> str:
    from . import __version__
    return json.dumps({
        "app": "robotrack",
        "app_version": __version__,
        "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {k: state.get(k, v) for k, v in DEFAULTS.items()},
    }, indent=2)


# ---------------------------------------------------------------------------
# Named config files
# ---------------------------------------------------------------------------

def write_config(path: str | Path, state: dict) -> Path:
    p = Path(path)
    if not p.suffix:
        p = p.with_suffix(CONFIG_SUFFIX)
    p.write_text(_document(state), encoding="utf-8")
    return p


def read_config(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "settings" in raw:
        return merge(raw["settings"])
    return merge(raw)


def config_origin(path: str | Path) -> str:
    """A one-line provenance note for the log after loading a config."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(raw, dict):
        return ""
    ver, when = raw.get("app_version", "?"), raw.get("saved", "")
    return f"written by robotrack {ver}" + (f" on {when}" if when else "")
