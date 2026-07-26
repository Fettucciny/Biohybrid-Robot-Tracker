"""robotrack -- GPU video tracking for muscle-driven soft robots."""

__version__ = "0.10.0"

__all__ = ["probe", "VideoInfo", "RunConfig", "run", "Result", "__version__"]

# The public names resolve on first use rather than at import time.
#
# Importing them eagerly means ``import robotrack`` pulls in torch, OpenCV and
# pandas, which makes two things impossible that matter here: reading
# ``__version__`` cheaply, and running the updater at all when the heavy stack is
# the thing that is broken. An update is exactly what fixes a bad overlay or a
# missing CUDA DLL, so it must not depend on them importing first.
#
# ``robotrack.probe`` and friends behave identically; they just pay their import
# cost on first attribute access. See PEP 562.

_LAZY = {
    "probe": ("robotrack.ingest", "probe"),
    "VideoInfo": ("robotrack.ingest", "VideoInfo"),
    "RunConfig": ("robotrack.pipeline", "RunConfig"),
    "run": ("robotrack.pipeline", "run"),
    "Result": ("robotrack.pipeline", "Result"),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value          # cache, so the lookup happens once
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
