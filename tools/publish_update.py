"""Publish a robotrack update to a channel folder.

    python tools/publish_update.py "C:\\Users\\Yikan\\Nextcloud2\\robotrack-updates"
    python tools/publish_update.py <channel> --notes "Fixes the DXF placement handle"
    python tools/publish_update.py <channel> --full dist\\robotrack-setup.exe

What it does
------------
Zips the ``robotrack`` package, hashes it, copies it into the channel folder and
rewrites ``robotrack-updates.json`` there. Anyone whose app points at that folder
sees the new version the next time they press Update.

The version comes from ``robotrack/__init__.py`` and is never passed on the
command line, so the number in the manifest, the number the running app reports
and the number baked into the installer cannot drift apart. Bump it there before
publishing; the script refuses to overwrite a version that is already published
unless you pass ``--force``, which is the guard against a lab member holding a
"0.2.0" that is not the 0.2.0 everyone else has.

A code patch carries only pure Python. If a release changes a dependency, the
Python version or the bundled ffmpeg, publish the installer with ``--full``
instead: an overlay cannot swap out a compiled wheel that is already loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "robotrack"
MANIFEST_NAME = "robotrack-updates.json"

# Only source ships in a code patch. Caches are per-interpreter and would be
# stale on arrival; theme kit JSON and any future data files are included.
# .wav carries the completion sounds; without it a code patch would ship
# the code that plays them and not the sounds themselves.
INCLUDE_SUFFIXES = {".py", ".json", ".txt", ".md", ".png", ".ico", ".svg", ".wav"}
EXCLUDE_DIRS = {"__pycache__", ".git", ".pytest_cache"}


def project_version() -> str:
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']',
                  (PKG / "__init__.py").read_text(encoding="utf-8"))
    if not m:
        sys.exit("Could not read __version__ from robotrack/__init__.py")
    return m.group(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_code_zip(dest: Path, version: str) -> Path:
    out = dest / f"robotrack-{version}-code.zip"
    files = [p for p in sorted(PKG.rglob("*"))
             if p.is_file()
             and p.suffix.lower() in INCLUDE_SUFFIXES
             and not (set(p.relative_to(PKG).parts) & EXCLUDE_DIRS)]
    if not files:
        sys.exit("No source files found to package.")
    # Deterministic: sorted order and a fixed timestamp, so republishing an
    # unchanged tree produces an identical hash instead of a spurious "update".
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            info = zipfile.ZipInfo(str(Path("robotrack") / p.relative_to(PKG)).replace("\\", "/"),
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, p.read_bytes())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("channel", help="folder the app is pointed at (local, UNC or synced), "
                                    "or the output folder when --code-zip-only is given")
    ap.add_argument("--notes", default="", help="release notes shown in the update dialog")
    ap.add_argument("--full", metavar="SETUP_EXE",
                    help="publish an installer instead of a code patch")
    ap.add_argument("--min-version", default="",
                    help="installs older than this must take the full installer")
    ap.add_argument("--force", action="store_true",
                    help="replace an already-published entry for this version")
    ap.add_argument("--code-zip-only", action="store_true",
                    help="just build robotrack-<version>-code.zip into the given "
                         "folder and print its path; write no manifest. This is "
                         "what CI uses: a GitHub release has no manifest, the "
                         "assets themselves are the channel.")
    a = ap.parse_args(argv)

    version = project_version()
    dest = Path(a.channel).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    if a.code_zip_only:
        out = build_code_zip(dest, version)
        print(out)
        return 0

    manifest_path = dest / MANIFEST_NAME

    manifest = {"releases": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            sys.exit(f"{manifest_path} exists but is not valid JSON. Move it aside.")
    releases = [r for r in manifest.get("releases", [])]

    if any(r.get("version") == version for r in releases) and not a.force:
        sys.exit(f"Version {version} is already published in {manifest_path}.\n"
                 f"Bump __version__ in robotrack/__init__.py, or pass --force.")

    if a.full:
        src = Path(a.full)
        if not src.exists():
            sys.exit(f"No such installer: {src}")
        # .exe on Windows, .dmg on macOS -- update.py picks the asset for the
        # platform it is running on by extension, so the name has to keep it.
        suffix = src.suffix.lower() or ".exe"
        stem = "setup" if suffix == ".exe" else "macos"
        payload = dest / f"robotrack-{version}-{stem}{suffix}"
        shutil.copy2(src, payload)
        kind = "full"
    else:
        payload = build_code_zip(dest, version)
        kind = "code"

    entry = {
        "version": version,
        "kind": kind,
        "file": payload.name,
        "sha256": sha256(payload),
        "size": payload.stat().st_size,
        "notes": a.notes,
        "min_version": a.min_version,
        "published": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    releases = [r for r in releases if r.get("version") != version] + [entry]
    releases.sort(key=lambda r: [int(x) if x.isdigit() else 0
                                 for x in str(r.get("version", "0")).split(".")],
                  reverse=True)
    manifest["releases"] = releases
    manifest["latest"] = releases[0]["version"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    kb = entry["size"] / 1024
    print(f"published robotrack {version} ({kind}, {kb:,.0f} KB)")
    print(f"  payload : {payload}")
    print(f"  manifest: {manifest_path}")
    print(f"\nPoint the app's update channel at:\n  {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
