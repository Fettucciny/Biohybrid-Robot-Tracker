"""In-app updates.

Why this exists in two tiers
----------------------------
The frozen bundle is 3-4 GB, almost all of which is PyTorch's CUDA runtime, Qt
and OpenCV. Those change rarely. What actually changes between releases is a few
hundred kilobytes of robotrack's own Python. Making every bug fix a 1.5 GB
download would mean fixes stop being pushed, so updates come in two kinds:

**code** -- a zip of the ``robotrack`` package alone, typically ~200 KB. It is
extracted into a per-user overlay directory which ``launcher/app.py`` puts at the
front of ``sys.path``. The frozen interpreter and every heavy dependency stay
exactly where they are; only the pure-Python layer is swapped. Applying one takes
about a second.

**full** -- the Inno Setup installer, run silently. Needed only when a
dependency, the Python version, or the bundled ffmpeg changes. The manifest says
which kind a release is, so that decision is made when the release is published
rather than guessed here.

Channels
--------
A channel is a place to look for a manifest. The spec string decides which:

    C:\\Users\\me\\Nextcloud\\robotrack-updates   -> local or UNC folder
    \\\\lab-nas\\share\\robotrack-updates          -> UNC folder
    https://example.org/robotrack/updates.json  -> plain HTTPS manifest
    github:owner/repo                           -> GitHub Releases API

They are interchangeable: publish to a synced folder now, point the same code at
GitHub later by changing one string in Settings. Nothing else in the app knows
which one is in use.

Trust
-----
Payloads are checked against the SHA-256 recorded in the manifest, which catches
a truncated download or a corrupted sync. It does *not* make an untrusted channel
safe -- whoever can write the manifest can write the hash. Point this at a
location only you can write to.

Safety net
----------
An overlay that fails to import would otherwise brick a windowed application:
the process dies before it can draw a dialog explaining why. So a marker file is
written before an overlay is first trusted and cleared once the GUI is up. If the
marker is still there on the next launch, the overlay is quarantined and the app
falls back to the version baked into the bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.error import URLError
from urllib.request import Request, urlopen

MANIFEST_NAME = "robotrack-updates.json"
USER_AGENT = "robotrack-updater"
NETWORK_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def user_dir() -> Path:
    """Per-user application data, in each platform's conventional place.

    Never the install directory: Program Files and /Applications are not
    writable without elevation, and a lab member running a per-user install
    still needs updates to work. Everything mutable lives here.
    """
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    elif os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_DATA_HOME"):
        root = Path(os.environ["XDG_DATA_HOME"])
    else:
        root = Path.home() / ".local" / "share"
    d = root / "robotrack"
    d.mkdir(parents=True, exist_ok=True)
    return d


def overlay_root() -> Path:
    d = user_dir() / "overlay"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path() -> Path:
    return user_dir() / "update-state.json"


def _pending_marker() -> Path:
    return user_dir() / "overlay-unverified"


def read_state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_state(state: dict) -> None:
    _state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def parse_version(v: str) -> tuple:
    """Compare dotted versions numerically, tolerating a suffix.

    ``0.10.0`` must sort above ``0.9.0``; a plain string compare gets that
    backwards, which would make the app refuse a real update.
    """
    nums, suffix = [], ""
    for part in str(v).strip().lstrip("vV").split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                suffix += ch
                break
        nums.append(int(digits) if digits else 0)
    while len(nums) < 3:
        nums.append(0)
    # A release with no suffix outranks a pre-release of the same numbers.
    return (tuple(nums), 1 if not suffix else 0, suffix)


def current_version() -> str:
    from . import __version__
    return __version__


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_dir() -> Path:
    """Where the application lives, or the project root when run from source."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def app_bundle() -> Path | None:
    """The enclosing ``.app`` on macOS, if this is a frozen bundle.

    A frozen mac app runs from ``Robotrack.app/Contents/MacOS/robotrack``, so
    ``sys.executable`` is three levels inside the thing the user thinks of as
    the application. Relaunching and replacing both need the bundle itself.
    """
    if sys.platform != "darwin" or not is_frozen():
        return None
    p = Path(sys.executable).resolve()
    for parent in p.parents:
        if parent.suffix == ".app":
            return parent
    return None


# ---------------------------------------------------------------------------
# Manifest / channel
# ---------------------------------------------------------------------------

@dataclass
class Release:
    version: str
    kind: str                    # "code" or "full"
    url: str                     # absolute URL, or a filesystem path
    sha256: str = ""
    size: int = 0
    notes: str = ""
    min_version: str = ""        # below this, a code patch is refused; needs a full install
    published: str = ""

    @property
    def is_code(self) -> bool:
        return self.kind == "code"

    def to_dict(self) -> dict:
        return asdict(self)


class UpdateError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _ssl_context():
    """An SSL context that can actually verify a certificate in a frozen app.

    Python does not carry a CA bundle of its own. On a normal install it borrows
    the operating system's, which works everywhere -- and stops working inside a
    PyInstaller bundle on macOS, where there is no Python installation for the
    usual ``Install Certificates.command`` to have run against and the OpenSSL
    that ships in the bundle has no store to point at. Every HTTPS request then
    fails with CERTIFICATE_VERIFY_FAILED, which is why "Check for updates" on a
    Mac returned an error instead of a version.

    certifi carries Mozilla's bundle as a data file, so it travels inside the
    app. If it is unavailable the default context is used, which is correct on
    Windows and on a source install.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl.create_default_context()
        except Exception:
            return None


def _read_bytes(location: str, timeout: int = NETWORK_TIMEOUT) -> bytes:
    """Fetch a manifest or payload from a URL or a filesystem path."""
    if _is_url(location):
        req = Request(location, headers={"User-Agent": USER_AGENT,
                                         "Accept": "application/json, */*"})
        ctx = _ssl_context()
        kwargs = {"timeout": timeout}
        if ctx is not None and str(location).lower().startswith("https"):
            kwargs["context"] = ctx
        try:
            with urlopen(req, **kwargs) as r:        # noqa: S310 - scheme checked below
                return r.read()
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            if "CERTIFICATE_VERIFY_FAILED" in str(reason):
                raise UpdateError(
                    "Could not verify the server's certificate.\n\n"
                    "This build is missing its certificate bundle. Reinstall "
                    "from the latest installer, or point the update channel at "
                    "a folder instead of an https address.") from exc
            raise
    return Path(location).read_bytes()


def _is_url(s: str) -> bool:
    return urlparse(str(s)).scheme in ("http", "https")


def normalize_channel(spec: str) -> str:
    return str(spec or "").strip().strip('"')


def describe_channel(spec: str) -> str:
    spec = normalize_channel(spec)
    if not spec:
        return "no update channel configured"
    if spec.lower().startswith("github:"):
        return f"GitHub Releases — {spec[7:]}"
    if _is_url(spec):
        return f"web — {spec}"
    return f"folder — {spec}"


def _manifest_location(spec: str) -> str:
    """Where the manifest itself lives, given a channel spec.

    A folder spec may point at the folder or straight at the json file; both are
    things a person will reasonably type, so accept either.
    """
    if _is_url(spec):
        return spec if spec.lower().endswith(".json") else spec.rstrip("/") + "/" + MANIFEST_NAME
    p = Path(spec)
    return str(p if p.suffix.lower() == ".json" else p / MANIFEST_NAME)


def _resolve_payload(spec: str, manifest_loc: str, ref: str) -> str:
    """Turn a manifest's file reference into something fetchable.

    References are relative to the manifest so a channel folder can be moved or
    re-shared without rewriting every entry.
    """
    if _is_url(ref):
        return ref
    if _is_url(manifest_loc):
        return urljoin(manifest_loc, ref)
    p = Path(ref)
    return str(p if p.is_absolute() else Path(manifest_loc).parent / p)


def _releases_from_manifest(spec: str, manifest_loc: str, data: dict) -> list[Release]:
    out = []
    for r in data.get("releases", []):
        ref = r.get("file") or r.get("url") or ""
        if not ref:
            continue
        out.append(Release(
            version=str(r.get("version", "0.0.0")),
            kind=str(r.get("kind", "code")).lower(),
            url=_resolve_payload(spec, manifest_loc, ref),
            sha256=str(r.get("sha256", "")).lower(),
            size=int(r.get("size", 0) or 0),
            notes=str(r.get("notes", "")),
            min_version=str(r.get("min_version", "")),
            published=str(r.get("published", "")),
        ))
    return out


def _releases_from_github(spec: str) -> list[Release]:
    """Read GitHub Releases, treating assets by filename convention.

    ``*-code.zip`` is a code patch; ``*setup*.exe`` is a full installer. A
    release carrying both is offered as a code patch, since that is the cheap
    path and the installer is only the fallback.
    """
    repo = spec.split(":", 1)[1].strip().strip("/")
    api = f"https://api.github.com/repos/{repo}/releases"
    data = json.loads(_read_bytes(api).decode("utf-8"))
    out = []
    for rel in data:
        if rel.get("draft"):
            continue
        assets = rel.get("assets", [])
        code = next((a for a in assets if a["name"].lower().endswith("-code.zip")), None)
        # The full installer is platform-specific; the code patch is not, being
        # pure Python. A release built by CI carries all three, so the choice of
        # full asset has to be made here rather than by the publisher.
        if sys.platform == "darwin":
            full = next((a for a in assets if a["name"].lower().endswith(".dmg")), None)
        else:
            full = next((a for a in assets if a["name"].lower().endswith(".exe")
                         and "setup" in a["name"].lower()), None)
        # Prefer the code patch: a few hundred kB against a gigabyte, and it
        # needs no reinstall. The installer is the fallback for a release that
        # changed a dependency or the bundled ffmpeg.
        chosen, kind = (code, "code") if code else (full, "full")
        if not chosen:
            continue
        out.append(Release(
            version=str(rel.get("tag_name") or rel.get("name") or "0.0.0"),
            kind=kind,
            url=chosen["browser_download_url"],
            size=int(chosen.get("size", 0) or 0),
            notes=str(rel.get("body") or ""),
            published=str(rel.get("published_at") or ""),
        ))
    return out


def fetch_releases(spec: str) -> list[Release]:
    """All releases the channel offers, newest first."""
    spec = normalize_channel(spec)
    if not spec:
        raise UpdateError("No update channel is configured. Set one in Settings.")
    try:
        if spec.lower().startswith("github:"):
            rels = _releases_from_github(spec)
        else:
            loc = _manifest_location(spec)
            data = json.loads(_read_bytes(loc).decode("utf-8"))
            rels = _releases_from_manifest(spec, loc, data)
    except FileNotFoundError:
        raise UpdateError(
            f"No update manifest found.\n\nLooked for:\n  {_manifest_location(spec)}\n\n"
            "Publish one with tools/publish_update.py, or check the channel path.")
    except (OSError, ValueError) as exc:
        raise UpdateError(f"Could not read the update channel:\n\n{exc}") from exc
    rels.sort(key=lambda r: parse_version(r.version), reverse=True)
    return rels


def check(spec: str, current: str | None = None) -> Release | None:
    """The newest release strictly newer than what is running, or None."""
    cur = parse_version(current or current_version())
    for r in fetch_releases(spec):
        if parse_version(r.version) > cur:
            return r
    return None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download(rel: Release, progress=None) -> Path:
    """Fetch a release payload to a temp file, verifying its hash.

    ``progress(done_bytes, total_bytes)`` is called as it streams; total is 0
    when the source does not report a length.
    """
    suffix = (".zip" if rel.is_code else
              (".dmg" if rel.url.lower().endswith(".dmg") else ".exe"))
    fd, tmp = tempfile.mkstemp(prefix="robotrack-update-", suffix=suffix)
    os.close(fd)
    tmp = Path(tmp)
    h = hashlib.sha256()
    try:
        if _is_url(rel.url):
            req = Request(rel.url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=NETWORK_TIMEOUT) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or rel.size or 0)
                done = 0
                while True:
                    chunk = r.read(262144)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
        else:
            src = Path(rel.url)
            total = src.stat().st_size
            done = 0
            with open(src, "rb") as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(262144)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise UpdateError(f"Download failed:\n\n{exc}") from exc

    if rel.sha256 and h.hexdigest() != rel.sha256:
        tmp.unlink(missing_ok=True)
        raise UpdateError(
            "The downloaded file does not match the checksum in the manifest.\n\n"
            f"expected {rel.sha256}\ngot      {h.hexdigest()}\n\n"
            "The file is corrupt or was replaced. Nothing has been installed.")
    return tmp


# ---------------------------------------------------------------------------
# Applying a code overlay
# ---------------------------------------------------------------------------

def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract, refusing entries that would escape the destination.

    A zip can contain ``../`` paths and absolute paths. Since this writes into a
    user directory on the strength of a manifest, the check is cheap insurance.
    """
    dest = dest.resolve()
    for member in zf.infolist():
        name = member.filename.replace("\\", "/")
        if name.startswith("/") or ".." in Path(name).parts:
            raise UpdateError(f"Update package contains an unsafe path: {member.filename}")
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest)):
            raise UpdateError(f"Update package contains an unsafe path: {member.filename}")
    zf.extractall(dest)


def apply_code_update(zip_path: Path, rel: Release) -> Path:
    """Unpack a code patch into a new overlay directory and make it active.

    Extraction goes to a staging directory first and is renamed into place only
    once it validates, so an interrupted update cannot leave a half-written
    package where the next launch would import it.
    """
    root = overlay_root()
    final = root / rel.version
    staging = root / f".staging-{rel.version}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            _safe_extract(zf, staging)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise UpdateError(f"The update package is not a readable zip:\n\n{exc}") from exc

    pkg = staging / "robotrack" / "__init__.py"
    if not pkg.exists():
        shutil.rmtree(staging, ignore_errors=True)
        raise UpdateError(
            "The update package does not contain a robotrack package at its root. "
            "It was probably built incorrectly.")

    shutil.rmtree(final, ignore_errors=True)
    staging.rename(final)

    state = read_state()
    state.update(active_overlay=rel.version,
                 channel_version=rel.version,
                 previous_overlay=state.get("active_overlay", ""),
                 applied_from=rel.url,
                 kind="code")
    state.pop("quarantined", None)
    write_state(state)
    # Trusted only once a launch proves it imports; see verify_overlay_startup.
    # ``attempts`` is what distinguishes "installed, not started yet" from
    # "started once and died before the window appeared" -- without it the very
    # first launch would look like a crash and quarantine a perfectly good build.
    write_marker(rel.version, 0)
    return final


def read_marker() -> tuple[str, int] | None:
    try:
        data = json.loads(_pending_marker().read_text(encoding="utf-8"))
        return str(data.get("version", "?")), int(data.get("attempts", 0))
    except (OSError, ValueError, TypeError):
        return None


def write_marker(version: str, attempts: int) -> None:
    try:
        _pending_marker().write_text(
            json.dumps({"version": version, "attempts": attempts}), encoding="utf-8")
    except OSError:
        pass


def active_overlay_path() -> Path | None:
    """The overlay directory to put on sys.path, if one is active and intact."""
    state = read_state()
    v = state.get("active_overlay")
    if not v:
        return None
    p = overlay_root() / v
    return p if (p / "robotrack" / "__init__.py").exists() else None


def disable_overlay(reason: str = "") -> None:
    """Fall back to the version inside the bundle."""
    state = read_state()
    bad = state.get("active_overlay", "")
    state["active_overlay"] = ""
    if bad:
        state["quarantined"] = {"version": bad, "reason": reason}
    write_state(state)
    _pending_marker().unlink(missing_ok=True)


def verify_overlay_startup() -> str | None:
    """Called at the very start of launch, before the overlay is used.

    Returns a message to show the user if the previous launch died while an
    unverified overlay was active. A windowed executable that dies during import
    shows nothing at all, so without this a bad patch is indistinguishable from
    the application simply refusing to open.
    """
    m = read_marker()
    if m is None:
        return None
    version, attempts = m
    state = read_state()
    if state.get("verified_version") == version:
        _pending_marker().unlink(missing_ok=True)
        return None
    if attempts < 1:
        # First launch since the update was applied. Record the attempt and let
        # it run; if it never reaches mark_overlay_verified, the next launch
        # lands in the branch below.
        write_marker(version, attempts + 1)
        return None
    disable_overlay(f"did not finish starting after updating to {version}")
    return (f"Update {version} did not start correctly and has been disabled.\n"
            "robotrack has reverted to the version it shipped with. "
            "The failed update is kept for inspection but will not be loaded again.")


def mark_overlay_verified() -> None:
    """Called once the GUI is actually up: the active overlay imports fine."""
    m = read_marker()
    if m is None:
        return
    state = read_state()
    state["verified_version"] = m[0]
    write_state(state)
    _pending_marker().unlink(missing_ok=True)


def rollback() -> str | None:
    """Drop back to the previous overlay, or to the bundled version."""
    state = read_state()
    prev = state.get("previous_overlay") or ""
    if prev and (overlay_root() / prev / "robotrack" / "__init__.py").exists():
        state["active_overlay"] = prev
        state["previous_overlay"] = ""
        write_state(state)
        _pending_marker().unlink(missing_ok=True)
        return prev
    disable_overlay("manual rollback")
    return None


# ---------------------------------------------------------------------------
# Applying a full installer, and relaunching
# ---------------------------------------------------------------------------

def apply_full_update(path: Path) -> None:
    """Install a full build: the Inno installer on Windows, a DMG on macOS.

    Neither can replace files this process holds open, so both are started
    detached and this process exits immediately.
    """
    if os.name == "nt":
        flags = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                 "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS"]
        creation = getattr(subprocess, "DETACHED_PROCESS", 0) | \
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen([str(path), *flags], close_fds=True, creationflags=creation)
        return
    if sys.platform == "darwin":
        # A DMG cannot install itself. Opening it puts the new app in front of
        # the user to drag across, which is the platform's expected gesture and
        # avoids this process trying to overwrite its own bundle while running.
        subprocess.Popen(["open", str(path)], close_fds=True)
        return
    raise UpdateError(f"Full updates are not supported on {sys.platform}.")


def relaunch_command() -> list[str]:
    bundle = app_bundle()
    if bundle is not None:
        # ``open -n`` asks Launch Services for a fresh instance, which is what
        # restarts a mac app correctly; running the inner binary directly loses
        # the bundle identity and with it the icon and the menu bar.
        return ["open", "-n", str(bundle)]
    if is_frozen():
        return [sys.executable]
    return [sys.executable, "-m", "robotrack.gui"]


def relaunch() -> None:
    """Start a fresh copy and let this one exit.

    Deliberately not os.execv: on Windows that keeps the original process handle
    and confuses Qt's cleanup. A detached child plus a clean exit is predictable.
    """
    creation = getattr(subprocess, "DETACHED_PROCESS", 0) | \
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(relaunch_command(), close_fds=True,
                     cwd=str(install_dir()), creationflags=creation)


# ---------------------------------------------------------------------------
# Convenience for the GUI
# ---------------------------------------------------------------------------

def can_apply(rel: Release) -> tuple[bool, str]:
    """Whether this release can be installed by this build, and why not."""
    if rel.min_version and parse_version(current_version()) < parse_version(rel.min_version):
        return False, (f"Version {rel.version} needs a full reinstall from {rel.min_version} "
                       f"or newer. Download and run the installer manually.")
    if not rel.is_code and not is_frozen():
        return False, ("This is a full installer update, which only applies to the packaged "
                       "application. Update your source checkout instead.")
    if not rel.is_code and not (os.name == "nt" or sys.platform == "darwin"):
        return False, f"Full updates are not supported on {sys.platform}."
    return True, ""


def status_line(spec: str) -> str:
    return f"robotrack {current_version()} · {describe_channel(spec)}"
