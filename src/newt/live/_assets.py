"""The web viewer's three files, fetched once and cached on the rig.

Adapted from ``src/so100_hackathon/web_assets.py`` in
https://github.com/mission-robotics-ai/so100-hackathon (a fork of
rerun-io/so100-hackathon), Copyright Rerun Technologies AB, dual-licensed
``MIT OR Apache-2.0``. The version pin, the three file names, the extensionless
route and the reason it has to exist are all theirs; the cache location, the
refusals and the offline behaviour below are ours.

**Why a fetch and not a bundle.** ``@rerun-io/web-viewer`` is 46 MB of WebAssembly
and its JavaScript glue is built against one exact SDK version — a mismatched pair
does not load at all. Vendoring it would put a 46 MB binary in this repo that goes
stale the moment the SDK moves. Fetching it keyed on ``rerun.__version__`` instead
makes the pair impossible to separate: whatever SDK is installed is the viewer that
gets served.

**Why no bundler.** The viewer's ``index.js`` reaches its glue through
``await import("./re_viewer")`` — extensionless, which every bundler resolves and no
browser does. The hackathon's fix is to route the extensionless path to the ``.js``
file at the server instead, which is a routing rule rather than a build step. That
is what ``VIEWER_ROUTES`` below is.
"""
from __future__ import annotations

import io
import os
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

#: The three files out of the npm tarball that the browser actually asks for.
WEB_VIEWER_FILES = ("index.js", "re_viewer.js", "re_viewer_bg.wasm")

NPM_TARBALL = "https://registry.npmjs.org/@rerun-io/web-viewer/-/web-viewer-{version}.tgz"

#: Route -> (file, content type). Two routes for one file is the whole trick: the
#: browser's failing extensionless import is satisfied by a server rule.
VIEWER_ROUTES: dict[str, tuple[str, str]] = {
    "/viewer/index.js": ("index.js", "text/javascript"),
    "/viewer/re_viewer": ("re_viewer.js", "text/javascript"),
    "/viewer/re_viewer.js": ("re_viewer.js", "text/javascript"),
    "/viewer/re_viewer_bg.wasm": ("re_viewer_bg.wasm", "application/wasm"),
}


class ViewerAssetsUnavailable(RuntimeError):
    """The viewer's files are neither cached nor fetchable."""


def cache_root() -> Path:
    """Where fetched viewer builds live. Honours ``XDG_CACHE_HOME``; a per-version
    directory, so two SDK versions on one machine never overwrite each other."""
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "newt" / "web-viewer"


def ensure_assets(version: str, *, allow_fetch: bool = True) -> dict[str, Path]:
    """Return the three viewer files for ``version``, fetching them once if needed.

    Fetches nothing when the cache is already complete, so a rig with no internet
    works forever after the first run. When the cache is incomplete and the fetch
    cannot happen, this refuses and names the machine, the version and the copy
    command — it never serves a partial viewer, because a page that boots against
    half a wasm blob fails inside the browser where nobody is looking.
    """
    cache = cache_root() / version
    targets = {name: cache / name for name in WEB_VIEWER_FILES}
    if all(path.is_file() and path.stat().st_size > 0 for path in targets.values()):
        return targets

    if not allow_fetch:
        raise ViewerAssetsUnavailable(
            f"The web viewer for rerun {version} is not cached on this machine, and "
            f"fetching was switched off.\n"
            f"Yours: nothing is wrong with the rig or the session — the viewer's "
            f"46 MB payload has simply never been downloaded here.\n"
            f"Do now: run once with fetching allowed, or copy the directory "
            f"{cache} from a machine that has it."
        )

    url = NPM_TARBALL.format(version=version)
    cache.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ViewerAssetsUnavailable(
            f"Could not download the web viewer for rerun {version} from "
            f"{url} ({type(exc).__name__}: {exc}).\n"
            f"Yours: this machine could not reach the npm registry. The session and "
            f"the rig are fine; only the page that would display them is missing.\n"
            f"Do now: give this machine network access once and run again, or copy "
            f"{cache} here from a machine that already has it. The download happens "
            f"once per rerun version and never again."
        ) from exc

    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            for name, destination in targets.items():
                member = tar.extractfile(f"package/{name}")
                if member is None:
                    raise ViewerAssetsUnavailable(
                        f"The @rerun-io/web-viewer@{version} tarball has no "
                        f"package/{name}.\n"
                        f"Ours: this SDK version's viewer package is not laid out the "
                        f"way newt expects, so nothing was cached.\n"
                        f"Do now: report the rerun version above — newt pins the viewer "
                        f"to the installed SDK and this pairing needs looking at."
                    )
                destination.write_bytes(member.read())
    except tarfile.TarError as exc:
        raise ViewerAssetsUnavailable(
            f"The @rerun-io/web-viewer@{version} download did not open as a tarball "
            f"({type(exc).__name__}: {exc}).\n"
            f"Ours or the network's: the bytes arrived corrupt or truncated. Nothing "
            f"was cached, so nothing partial will be served.\n"
            f"Do now: delete {cache} and run again."
        ) from exc

    return targets
