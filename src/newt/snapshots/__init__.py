"""newt.snapshots — real recorded observations for trying inference without hardware.

Each snapshot is one observation frame extracted from a recorded episode on a real
NT rig: the proprioceptive state vector, the episode's camera frames, and the prompt
the episode was collected with. Because the obs carries its own prompt,
`robot.infer(snapshots.load(name))` works with no extra arguments — the model receives
the exact language instruction it was trained against and returns a non-degenerate
action chunk.

Snapshots differ in SHAPE by the rig they came from, and that shape is the whole point:
the state dimension and camera set must match the model's declared contract
(`robot.contract`). `newt run` reads the contract and picks the matching snapshot for
you; a mismatch is an honest error, never a silent coercion.

Usage:
    import newt
    from newt import snapshots

    robot = newt.Robot(api_key=..., model="nt0-fp3-pour")
    response = robot.infer(snapshots.load("cup_stacking"))
    print(response)  # labeled chunk

Available snapshots:
    cup_stacking       — 8-axis nt0 rig (left-arm tr3 cup-stacking episode; wrist camera
                         remapped to right-wrist-camera). Cameras: right-wrist-camera,
                         surrounding1, surrounding2.
    pour_coffee_beans  — 8-axis nt0 rig (native right-arm tr2 pour episode). Same cameras.
    red_cube           — 6-axis SO-101 rig (red-cube-bowl episode). Cameras: top, side at
                         378x378. Matches the so101 serve contract.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from importlib import resources

import numpy as np


@dataclass(frozen=True)
class _Snapshot:
    """A bundled snapshot's definition: its data file, its camera mapping, and an
    optional prompt override.

    ``cameras`` maps the contract camera name (the key the wire protocol expects) to
    the ``.npz`` key holding that camera's JPEG bytes — an ORDERED tuple because the
    contract declares cameras in order. Each snapshot carries its OWN camera set: an
    8-axis nt0 rig has three cameras, a 6-axis SO-101 rig has two (top/side). There is
    no global camera list — the shape is per-rig, and conflating them is exactly the
    bug this snapshot module exists to avoid.
    """

    file: str
    cameras: tuple[tuple[str, str], ...]  # ((contract_camera_name, npz_jpeg_key), ...)
    prompt_override: str | None = None


# Maps snapshot name -> its definition. Camera frames are stored JPEG-compressed to keep
# each snapshot under the package-size budget; load() decodes them back to (3, H, W)
# uint8 CHW arrays (H/W per the rig's contract — 240x320 for nt0, 378x378 for so101).
#
# cup_stacking is listed first: it is the historical default `newt run` reaches for when
# a model's contract declares nothing to match on (an nt0 base with no state_shape/cameras).
_SNAPSHOTS: dict[str, _Snapshot] = {
    "cup_stacking": _Snapshot(
        file="cup_stacking.npz",
        cameras=(
            ("right-wrist-camera", "jpeg_right_wrist"),
            ("surrounding1", "jpeg_surrounding1"),
            ("surrounding2", "jpeg_surrounding2"),
        ),
        # Override the raw recorded tag ("blue cups stitch cup") — too rough for docs.
        # Substitute NT0-FP3's clean canonical task prompt, which still matches the obs
        # and returns a non-degenerate chunk.
        prompt_override="Stack one cup into another cup.",
    ),
    "pour_coffee_beans": _Snapshot(
        file="pour_coffee_beans.npz",
        cameras=(
            ("right-wrist-camera", "jpeg_right_wrist"),
            ("surrounding1", "jpeg_surrounding1"),
            ("surrounding2", "jpeg_surrounding2"),
        ),
    ),
    # red_cube — 6-axis SO-101 red-cube-bowl frame — is registered in the SAME commit as
    # its data file (data/red_cube.npz), extracted from a real recorded episode. It is
    # withheld from the registry until that real frame is built (Rule 10: no entry that
    # would `load()` a file that isn't there, and no synthesized stand-in frame). Its
    # definition, kept here so the shape is reviewable, is:
    #     "red_cube": _Snapshot(
    #         file="red_cube.npz",
    #         cameras=(("top", "jpeg_top"), ("side", "jpeg_side")),
    #         # no override — the episode's real recorded task prompt rides out as-is
    #     ),
}


def _decode_jpeg(buf: np.ndarray) -> np.ndarray:
    """Decode JPEG bytes (stored as a uint8 array) to (3, H, W) uint8 CHW.

    The H/W come from the encoded frame itself — never assumed. An nt0 frame decodes to
    (3, 240, 320); an so101 frame to (3, 378, 378).
    """
    try:
        from PIL import Image
    except ImportError as exc:  # fail loud — never silently skip a camera
        raise ImportError(
            "newt.snapshots needs Pillow to decode the bundled camera frames. "
            "Install it with: pip install pillow"
        ) from exc

    img = Image.open(io.BytesIO(buf.tobytes()))
    arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # (3, H, W)


def _data_file(snap: _Snapshot):
    """Traversable handle to a snapshot's bundled .npz."""
    return resources.files(__package__) / "data" / snap.file


def load(name: str) -> dict:
    """Load a real recorded observation by name.

    Args:
        name: One of the available snapshots (see `available()`).

    Returns:
        Observation dict ready for `robot.infer(...)`:
            {
                "state":  (D,) float32   — the rig's proprioceptive state vector,
                "images": {camera_name: (3, H, W) uint8} for the rig's cameras,
                "prompt": str            — the instruction the episode was collected with,
            }
        D and (H, W) are per-rig (8 / 240x320 for nt0, 6 / 378x378 for so101). The
        "prompt" rides along in the obs, so the caller doesn't pass it separately —
        `_build_obs_frame` prefers an obs-carried prompt.

    Raises:
        KeyError:    `name` is not a known snapshot.
        ImportError: Pillow is not installed (needed to decode camera frames).
    """
    snap = _get(name)

    with _data_file(snap).open("rb") as f:
        npz = np.load(f, allow_pickle=False)
        state = np.ascontiguousarray(npz["state"], dtype=np.float32)
        images = {
            cam: _decode_jpeg(npz[jpeg_key]) for cam, jpeg_key in snap.cameras
        }
        prompt = snap.prompt_override or str(npz["prompt"])

    return {"state": state, "images": images, "prompt": prompt}


def describe(name: str) -> dict:
    """Return a snapshot's shape signature WITHOUT decoding its camera frames.

    This is what contract-aware selection matches against: a snapshot's state shape and
    its ordered camera names, read straight from the bundled data + definition (never
    hard-coded alongside — the data is the source of truth, so the two can't drift).

    Returns:
        {
            "state_shape": tuple[int, ...],   — e.g. (8,) for nt0, (6,) for so101,
            "cameras":     tuple[str, ...],   — contract camera names in order,
        }

    Raises:
        KeyError: `name` is not a known snapshot.
    """
    snap = _get(name)

    # np.load is lazy per-key; reading only `state` avoids touching the JPEG blobs.
    with _data_file(snap).open("rb") as f:
        npz = np.load(f, allow_pickle=False)
        state_shape = tuple(int(d) for d in npz["state"].shape)

    return {
        "state_shape": state_shape,
        "cameras": tuple(cam for cam, _ in snap.cameras),
    }


def available() -> list[str]:
    """Return the names of the bundled snapshots."""
    return list(_SNAPSHOTS)


def _get(name: str) -> _Snapshot:
    """Resolve a snapshot definition or raise a helpful KeyError."""
    try:
        return _SNAPSHOTS[name]
    except KeyError:
        raise KeyError(
            f"Unknown snapshot {name!r}. Available: {', '.join(available())}."
        ) from None
