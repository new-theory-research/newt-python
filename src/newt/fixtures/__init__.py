"""newt.fixtures — real recorded observations for trying inference without hardware.

Each fixture is one observation frame extracted from a recorded episode on an NT
Trossen rig: the 8D wire state (pos + quaternion + gripper), three camera frames,
and the prompt the episode was collected with. Because the obs carries its own
prompt, `robot.infer(fixtures.load(name))` works with no extra arguments — the
model receives the exact language instruction it was trained against and returns a
non-degenerate action chunk.

Usage:
    import newt
    from newt import fixtures

    robot = newt.Robot(api_key=..., read_state=lambda: {}, execute=lambda c: None)
    response = robot.infer(fixtures.load("cup_stacking"))
    print(response)  # labeled chunk: x, y, z, qw, qx, qy, qz, gripper

Available fixtures:
    cup_stacking       — from a tr3 cup-stacking episode (left-arm rig, wrist camera
                         remapped to right-wrist-camera).
    pour_coffee_beans  — from a tr2 pour episode (native right-arm rig).
"""
from __future__ import annotations

import io
from importlib import resources

import numpy as np

# Maps fixture name -> bundled data file. Camera frames are stored JPEG-compressed
# to keep each fixture under the package-size budget; load() decodes them back to
# the (3, 240, 320) uint8 CHW arrays the wire protocol expects.
_FIXTURES = {
    "cup_stacking": "cup_stacking.npz",
    "pour_coffee_beans": "pour_coffee_beans.npz",
}

# Order matters: the keys are the camera names the SDK contract expects.
_CAMERA_KEYS = ("right-wrist-camera", "surrounding1", "surrounding2")
_JPEG_KEYS = ("jpeg_right_wrist", "jpeg_surrounding1", "jpeg_surrounding2")

# Override the prompt carried in the recorded .npz. cup_stacking's raw recorded
# tag ("blue cups stitch cup") is too rough for docs; substitute NT0-FP3's clean
# canonical task prompt, which still matches the obs and returns a non-degenerate
# chunk. Fixtures absent from this map keep their recorded prompt.
_PROMPT_OVERRIDES = {
    "cup_stacking": "Stack one cup into another cup.",
}


def _decode_jpeg(buf: np.ndarray) -> np.ndarray:
    """Decode JPEG bytes (stored as a uint8 array) to (3, 240, 320) uint8 CHW."""
    try:
        from PIL import Image
    except ImportError as exc:  # fail loud — never silently skip a camera
        raise ImportError(
            "newt.fixtures needs Pillow to decode the bundled camera frames. "
            "Install it with: pip install pillow"
        ) from exc

    img = Image.open(io.BytesIO(buf.tobytes()))
    arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # (3, H, W)


def load(name: str) -> dict:
    """Load a real recorded observation by name.

    Args:
        name: One of the available fixtures (see `available()`): "cup_stacking",
              "pour_coffee_beans".

    Returns:
        Observation dict ready for `robot.infer(...)`:
            {
                "state":  (8,) float32  — [x, y, z, qw, qx, qy, qz, gripper],
                "images": {camera_name: (3, 240, 320) uint8} for the three cameras,
                "prompt": str           — the instruction the episode was collected with,
            }
        The "prompt" rides along in the obs, so the caller doesn't pass it
        separately — `_build_obs_frame` prefers an obs-carried prompt.

    Raises:
        KeyError:   `name` is not a known fixture.
        ImportError: Pillow is not installed (needed to decode camera frames).
    """
    if name not in _FIXTURES:
        raise KeyError(
            f"Unknown fixture {name!r}. Available: {', '.join(available())}."
        )

    data_pkg = resources.files(__package__) / "data" / _FIXTURES[name]
    with data_pkg.open("rb") as f:
        npz = np.load(f, allow_pickle=False)
        state = np.ascontiguousarray(npz["state"], dtype=np.float32)
        images = {
            cam: _decode_jpeg(npz[jpeg_key])
            for cam, jpeg_key in zip(_CAMERA_KEYS, _JPEG_KEYS)
        }
        prompt = _PROMPT_OVERRIDES.get(name, str(npz["prompt"]))

    return {"state": state, "images": images, "prompt": prompt}


def available() -> list[str]:
    """Return the names of the bundled fixtures."""
    return list(_FIXTURES)
