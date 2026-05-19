import os
from pathlib import Path
from typing import Any


def get_data_root() -> Path:
    """
    Retrieve the dataset root directory from the NT_DATA_ROOT environment variable.

    Returns:
        Path: The path to the data root.

    Raises:
        RuntimeError: If NT_DATA_ROOT is not set.
    """
    root = os.environ.get("NT_DATA_ROOT")
    if root is None:
        raise RuntimeError(
            "NT_DATA_ROOT environment variable is not set. "
            "Please set it to the root directory containing your datasets."
        )
    return Path(root).expanduser().resolve()


def get_droid_root() -> Path:
    """Root of the preprocessed DROID dataset tree.

    Reads NT_DROID_ROOT, falling back to /mnt/droid for legacy non-Modal
    hosts where DROID is mounted there directly. Modal sets NT_DROID_ROOT
    to /mnt/datasets/droid (sibling of /mnt/datasets/droid).
    """
    root = os.environ.get("NT_DROID_ROOT", "/mnt/droid")
    return Path(root).expanduser().resolve()


def get_artifacts_root() -> Path:
    """
    Retrieve the artifacts root directory from the NT_ARTIFACTS_ROOT environment variable.

    Returns:
        Path: The path to the artifacts root.

    Raises:
        RuntimeError: If NT_ARTIFACTS_ROOT is not set.
    """
    root = os.environ.get("NT_ARTIFACTS_ROOT")
    if root is None:
        raise RuntimeError(
            "NT_ARTIFACTS_ROOT environment variable is not set. "
            "Please set it to the root directory where training artifacts should be written."
        )
    return Path(root).expanduser().resolve()


def import_module_from_path(path: str | Path) -> Any:
    """Dynamically import a module from a filesystem path.

    Args:
        path: Path to the python file to import.

    Returns:
        The imported module object.

    Raises:
        FileNotFoundError: If the path does not exist.
        ImportError: If the module cannot be loaded.
    """
    import importlib.util

    # Resolve the path eagerly so error messages mention the fully expanded location.
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Module path '{path}' not found or is not a file.")

    # Forge a synthetic module name derived from the path so Python can cache it correctly.
    module_name = f"external_module_{resolved_path.stem}_{abs(hash(resolved_path))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from '{resolved_path}'.")

    # Materialise and execute the module object, then return it to the caller.
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module
