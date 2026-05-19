from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from ml_collections import config_dict

from nt._client.imitation_mirror.config_sections import ConfigSection
from nt._client.imitation_mirror.utils.env_utils import import_module_from_path


def _apply_override(
    config: ConfigSection,
    dotted_key: str,
    raw_value: str,
) -> None:
    """Traverse ``config`` following ``dotted_key`` and assign ``raw_value`` at the leaf.

    Example:
        This helper is called by :func:`load_config` for every CLI override that
        follows ``key=value`` syntax. For example, running
        ``python train.py --config=... optimization.lr=5e-4`` results in
        ``_apply_override(config, "optimization.lr", "5e-4")`` which updates the
        nested :class:`ConfigDict` in place.
    """

    # Split the dotted path and drop an optional leading "config." prefix for CLI convenience.
    parts = dotted_key.split(".")
    if parts and parts[0] == "config":
        parts = parts[1:]
    if not parts:
        raise ValueError("Override keys must be non-empty.")

    # Descend one level at a time until we reach the parent ConfigDict containing the target key.
    cursor: Any = config
    for part in parts[:-1]:
        if isinstance(cursor, config_dict.ConfigDict):
            if part not in cursor:
                raise KeyError(
                    f"Override path '{dotted_key}' references unknown section '{part}'."
                )
            cursor = cursor[part]
        else:
            raise TypeError(
                f"Cannot descend into '{part}' on object of type {type(cursor).__name__} "
                f"while applying override '{dotted_key}'."
            )

    # Validate that the final hop points to a ConfigDict so we can assign into it safely.
    leaf = parts[-1]
    if not leaf:
        raise ValueError(
            f"Override '{dotted_key}' ends with a dot; supply a proper field name."
        )
    if not isinstance(cursor, config_dict.ConfigDict):
        raise TypeError(
            f"Override target '{leaf}' lives on object type {type(cursor).__name__}, "
            "which does not support mapping-style updates."
        )

    # Parse the CLI string literal into Python types (ints, floats, tuples, etc.) when possible.
    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        parsed = raw_value

    # Delegate to ConfigDict so type safety and nested conversions remain consistent everywhere.
    cursor[leaf] = parsed


def _call_config_hook(
    hook: Callable[..., Any],
    config: ConfigSection,
    identifier: str,
) -> ConfigSection:
    """Invoke a config hook that may either mutate ``config`` in-place or return a copy.

    Example:
        Experiment modules under ``experiments/`` expose a ``get_config`` (or
        similar) hook. :func:`load_config` routes those hooks here so experiment
        authors can stamp their custom settings onto the shared defaults. If a
        hook mutates in place it can return ``None``; otherwise it may return a
        brand new :class:`ConfigDict`.
    """

    # Attempt to pass the config into the hook; this is the common case.
    result = hook(config)

    # Hooks that accept the ConfigDict can still return None to signal "mutated in place".
    if result is None:
        return config
    if not isinstance(result, (ConfigSection, config_dict.ConfigDict)):
        raise TypeError(
            f"Config hook '{identifier}' must return a ConfigSection or ConfigDict when accepting arguments."
        )
    return result


def _apply_config_module(
    config: ConfigSection,
    module: ModuleType,
    identifier: str,
) -> ConfigSection:
    """Apply configuration customisations exported by ``module`` to ``config``.

    Example:
        When ``train_policy.py`` loads ``experiments/dit_block_toast_jan_27.py``
        we end up here. The helper looks for ``build_config``/``get_config``/
        ``configure`` and delegates to :func:`_call_config_hook`. This keeps the
        experiment surface flexible while centralising the wiring.
    """

    # Search through common hook names in priority order and invoke the first one we find.
    for attr in ("build_config", "get_config", "configure"):
        if hasattr(module, attr):
            hook = getattr(module, attr)
            return _call_config_hook(hook, config, f"{identifier}.{attr}")

    # Explicitly fail when the module does not expose any recognised hook names.
    raise AttributeError(
        f"Config module '{identifier}' must expose one of: build_config, get_config, configure."
    )


def _resolve_spec(spec: str) -> tuple[ModuleType, str]:
    """Resolve a dotted module or filesystem path into an importable module and identifier. Used
    when we specify a config file via ``--config=....py``.

    Example:
        The top-level :func:`load_config` passes the raw ``--config`` flag to
        this helper. If the argument looks like a path we feed it through
        :func:`_import_module_from_path`; otherwise we rely on
        :func:`importlib.import_module`.
    """

    # Prefer filesystem resolution when the provided string points to an existing file.
    path_candidate = Path(spec).expanduser()
    if path_candidate.exists():
        module = import_module_from_path(path_candidate)
        return module, str(path_candidate)

    # Fall back to a standard import using Python's module search path.
    module = importlib.import_module(spec)
    return module, spec


def load_base_cfg_overrides(
    *,
    base_config: ConfigSection,
    spec: str | None = None,
    overrides: list[str] | None = None,
) -> ConfigSection:
    """Apply module config and dotted overrides on top of an existing base config.

    Args:
        base_config: Pre-built base config object to mutate.
        spec: Optional module/path spec exposing config hook(s).
        overrides: Optional dotted ``key=value`` overrides.

    Returns:
        ConfigSection: Resolved config after module hook and overrides.
    """

    # Start from caller-provided base config.
    config = base_config

    # Apply module config hook only when a spec is provided.
    if spec is not None and spec != "":
        module, identifier = _resolve_spec(spec)
        config = _apply_config_module(config, module, identifier)

    # Apply optional dotted key overrides on top of resolved config.
    if overrides:
        config = cfg_overrides(config, overrides)

    # Return final resolved config object.
    return config


def cfg_overrides(
    config: ConfigSection,
    overrides: list[str] | None = None,
) -> ConfigSection:
    """Apply CLI-style overrides directly onto an existing config object.

    Example:
        This helper is similar to :func:`load_config_and_apply_overrides` but
        assumes the base config is already available. It is useful for cases
        where the config is built programmatically instead of via an experiment
        module.
    """

    # Iterate over CLI overrides (if any) and set each dotted key/value pair.
    if overrides:
        for override in overrides:
            key, value = override.split("=", maxsplit=1)
            _apply_override(config, key, value)

    # Return the fully materialised ConfigDict ready for downstream consumption.
    return config


__all__ = [
    "load_base_cfg_overrides",
    "cfg_overrides",
]
