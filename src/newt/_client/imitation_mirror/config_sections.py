from __future__ import annotations

# Capture typing utilities that power the ConfigSection schema bookkeeping.
from importlib import import_module
from typing import Any, ClassVar, Dict, TypeVar, get_origin
import dataclasses

# Surface ml_collections as the underlying storage layer so overrides behave predictably.
from ml_collections import config_dict


# Provide a readable sentinel object that highlights uninitialised required fields.
class _UnsetSentinel:
    """Human-friendly singleton standing in for required-but-unset fields."""

    def __repr__(self) -> str:
        return "<UNSET>"

    __str__ = __repr__


# Instantiate the sentinel once and cache a TypeVar for downstream annotations.
_UNSET = _UnsetSentinel()
T = TypeVar("T", bound="ConfigSection")


class ConfigSection(config_dict.ConfigDict):
    """Structured config building block that keeps field metadata from annotations."""

    # Record per-subclass schema/type metadata derived from class annotations.
    __config_schema__: ClassVar[Dict[str, Any]] = {}

    # Cache per-subclass default values so instances can start from a clean baseline.
    __config_defaults__: ClassVar[Dict[str, Any]] = {}

    # Track which fields remain unresolved (i.e. still equal to the sentinel).
    __required_fields__: ClassVar[set[str]] = set()

    # ------------------------------------------------------------------
    # Metaprogramming: harvest annotations + defaults once per subclass.

    # Upon subclass definition, collect annotations/defaults across the MRO so nested sections inherit cleanly.
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Initialise temporary containers that aggregate schema/default/required information.
        schema: Dict[str, Any] = {}
        defaults: Dict[str, Any] = {}
        required: set[str] = set()

        # Traverse bases from oldest to newest so child attributes override parents naturally.
        for base in reversed(cls.__mro__):
            if not issubclass(base, ConfigSection):
                continue

            # Fetch annotations from this base; we only care about non-ClassVar fields.
            annotations = getattr(base, "__annotations__", {})
            for name, annotation in annotations.items():
                origin = get_origin(annotation)
                if origin is ClassVar:
                    continue
                if (
                    origin is None
                    and isinstance(annotation, str)
                    and annotation.startswith("ClassVar")
                ):
                    continue

                # Update the schema entry and capture the default (falling back to the sentinel).
                schema[name] = annotation
                default = getattr(base, name, _UNSET)
                defaults[name] = default

                # Flag required fields by checking for the sentinel, otherwise clear the requirement.
                if default is _UNSET:
                    required.add(name)
                elif name in required:
                    required.discard(name)

        # Publish the aggregated metadata onto the subclass so instances can reuse it.
        cls.__config_schema__ = schema
        cls.__config_defaults__ = defaults
        cls.__required_fields__ = required

    # ------------------------------------------------------------------
    # Construction: instantiate a ConfigDict seeded with schema defaults.

    # During instantiation, populate the backing dict with either user overrides or cloned defaults.
    def __init__(
        self, initial_dictionary: dict[str, Any] | None = None, **overrides: Any
    ) -> None:
        # Support positional dict argument for compatibility with ml_collections internal recursion.
        if initial_dictionary:
            overrides = {**initial_dictionary, **overrides}

        # Build the payload dict that will be forwarded into ConfigDict.__init__.
        data: Dict[str, Any] = {}
        defaults = object.__getattribute__(self, "__config_defaults__")

        # Copy defaults for every field while respecting explicit keyword overrides.
        for name, default in defaults.items():
            if name in overrides:
                data[name] = overrides.pop(name)
            else:
                data[name] = default

        # Surface unknown keyword arguments early so typos do not silently create new fields.
        if overrides:
            unknown = ", ".join(sorted(overrides.keys()))
            raise AttributeError(
                f"{type(self).__name__} received unknown fields: {unknown}"
            )

        # Delegate to ConfigDict which handles nested dict conversions and type safety.
        super().__init__(data)

    # ------------------------------------------------------------------
    # Mutation helpers: align attribute syntax with dict-style storage.

    # Normalise __setitem__ so we can enforce schema membership and clear sentinel entries.
    def __setitem__(self, key: str, value: Any) -> None:
        # Reject unknown fields since ConfigSection is meant to be declarative.
        schema = object.__getattribute__(self, "__config_schema__")
        if key not in schema:
            raise AttributeError(
                f"{key} is not a valid field for {type(self).__name__}."
            )

        # Remove the sentinel before delegating so ml_collections does not compare against custom types.
        fields = object.__getattribute__(self, "_fields")
        if key in fields and fields[key] is _UNSET:
            fields.pop(key)

        # Let ConfigDict perform its usual type-checking and nested conversions.
        super().__setitem__(key, value)

    # Mirror attribute assignment into __setitem__ so call sites can use dot notation naturally.
    def __setattr__(self, key: str, value: Any) -> None:
        # Enforce immutability of the implementation target.
        if key == "_target_class_":
            raise AttributeError(
                f"Cannot modify immutable field '{key}' on {type(self).__name__}."
            )

        # Private attributes bypass the schema guard; public fields are forwarded to __setitem__.
        schema = object.__getattribute__(self, "__config_schema__")
        if key.startswith("_") or key not in schema:
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    # Redirect attribute reads through ConfigDict.get so we can raise when encountering unset fields.
    def __getattribute__(self, key: str) -> Any:
        # Schema entries are pulled from the backing dict; everything else uses standard lookup.
        schema = object.__getattribute__(self, "__config_schema__")
        if key in schema:
            value = config_dict.ConfigDict.get(self, key, _UNSET)
            if value is _UNSET:
                raise AttributeError(
                    f"{type(self).__name__}.{key} has not been set yet; assign it before reading."
                )
            return value
        return super().__getattribute__(key)

    # Keep __getattr__ consistent so accessing unset attributes still raises with a friendly message.
    def __getattr__(self, key: str) -> Any:
        # Schema fields intentionally raise, everything else falls back to the parent implementation.
        schema = object.__getattribute__(self, "__config_schema__")
        if key in schema:
            raise AttributeError(
                f"{type(self).__name__}.{key} has not been set yet; assign it before reading."
            )
        return super().__getattr__(key)

    # ------------------------------------------------------------------
    # Validation helpers

    # Produce a list of required fields that still hold the sentinel for downstream diagnostics.
    def unresolved_fields(self) -> list[str]:
        # Iterate over the cached required-field set and retain those still equal to _UNSET.
        required = object.__getattribute__(self, "__required_fields__")
        return [name for name in required if self.get(name, _UNSET) is _UNSET]

    # Raise when unresolved fields remain, recursing into nested ConfigSections for completeness.
    def assert_resolved(self) -> None:
        # Bubble up missing-field errors with a deterministic ordering for easier debugging.
        missing = self.unresolved_fields()
        if missing:
            raise ValueError(
                f"{type(self).__name__} is missing required fields: {', '.join(sorted(missing))}."
            )

        # Recursively walk nested sections so downstream consumers can rely on fully populated trees.
        schema = object.__getattribute__(self, "__config_schema__")
        for name in schema:
            value = self.get(name, _UNSET)
            if isinstance(value, ConfigSection):
                value.assert_resolved()

    # ------------------------------------------------------------------
    # Serialization helpers

    @staticmethod
    def _is_dataclass_instance(obj: Any) -> bool:
        return dataclasses.is_dataclass(obj) and not isinstance(obj, type)

    def serialize(self, *, include_meta: bool = True) -> Dict[str, Any]:
        """Recursively convert the config into a plain dict, with optional class metadata."""

        def _serialize(val: Any) -> Any:
            if isinstance(val, ConfigSection):
                return val.serialize(include_meta=include_meta)
            if self._is_dataclass_instance(val):
                # Dataclasses are serialized as dicts with a target pointer.
                # We iterate fields manually instead of using asdict() to prevent
                # premature conversion of nested dataclasses to dicts, losing their type info.
                data = {}
                for f in dataclasses.fields(val):
                    if f.name.startswith("_") or not f.init:
                        continue
                    data[f.name] = _serialize(getattr(val, f.name))

                if include_meta:
                    data["__type__"] = "dataclass"
                    # Capture the fully qualified name so we can reconstruct it.
                    data["__target__"] = (
                        f"{type(val).__module__}.{type(val).__qualname__}"
                    )
                return data
            if isinstance(val, config_dict.ConfigDict):
                return {
                    "__type__": "ConfigDict",
                    "__target__": f"{type(val).__module__}.{type(val).__name__}",
                    "__items__": {k: _serialize(v) for k, v in val.items()},
                }
            if isinstance(val, dict):
                return {k: _serialize(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return [_serialize(v) for v in val]
            if hasattr(val, "tolist"):  # Handle Tensors/Arrays
                return val.tolist()
            return val

        payload: Dict[str, Any] = {k: _serialize(v) for k, v in self.items()}
        if include_meta:
            payload["__type__"] = "ConfigSection"
            payload["__target__"] = f"{type(self).__module__}.{type(self).__name__}"
        return payload

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> "ConfigSection":
        """Reconstruct a ConfigSection (and nested sections) from ``serialize`` output."""

        def _deserialize(obj: Any) -> Any:
            if isinstance(obj, list):
                return [_deserialize(v) for v in obj]
            if isinstance(obj, dict):
                # Handle ConfigSections
                if obj.get("__type__") == "ConfigSection" and "__target__" in obj:
                    target = obj["__target__"]
                    module_path, class_name = target.rsplit(".", 1)
                    target_cls = getattr(import_module(module_path), class_name)
                    fields = {
                        k: _deserialize(v)
                        for k, v in obj.items()
                        if k not in {"__type__", "__target__"}
                    }
                    return target_cls(**fields)

                # Handle Dataclasses
                if obj.get("__type__") == "dataclass" and "__target__" in obj:
                    target = obj["__target__"]
                    module_path, class_name = target.rsplit(".", 1)
                    target_cls = getattr(import_module(module_path), class_name)
                    allowed_fields = {
                        f.name for f in dataclasses.fields(target_cls) if f.init
                    }
                    fields = {
                        k: _deserialize(v)
                        for k, v in obj.items()
                        if k not in {"__type__", "__target__"} and k in allowed_fields
                    }
                    return target_cls(**fields)

                # Handle ConfigDicts
                if obj.get("__type__") == "ConfigDict" and "__items__" in obj:
                    items = {k: _deserialize(v) for k, v in obj["__items__"].items()}
                    return config_dict.ConfigDict(items)

                return {k: _deserialize(v) for k, v in obj.items()}
            return obj

        root = _deserialize(data)
        if isinstance(root, ConfigSection):
            return root
        if isinstance(root, config_dict.ConfigDict):
            return root
        # Fallback: if the root was a dataclass, we might return that too.
        if dataclasses.is_dataclass(root):
            return root

        raise TypeError(
            f"Deserialized config is not a ConfigSection/ConfigDict (got {type(root)})."
        )


# Export the class so other modules can import directly from infra.config_sections.
__all__ = ["ConfigSection"]
