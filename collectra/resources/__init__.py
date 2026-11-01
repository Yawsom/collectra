"""Named helpers that tasks can use, such as lookups an LLM can query."""

from importlib import import_module

_EXPORTS = {
    "Resource": "collectra.resources.base",
    "EntityLookup": "collectra.resources.entity_lookup",
    "LookupHit": "collectra.resources.entity_lookup",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
