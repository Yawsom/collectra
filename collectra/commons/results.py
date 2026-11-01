"""The header of a results file, and how it is found, read and written.

A results file begins with an entry that describes the file itself, declared
like any other entry by its ``type``::

    results:
      type: collectra.Results
      partition: validation
      timestamp: '2026-02-13T17:36:26.080812'

``Results`` can be subclassed to carry more, and a file's header is read back as
the class its ``type`` names. The older ``collectra_results_metadata:`` mapping
is still read, with a warning, and is replaced by the new entry the next time
Collectra saves the file.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pydantic

from ..utils import load_class_from_string

LEGACY_KEY = "collectra_results_metadata"
HEADER_KEY = "results"
HEADER_TYPE = "collectra.Results"
# Where the header goes in a file that already has a node called ``results``.
FALLBACK_KEY = "collectra_results"

_warned = False


class Results(pydantic.BaseModel):
    """What a results file records about itself.

    Subclass it to add fields; any other key in the entry is kept as well.
    """

    # Some results files have an unquoted version (e.g. `version: 1.0`), which
    # YAML reads as a float rather than text.
    model_config = pydantic.ConfigDict(coerce_numbers_to_str=True, extra="allow")

    workflow: str | None = None
    version: str | None = None
    partition: str | None = None
    timestamp: str | None = None

    def model_dump(self, *args, **kwargs) -> dict:
        """The attributes as written to a file: stamped now, without unset ones."""
        data = super().model_dump(*args, **kwargs)
        data["timestamp"] = datetime.now().isoformat()
        for name in ("partition", "workflow", "version"):
            if data[name] is None:
                data.pop(name)
        return data

    def entry(self) -> dict:
        """This header as the entry written at the top of a results file."""
        cls = type(self)
        path = HEADER_TYPE if cls is Results else f"{cls.__module__}.{cls.__qualname__}"
        return {"type": path, **self.model_dump()}


def is_header(value: Any) -> bool:
    """Whether a results entry is the file's header rather than an artefact."""
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        return False
    declared = value["type"]
    if declared == HEADER_TYPE:
        return True
    if declared.startswith("collectra."):
        return False  # one of the built-in artefacts: no need to load it
    try:
        cls = load_class_from_string(declared)
    except (ImportError, AttributeError, ValueError):
        return False
    return isinstance(cls, type) and issubclass(cls, Results)


def _find(data: dict) -> str | None:
    """The key of the header: the old mapping, else the first header entry."""
    if LEGACY_KEY in data:
        return LEGACY_KEY
    return next((key for key, value in data.items() if is_header(value)), None)


def _warn_legacy(source: str | Path | None) -> None:
    """Say once, not for every file, that the old header is deprecated."""
    global _warned
    if _warned:
        return
    _warned = True
    where = f" (first seen in {source})" if source else ""
    warnings.warn(
        f"Results files with a {LEGACY_KEY!r} header are deprecated{where}. The "
        f"header is now an entry such as '{HEADER_KEY}: {{type: {HEADER_TYPE}, "
        "partition: ...}'. Files are updated when Collectra next saves them. "
        "This warning is shown once.",
        FutureWarning,
        stacklevel=3,
    )


def pop_header(data: dict, source: str | Path | None = None) -> Results:
    """Remove the header from ``data`` and return it (empty if there is none)."""
    key = _find(data)
    if key is None:
        return Results()
    entry = data.pop(key)
    if key == LEGACY_KEY:
        _warn_legacy(source)
        return Results(**(entry or {}))
    entry = dict(entry)
    declared = entry.pop("type")
    cls = Results if declared == HEADER_TYPE else load_class_from_string(declared)
    return cls(**entry)


def peek_header(data: dict, source: str | Path | None = None) -> Results:
    """The header of ``data``, leaving ``data`` as it is."""
    return pop_header(dict(data), source)


def put_header(header: Results, data: dict) -> dict:
    """``data`` with ``header`` as its first entry, in the current form."""
    key = FALLBACK_KEY if HEADER_KEY in data else HEADER_KEY
    return {key: header.entry(), **data}


def upgrade_header(data: dict) -> dict | None:
    """``data`` with an old-style header turned into the current entry.

    The header's values, timestamp included, are kept as they are. Returns None
    when there is no old-style header, so there is nothing to change.
    """
    if LEGACY_KEY not in data:
        return None
    rest = {key: value for key, value in data.items() if key != LEGACY_KEY}
    key = FALLBACK_KEY if HEADER_KEY in rest else HEADER_KEY
    return {key: {"type": HEADER_TYPE, **(data[LEGACY_KEY] or {})}, **rest}
