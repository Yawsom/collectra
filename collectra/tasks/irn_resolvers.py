"""IRN (Internal Reference Number) resolver tasks for Collectra workflows.

This module provides an IRN resolver that uses semantic similarity search
(via an on-disk vector store + OpenAI embeddings) plus an LLM verification step to map
free-text query inputs to canonical IRN identifiers loaded from an xlsx
source file.

Public API:
    IRNResolver: Task class that resolves a query to an IRN string.
    normalise_column_name: Helper that normalises raw spreadsheet column names.
    render_card: Helper that renders a per-row card from a template.
"""

__all__ = ["IRNResolver", "normalise_column_name", "render_card"]

import hashlib
import math
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from collectra.tasks.canonicalisers import LLMCanonicaliser
from collectra.tasks.vector_store import VectorStore
from collectra.types.texts import Text

from ..logger import get_logger

load_dotenv()

logger = get_logger(__name__)


NO_MATCH_INSTRUCTION = (
    "If none of the candidates is clearly the same entity referenced in the input, "
    'respond with empty string (""). Do NOT guess. Respond with ONLY the IRN integer, '
    "or empty string."
)


class _SafeDict(dict):
    """A dict subclass that returns "" for missing keys when used with format_map."""

    def __missing__(self, key):
        return ""


def normalise_column_name(raw: str) -> str:
    """Normalise a raw spreadsheet column name into a snake_case identifier.

    Rules applied in order:
        1. Strip a single trailing parenthetical ``(...)`` together with surrounding
           whitespace.
        2. Strip a trailing ``:``.
        3. Lowercase.
        4. Replace runs of whitespace and ``/`` with ``_``.
        5. Drop any character not in ``[a-z0-9_]``.
        6. Collapse runs of ``_`` and strip leading/trailing ``_``.
    """
    s = str(raw)

    # 1. Strip a single trailing parenthetical (...) with surrounding whitespace.
    s = re.sub(r"\s*\([^()]*\)\s*$", "", s)

    # 2. Strip a trailing ':'.
    s = re.sub(r":\s*$", "", s)
    s = s.rstrip(":")

    # 3. Lowercase.
    s = s.lower()

    # 4. Replace runs of whitespace AND '/' with '_'.
    s = re.sub(r"[\s/]+", "_", s)

    # 5. Drop any character not in [a-z0-9_].
    s = re.sub(r"[^a-z0-9_]", "", s)

    # 6. Collapse runs of '_' and strip leading/trailing '_'.
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")

    return s


def _is_missing(value) -> bool:
    """Return True if value is None, NaN, or pandas-NA."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        try:
            if math.isnan(value):
                return True
        except (TypeError, ValueError):
            pass
    return False


def render_card(template: str, row: dict) -> str:
    """Render a card from a template and a row mapping.

    Replaces None/NaN values with "", applies ``template.format(**row)``,
    then collapses runs of separator-only segments (``|`` or ``,`` with
    surrounding whitespace) and strips leading/trailing separators.
    """
    cleaned = {k: ("" if _is_missing(v) else str(v)) for k, v in row.items()}
    rendered = template.format_map(_SafeDict(cleaned))

    # Collapse runs of (\s*[|,]\s*){2,} into the LAST single \s*[|,]\s* in the
    # run. We reconstruct the "last single" as: optional single leading space
    # (if the run begins with whitespace), the last separator character, and
    # optional single trailing space (if the run ends with whitespace). This
    # avoids the greedy-matching artefact where earlier iterations of the
    # capture group consume whitespace that should belong to the last match.
    sep_chars = "|,"

    def _collapse(match: re.Match) -> str:
        run = match.group(0)
        # Last separator character in the run.
        last_sep = None
        for ch in reversed(run):
            if ch in sep_chars:
                last_sep = ch
                break
        if last_sep is None:
            return run
        leading = " " if run and run[0].isspace() else ""
        trailing = " " if run and run[-1].isspace() else ""
        return f"{leading}{last_sep}{trailing}"

    rendered = re.sub(r"(\s*[|,]\s*){2,}", _collapse, rendered)

    # Strip leading/trailing whitespace and any leading/trailing | or , plus
    # surrounding whitespace.
    rendered = re.sub(r"^(\s*[|,]\s*)+", "", rendered)
    rendered = re.sub(r"(\s*[|,]\s*)+$", "", rendered)
    rendered = rendered.strip()

    return rendered


class IRNResolver(LLMCanonicaliser):
    """Resolve a free-text query to an IRN identifier.

    Builds one vector sub-index per entry in ``index_filters`` from rows of
    a source xlsx file. At query time, tries each filter in order, retrieves
    the top-``count`` candidates by embedding distance, applies acceptance
    rules (``score_floor`` and optional ``score_gap_eps``), then asks the LLM
    to either pick a candidate IRN or return empty string.

    Attributes:
        source (Path): Path to the xlsx source file.
        card_template (str): str.format template using normalised column names.
        query_template (str): str.format template using input Text names.
        irn_column (str): Name of the IRN column (default "IRN").
        count (int): Number of candidates to retrieve per filter (default 20).
        embedding_model (str): Embedding model name.
        score_floor (float): Maximum acceptable (squared L2) distance for top-1.
        score_gap_eps (float): Minimum gap between top-1 and top-2 distances.
        index_filters (list): Pandas-query strings (or None) - one per sub-index.
    """

    def __init__(self, name: str, model: str, **kwargs):
        # Required kwargs (validate before super-init since parent doesn't know
        # about them).
        if "source" not in kwargs:
            raise ValueError("IRNResolver requires a 'source' kwarg (path to xlsx).")
        if "card_template" not in kwargs:
            raise ValueError("IRNResolver requires a 'card_template' kwarg.")
        if "query_template" not in kwargs:
            raise ValueError("IRNResolver requires a 'query_template' kwarg.")

        # The parent LLMCanonicaliser.__init__ wraps `entities` in
        # Path(...) and then checks .exists()/.suffix. We don't use the
        # parent's entity machinery (retrieve_entities and run are overridden),
        # so we inject a Path-safe sentinel. Using "." makes Path(".") valid
        # and exists() True; the parent then assigns self.entities = Path(".")
        # without triggering file reads.
        kwargs.setdefault("entities", ".")
        super().__init__(name, model, **kwargs)

        self.source: Path = Path(kwargs["source"])
        self.card_template: str = kwargs["card_template"]
        self.query_template: str = kwargs["query_template"]

        self.irn_column: str = kwargs.get("irn_column", "IRN")
        self.count: int = kwargs.get("count", 20)
        self.embedding_model: str = kwargs.get(
            "embedding_model", "text-embedding-3-large"
        )
        self.score_floor: float = kwargs.get("score_floor", 0.5)
        self.score_gap_eps: float = kwargs.get("score_gap_eps", 0.0)
        self.index_filters: list = kwargs.get("index_filters", [None])

        # Augment preamble once with the no-match instruction.
        if self.preamble:
            self.preamble = f"{self.preamble}\n\n{NO_MATCH_INSTRUCTION}"
        else:
            self.preamble = NO_MATCH_INSTRUCTION

        # Build/load one vector store per filter.
        from langchain_openai import OpenAIEmbeddings

        self._embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self._vector_stores: list[VectorStore] = []
        self._build_indices()

    # ------------------------------------------------------------------
    # Index building / caching
    # ------------------------------------------------------------------
    def _source_hash(self) -> str:
        with open(self.source, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _index_path_for(self, i: int) -> Path:
        return self.source.parent / f"{self.source.stem}_irn_index_{i}"

    def _build_indices(self) -> None:
        current_hash = self._source_hash()
        df_raw_cache: pd.DataFrame | None = None

        for i, filter_str in enumerate(self.index_filters):
            index_path = self._index_path_for(i)
            sidecar = index_path / ".source_hash"

            reuse = (
                VectorStore.exists(index_path, self.embedding_model)
                and sidecar.exists()
                and sidecar.read_text().strip() == current_hash
            )

            if reuse:
                logger.info(
                    "Reusing existing IRN index at %s for filter %r",
                    index_path,
                    filter_str,
                )
                self._vector_stores.append(
                    VectorStore.load(index_path, self._embeddings)
                )
                continue

            # Need to (re)build this index.
            if df_raw_cache is None:
                df_raw_cache = pd.read_excel(self.source)
                self.df = df_raw_cache.rename(
                    columns={c: normalise_column_name(c) for c in df_raw_cache.columns}
                )

            # Filter against the raw spreadsheet column names so users can write
            # filters like "Species.isna()" using the actual xlsx headers.
            df = df_raw_cache
            if filter_str is not None:
                df = df.query(filter_str)
            df = df.rename(columns={c: normalise_column_name(c) for c in df.columns})

            normalised_irn_column = normalise_column_name(self.irn_column)
            id_col = (
                self.irn_column
                if self.irn_column in df.columns
                else normalised_irn_column
            )
            if id_col not in df.columns:
                raise ValueError(
                    f"IRN column {self.irn_column!r} (normalised {normalised_irn_column!r}) "
                    f"not found in source columns: {list(df.columns)}"
                )

            ids: list[str] = []
            texts: list[str] = []
            seen: set[str] = set()
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                irn_value = row_dict.get(id_col)
                if _is_missing(irn_value):
                    continue
                irn_str = str(irn_value).strip()
                if not irn_str or irn_str in seen:
                    continue
                card = render_card(self.card_template, row_dict)
                if not card:
                    continue
                ids.append(irn_str)
                texts.append(card)
                seen.add(irn_str)

            vector_store = VectorStore.build(
                index_path, self._embeddings, texts, ids, self.embedding_model
            )

            sidecar.write_text(current_hash)
            logger.info(
                "Built IRN index at %s with %d rows for filter %r",
                index_path,
                len(texts),
                filter_str,
            )
            self._vector_stores.append(vector_store)

    # ------------------------------------------------------------------
    # Retrieval / run
    # ------------------------------------------------------------------
    def retrieve_entities(
        self, query_data: str, filter_index: int = 0
    ) -> list[tuple[str, str]]:
        """Retrieve top-``count`` (id, page_content) pairs for the given filter."""
        vector_store = self._vector_stores[filter_index]
        results = vector_store.similarity_search_with_score(query_data, k=self.count)
        out: list[tuple[str, str]] = []
        for doc, _score in results:
            out.append((str(doc.id), doc.page_content))
        return out

    def _render_query(self, inputs_dict: dict) -> str:
        return self.query_template.format_map(_SafeDict(inputs_dict))

    def _try_filter(
        self,
        filter_index: int,
        query_text: str,
        args: tuple,
    ) -> str | None:
        """Try one filter; return matched IRN string or None."""
        vector_store = self._vector_stores[filter_index]
        results = vector_store.similarity_search_with_score(query_text, k=self.count)
        if not results:
            return None
        if results[0][1] > self.score_floor:
            return None
        if (
            self.score_gap_eps > 0
            and len(results) >= 2
            and (results[1][1] - results[0][1]) < self.score_gap_eps
        ):
            return None

        candidate_ids: list[str] = [str(doc.id) for doc, _ in results]
        candidates_string = "\n".join(
            f"IRN {doc.id}: {doc.page_content}" for doc, _ in results
        )

        # Mirror LLMCanonicaliser.run for the LLM call portion.
        prompt, pattern = self.pre_run()
        messages = self.replace_inputs(
            pattern, prompt, *args, entities=candidates_string
        )
        self.messages.append(HumanMessage(content=messages))
        self.check_inputs(*args)
        response = self.invoke()

        # Parse response: strip whitespace and surrounding quotes.
        cleaned = response.strip()
        if cleaned and cleaned[0] in ("'", '"') and cleaned[-1] == cleaned[0]:
            cleaned = cleaned[1:-1].strip()

        if not cleaned or cleaned.lower() in ("none", "n/a"):
            return None

        if cleaned in candidate_ids:
            return cleaned
        return None

    def run(self, *args: Text) -> Text:
        """Resolve the inputs to an IRN string by trying each filter in order."""
        inputs_dict = {arg.name: str(arg.data) for arg in args}
        query_text = self._render_query(inputs_dict)

        for filter_index in range(len(self._vector_stores)):
            matched = self._try_filter(filter_index, query_text, args)
            if matched is not None:
                return self.text_output(matched)

        logger.info("IRN resolver %s returned empty (no acceptable match)", self.name)
        return self.text_output("")
