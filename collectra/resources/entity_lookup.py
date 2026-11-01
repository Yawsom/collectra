"""Look up likely entries in a list of known names, with BM25."""

import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from collectra.resources.base import Resource

__all__ = ["EntityLookup", "LookupHit"]


class LookupHit(BaseModel):
    """One candidate from a lookup."""

    entity: str
    score: float  # relative to the best candidate, 0-1
    exact: bool  # the query equals this entry apart from case, accents, punctuation


def _normalise(text: str) -> str:
    """Lower-case, strip accents, and reduce punctuation to single spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", plain.casefold()).strip()


class EntityLookup(Resource):
    """A list of known entities that can be searched by approximate text.

    Entries are indexed with BM25 over whole words and over character n-grams,
    so a query matches a word of a longer entry (``Axedale`` finds ``Parish of
    Axedale, allot 6``) and still finds an entry despite a misspelling
    (``Didymograpus`` finds ``Didymograptus``). The index is built in memory
    when the lookup is created: there are no index files to keep up to date.

    ``file`` holds one entity per line. ``limit`` is the default number of
    candidates returned.
    """

    file: Path
    limit: int = Field(default=10, ge=1)
    ngram: int = Field(default=3, ge=2)
    k1: float = 1.5
    b: float = 0.75

    _entities: list[str] = PrivateAttr(default_factory=list)
    _normalised: list[str] = PrivateAttr(default_factory=list)
    _postings: dict[str, list[tuple[int, int]]] = PrivateAttr(default_factory=dict)
    _lengths: list[int] = PrivateAttr(default_factory=list)
    _idf: dict[str, float] = PrivateAttr(default_factory=dict)
    _average: float = PrivateAttr(default=1.0)

    def model_post_init(self, __context: Any) -> None:
        path = Path(self.file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"EntityLookup {self.name!r}: entity file not found: {self.file}"
            )
        self.file = path
        seen: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            entity = line.strip()
            if entity and entity not in seen:
                seen.add(entity)
                self._entities.append(entity)
        self._build()

    @property
    def entities(self) -> list[str]:
        return list(self._entities)

    # --- the index ---------------------------------------------------------------

    def _tokens(self, normalised: str) -> list[str]:
        tokens = []
        for word in normalised.split():
            tokens.append(f"w:{word}")
            padded = f"^{word}$"
            if len(padded) <= self.ngram:
                tokens.append(f"n:{padded}")
            else:
                tokens.extend(
                    f"n:{padded[i : i + self.ngram]}"
                    for i in range(len(padded) - self.ngram + 1)
                )
        return tokens

    def _build(self) -> None:
        self._normalised = [_normalise(entity) for entity in self._entities]
        documents = [self._tokens(text) for text in self._normalised]
        self._lengths = [len(tokens) for tokens in documents]
        self._average = (sum(self._lengths) / len(documents)) if documents else 1.0
        for index, tokens in enumerate(documents):
            for token, frequency in Counter(tokens).items():
                self._postings.setdefault(token, []).append((index, frequency))
        total = len(documents)
        self._idf = {
            token: math.log(1 + (total - len(posts) + 0.5) / (len(posts) + 0.5))
            for token, posts in self._postings.items()
        }

    # --- searching ---------------------------------------------------------------

    def search(self, query: str, limit: int | None = None) -> list[LookupHit]:
        """Candidates for ``query``, an exact match first, then best score first."""
        normalised = _normalise(query)
        if not normalised or not self._entities:
            return []
        scores: dict[int, float] = {}
        for token in set(self._tokens(normalised)):
            for index, frequency in self._postings.get(token, ()):
                length = self._lengths[index] / self._average
                weight = (
                    frequency
                    * (self.k1 + 1)
                    / (frequency + self.k1 * (1 - self.b + self.b * length))
                )
                scores[index] = scores.get(index, 0.0) + self._idf[token] * weight
        if not scores:
            return []
        best = max(scores.values())
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        exact = [i for i in ranked if self._normalised[i] == normalised]
        order = exact + [i for i in ranked if i not in set(exact)]
        count = limit if limit is not None else self.limit
        return [
            LookupHit(
                entity=self._entities[i],
                score=1.0 if i in exact else round(scores[i] / best, 3),
                exact=i in exact,
            )
            for i in order[:count]
        ]

    # --- the tool an LLM can call ---------------------------------------------------

    def as_tool(self):
        """This lookup as a LangChain tool the model can call with a ``query``."""
        from langchain_core.tools import StructuredTool

        def lookup(query: str) -> str:
            if not query.strip():
                return "Give a non-empty query: the text to look up."
            hits = self.search(query)
            if not hits:
                return (
                    f"No candidates for {query!r}. This list is not exhaustive: "
                    "if you are confident of your reading, keep it."
                )
            lines = [f"Candidates for {query!r} (best first):"]
            for position, hit in enumerate(hits, start=1):
                note = ", exact match" if hit.exact else ""
                lines.append(f"{position}. {hit.entity}  [score {hit.score:.2f}{note}]")
            return "\n".join(lines)

        about = self.description.strip().rstrip(".")
        return StructuredTool.from_function(
            func=lookup,
            name=self.name,
            description=(
                f"Look up known entries{f' ({about})' if about else ''}. Pass the "
                "text you read, or a spelling variant of it. Returns up to "
                f"{self.limit} candidates, best first, each with a relative score; "
                "an exact match is marked. The list is not necessarily complete, "
                "so no match does not mean your reading is wrong."
            ),
        )
