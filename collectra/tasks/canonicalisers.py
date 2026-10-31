"""Entity canonicalization and text processing tasks for Collectra workflows.

This module provides specialized tasks for entity recognition and canonicalization
operations. It includes functionality for matching text entities against known
canonical forms using fuzzy string matching algorithms.

The module supports:
    - Entity canonicalization with configurable similarity thresholds
    - Fuzzy string matching using difflib algorithms
    - Entity lists loaded from files or provided directly

Classes:
    EntityCanonicalizer: Task for canonicalizing entities in text
"""

__all__ = ["LLMCanonicaliser"]

import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from collectra.tasks.llms import LLM as CollectraLLM
from collectra.tasks.vector_store import VectorStore
from collectra.types.texts import Text

load_dotenv()


class LLMCanonicaliser(CollectraLLM):
    """Task for canonicalizing entities in text using fuzzy string matching.

    Matches input text against a list of known entities and returns the best
    canonical match if it exceeds the similarity threshold. Can load entity
    lists from files or use provided lists directly.

    Attributes:
        entities (list[str]|Path): List of canonical entities or path to file containing them.
        threshold (float): Minimum similarity threshold for matches (0.0 to 1.0).
    """

    def __init__(self, name: str, model: str, **kwargs):
        super().__init__(name, model, **kwargs)
        self.embedding_model = kwargs.get("embedding_model", "text-embedding-3-large")
        self.threshold = kwargs.get("threshold", 0.8)
        self.count = kwargs.get("count", 5)
        self.entities = kwargs.get("entities", [])
        entities_path = Path(self.entities)
        if not entities_path.exists() and entities_path.suffix == "":
            if isinstance(self.entities, str) or not isinstance(self.entities, list):
                self.entities = [entity for entity in str(self.entities).split(",")]
        else:
            self.entities = entities_path

    def retrieve_entities(self, query_data: str) -> dict:
        from langchain_openai import OpenAIEmbeddings

        index_path = Path(self.entities.stem + "_index.embed")
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        if VectorStore.exists(index_path, self.embedding_model):
            print(f"Loading existing vector store from {index_path}")
            vector_store = VectorStore.load(index_path, embeddings)
        else:
            with open(self.entities, "r") as f:
                raw_data = [line.strip() for line in f.readlines() if line.strip()]
            ids = [str(i) for i in range(len(raw_data))]
            vector_store = VectorStore.build(
                index_path, embeddings, raw_data, ids, self.embedding_model
            )

        entities = dict()

        results = vector_store.similarity_search_with_score(query_data, k=self.count)
        for result in results:
            if result[0].page_content in entities:
                entities[result[0].page_content].append(result[0].id)
            else:
                entities[result[0].page_content] = [result[0].id]
        return entities

    def run(self, *args: Text) -> Text:
        """Canonicalize the input text against known entities.

        Args:
            text (str): Input text to canonicalize.

        Returns:
            str: Canonicalized text if a match is found, otherwise original text.
        """
        prompt, pattern = self.pre_run()

        entity_stem = re.sub("_.*", "", self.name)
        task_inputs = list(args)
        query_data: str = ", ".join(
            [
                str(task_input.data)
                for task_input in task_inputs
                if entity_stem in task_input.name
            ]
        )
        entities = ""

        if isinstance(self.entities, Path):
            entities_dict = self.retrieve_entities(query_data)
            entities = ", ".join(list(entities_dict.keys()))
        elif isinstance(self.entities, list):
            entities = ", ".join(self.entities)
        elif isinstance(self.entities, str):
            entities = self.entities

        messages = self.replace_inputs(pattern, prompt, *args, entities=entities)

        self.messages.append(HumanMessage(content=messages))

        self.check_inputs(*args)

        response = self.invoke()

        return self.text_output(response)
