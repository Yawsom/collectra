"""Large Language Model (LLM) integration tasks for Collectra workflows.

This module provides integration with various Large Language Models through
the LangChain framework. It supports both text and multimodal (text+image)
processing tasks with configurable prompts and model parameters.

The module includes:
    - LLM task class with configurable model backends
    - Support for text and image inputs
    - Template-based prompt formatting
    - Integration with various LLM providers via llmloader

Classes:
    LLM: Task for Large Language Model inference operations
"""

__all__ = ["LLM"]

import hashlib
import json
import os
import re
from functools import cached_property
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, create_model, field_validator

from collectra.tasks.text_output import TextOutputTask
from collectra.types.images import Image
from collectra.types.texts import Text

from ..logger import get_logger


class StructuredOutputUnsupported(RuntimeError):
    """The model cannot produce structured output (it has no tool calling)."""


def _as_text(cls, value):
    """A field the model left out or filled with a non-string is still a string."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def get_llmloader():
    try:
        import llmloader
    except ModuleNotFoundError as error:
        if error.name != "llmloader":
            raise
        raise ImportError(
            'LLM execution requires optional dependencies. Install with: pip install "collectra[ml]"'
        ) from error
    return llmloader


load_dotenv()

logger = get_logger(__name__)


class LLM(TextOutputTask):
    """Task for Large Language Model inference operations.

    Integrates with various LLM providers to perform text generation and
    multimodal processing tasks. Supports configurable prompts, temperature,
    and token limits with template-based input formatting.

    Attributes:
        name (str): Name identifier for the LLM task.
        model (str): Model identifier or path for the LLM.
        template (str | Path): Prompt template string or path to template file.
            ``prompt`` is accepted as an alias, matching other tasks' option name.
        temperature (float | None): Sampling temperature for generation. Leave
            unset (None) to use the model's own default.
        max_tokens (int): Maximum number of tokens to generate.
        variables (dict): Additional variables for template substitution.
        batch_size (int): Number of prompts sent to the model concurrently when
            the pipeline prefetches many files at once. 1 disables batching.
        output: One node name, or several. With several, the model answers in
            structured form: one string field per node, named after the node,
            and every node gets a Text (empty when the model found nothing).
            The prompt template is where each field is described.
    """

    def __init__(self, name: str, model: str, **kwargs):
        super().__init__(name, **kwargs)
        if kwargs.get("template") and kwargs.get("prompt"):
            raise ValueError(
                f"LLM task {name!r} sets both template and prompt; use only one"
            )
        self.template: str = kwargs.get("template") or kwargs.get("prompt", "")
        self.preamble: str = kwargs.get("preamble", "")
        # None leaves the model at its own default: some models reject the setting.
        self.temperature = kwargs.get("temperature")
        self.max_tokens = kwargs.get("max_tokens", None)
        self.model = model
        self.parser = StrOutputParser()
        if int(self.batch_size) < 1:
            raise ValueError(f"LLM task {name!r} needs batch_size >= 1")
        self.batch_size = int(self.batch_size)
        self._prefetched: dict[str, list] = {}
        if len(self.output_names) != len(set(self.output_names)):
            raise ValueError(f"LLM task {name!r} lists the same output more than once")

        init_messages = (
            SystemMessage(content=kwargs.get("system", ""))
            if kwargs.get("system", "")
            else SystemMessage(content="You are a helpful assistant.")
        )
        self.messages: list[SystemMessage | HumanMessage] = [init_messages]

    @cached_property
    def llm(self):
        """Load the LLM lazily so that it is only instantiated when first needed."""
        return get_llmloader().load(
            self.model, temperature=self.temperature, max_tokens=self.max_tokens
        )

    @property
    def output_names(self) -> list[str]:
        """The output node names, whether given as one name or a list."""
        output = getattr(self, "output", None)
        if output is None:
            return []
        return [output] if isinstance(output, str) else list(output)

    @property
    def is_structured(self) -> bool:
        """Several output nodes mean the model must answer field by field."""
        return len(self.output_names) > 1

    @cached_property
    def output_model(self) -> type[BaseModel]:
        """One optional string field per output node, named after the node.

        Fields are positional (``field_0``...) with the node name as alias, so
        any node name works, even ones pydantic would reject as an attribute.
        """
        fields = {
            f"field_{index}": (str, Field(default="", alias=name))
            for index, name in enumerate(self.output_names)
        }
        return create_model(
            "LLMOutputs",
            __doc__=(
                "Fill in every field from the input. Leave a field as an empty "
                "string when the input does not contain it."
            ),
            __validators__={"as_text": field_validator("*", mode="before")(_as_text)},
            **fields,
        )

    @cached_property
    def structured_llm(self):
        """The model wrapped to answer in ``output_model`` form, with the raw reply."""
        try:
            return self.llm.with_structured_output(self.output_model, include_raw=True)
        except NotImplementedError as error:
            raise StructuredOutputUnsupported(
                f"LLM task {self.name!r} has {len(self.output_names)} outputs, which "
                f"needs structured output, but model {self.model!r} does not support "
                "tool calling. Use a model that does, or give the task one output."
            ) from error

    @property
    def chain(self):
        return self.structured_llm if self.is_structured else self.llm

    def _is_unreadable(self, response) -> bool:
        """A structured reply the schema could not parse (or a failed call)."""
        if isinstance(response, Exception):
            return True
        return self.is_structured and (
            not isinstance(response, dict) or response.get("parsed") is None
        )

    def get_pattern_matches(self) -> tuple:
        prompt = f"{self.preamble}\n\n{self.template}".strip()
        if not prompt:
            raise ValueError(
                f"LLM task {self.name!r} has no prompt; set template: (or prompt:) "
                "in its pipeline.yaml entry"
            )
        pattern = r"\{(.*?)\}"
        return prompt, pattern

    def image_content(self, image: Image):
        return get_llmloader().LLMWrapper.format(
            self.llm,
            data_type="image",
            data={
                "data": image.get_encoding(),
                "mime_type": image.mime(),
            },
        )

    def _add_content(self, value: Image | Text) -> dict:
        if isinstance(value, Image):
            return self.image_content(value)
        return self._add_text(value())

    def _add_text(self, text: str) -> dict:
        text = (
            "''" if len(text.strip()) == 0 else text.strip()
        )  # Avoid empty string issues in some LLMs
        return {"type": "text", "text": text}

    def pre_run(self) -> tuple:
        self.messages = self.messages[:1]  # Reset to initial system message
        prompt, pattern = self.get_pattern_matches()
        return prompt, pattern

    def check_inputs(self, *args: Text | Image) -> None:
        """Validate that the messages are not longer than the expected number of inputs."""
        residual_inputs = len(args)
        if len(self.messages) != 2:
            raise ValueError("Expected one system message and one human message.")
        human_message = self.messages[1]
        for content in human_message.content:
            if isinstance(content, dict) and (
                content.get("type") == "image" or content.get("type") == "text"
            ):
                residual_inputs -= 1
        if residual_inputs > 0:
            raise ValueError("Number of inputs does not match the expected count")

    def replace_inputs(
        self, pattern: str, prompt: str, *args: Text | Image, **kwargs
    ) -> list[str | dict]:
        messages: list[str | dict] = list()
        while re.search(pattern, prompt):
            match = next(re.finditer(pattern, prompt))
            start, end = match.span()
            item = match[1].strip()
            replaced = False
            if prompt[:start]:
                messages.append(self._add_text(prompt[:start]))
            for arg in args:
                key = arg.name
                if key == item:
                    messages.append(self._add_content(arg))
                    replaced = True
                    break
            if item == "entities" and kwargs.get("entities", None):
                replaced = True
                messages.append(self._add_text(kwargs["entities"]))
            if not replaced:
                messages.append(
                    self._add_text(f"No content provided for {item}. Ignore this part.")
                )
            prompt = prompt[end:].strip()

        if prompt:
            messages.append(self._add_text(prompt))

        return messages

    def _build_messages(self, args: tuple) -> list[str | dict]:
        """Fill the prompt template for ``args``, leaving ``self.messages`` ready to send."""
        prompt, pattern = self.pre_run()
        content = self.replace_inputs(pattern, prompt, *args)
        self.messages.append(HumanMessage(content=content))
        self.check_inputs(*args)
        return content

    @staticmethod
    def _cache_key(content: list[str | dict]) -> str:
        """Identify a prompt by its content (images are hashed, not stored)."""
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()

    def prefetch(self, entries: list[list]) -> None:
        """Send the prompts for many entries to the model in concurrent batches.

        Raw responses are cached by prompt content and consumed by ``run``, so
        usage is recorded against the file that is actually being processed. A
        prompt that fails here is simply left for ``run`` to retry on its own.
        """
        self._prefetched = {}
        prompts: list[tuple[str, list]] = []
        for entry in entries:
            if not entry:
                continue
            try:
                content = self._build_messages(tuple(entry))
            except Exception as e:
                logger.warning(
                    "LLM task '%s' could not prepare an entry: %s", self.name, e
                )
                continue
            prompts.append((self._cache_key(content), list(self.messages)))
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            responses = self.chain.batch(
                [messages for _, messages in batch],
                config={"max_concurrency": self.batch_size},
                return_exceptions=True,
            )
            for (key, _), response in zip(batch, responses):
                if self._is_unreadable(response):
                    logger.warning(
                        "LLM task '%s' batch item failed, will retry alone: %s",
                        self.name,
                        response,
                    )
                    continue
                self._prefetched.setdefault(key, []).append(response)

    def clear_prefetch(self) -> None:
        self._prefetched = {}

    def _fields(self, response: dict) -> dict[str, str]:
        """The parsed structured reply as ``{node name: text}``, in output order."""
        parsed = response.get("parsed") if isinstance(response, dict) else None
        if parsed is None:
            error = (
                response.get("parsing_error") if isinstance(response, dict) else None
            )
            raise ValueError(
                f"LLM task {self.name!r} could not read the model's structured "
                f"reply: {error}"
            )
        values = {}
        for index, name in enumerate(self.output_names):
            value = getattr(parsed, f"field_{index}")
            values[name] = "" if value in ('""', "''") else value.strip()
        return values

    def invoke(self, response=None) -> str | dict[str, str]:
        """Ask the model (unless ``response`` is already in hand) and read the reply.

        Returns text for a single output, or ``{node name: text}`` for several.
        """
        if response is None:
            response = self.chain.invoke(self.messages)

        # Use context for usage tracking instead of environment variable
        if self.context.usage_file:
            raw = response["raw"] if self.is_structured else response
            usage = get_llmloader().LLMWrapper.get_token_count(raw)
            usage_file = self.context.usage_file
            data = {f"{self.name}": usage}
            old_data = {}
            if usage_file.exists():
                with open(usage_file, "r") as f:
                    old_data = yaml.safe_load(f) or dict()
            for key, value in old_data.items():
                if key not in data:
                    data[key] = value
                else:
                    for item, item_value in value.items():
                        data[key][item] = data.get(key, {}).get(item, 0) + item_value
            usage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(usage_file, "w") as f:
                yaml.dump(data, f)

        if self.is_structured:
            return self._fields(response)

        response = self.parser.invoke(response)

        return "" if response in ('""', "''") else response.strip()

    def run(self, *args: Text | Image) -> Text | list[Text] | None:
        """Execute LLM inference on the provided inputs with template-based prompt generation.

        For list inputs, each item is processed individually and results are collected
        in a list. The method supports multimodal inputs (text + images) by creating
        appropriate message structures for the LangChain conversation format.

        Args:
            **kwargs: Input data for LLM processing. Keys should match template
                     placeholders. Values can be text strings, file paths (Path objects),
                     Image objects, or lists of these types for batch processing.

        Side Effects:
            Updates self.output dictionary with generated text responses. Output keys
            that contain the input key as a substring will be populated with LLM results.
            Prints a success message when inference completes.
        """
        try:
            content = self._build_messages(args)

            cached = self._prefetched.get(self._cache_key(content))
            response = self.invoke(cached.pop(0) if cached else None)

            if self.is_structured:
                return [
                    self.text_output(value, name=node)
                    for node, value in response.items()
                ]
            return self.text_output(response)
        except StructuredOutputUnsupported:
            raise  # a setup problem, not a bad reply: say so rather than log per page
        except Exception as e:
            logger.error("LLM task '%s' failed: %s", self.name, e, exc_info=True)
            return None
