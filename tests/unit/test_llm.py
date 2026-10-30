"""Tests for collectra.llms module"""

from unittest.mock import MagicMock, patch

import pytest
import yaml
from langchain_core.messages import AIMessage

from collectra.commons import TaskContext
from collectra.tasks.llms import LLM
from collectra.types import Image, Text


def test_add_text():
    mock_self = MagicMock(spec=LLM)
    result = LLM._add_text(mock_self, "test text")
    assert result == {"type": "text", "text": "test text"}


def test_add_content():
    mock_self = MagicMock(spec=LLM)

    # Test Text branch
    mock_self._add_text.return_value = {"type": "text", "text": "hello"}
    mock_text = MagicMock(spec=Text)
    mock_text.return_value = "hello"

    result = LLM._add_content(mock_self, mock_text)
    assert result == {"type": "text", "text": "hello"}

    # Test Image branch
    mock_self.image_content.return_value = {
        "type": "image",
        "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
    }
    mock_image = MagicMock(spec=Image)
    result = LLM._add_content(mock_self, mock_image)
    assert result == {"type": "image", "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA"}


def test_image_content():
    mock_self = MagicMock(spec=LLM)
    mock_self.llm = MagicMock()

    mock_image = MagicMock(spec=Image)
    mock_image.get_encoding.return_value = "iiVBORw0KGgoAAAANSUhEUgAAAAUA"
    mock_image.mime.return_value = "image/png"

    with patch("collectra.tasks.llms.get_llmloader") as mock_loader:
        mock_format = mock_loader.return_value.LLMWrapper.format
        mock_format.return_value = {
            "type": "image",
            "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
            "mime_type": "image/png",
        }

        result = LLM.image_content(mock_self, mock_image)

        mock_format.assert_called_once_with(
            mock_self.llm,
            data_type="image",
            data={
                "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
                "mime_type": "image/png",
            },
        )

    assert result == {
        "type": "image",
        "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
        "mime_type": "image/png",
    }


def test_get_pattern_matches():
    mock_self = MagicMock(spec=LLM)
    mock_self.template = "Hello {name}"
    mock_self.preamble = "Intro"

    prompt, pattern = LLM.get_pattern_matches(mock_self)

    assert prompt == "Intro\n\nHello {name}"
    assert pattern == r"\{(.*?)\}"


def test_prompt_is_an_alias_for_template():
    assert LLM("reader", model="m", prompt="Read {image}").template == "Read {image}"
    assert LLM("reader", model="m", template="Read {image}").template == "Read {image}"
    with pytest.raises(ValueError, match="both template and prompt"):
        LLM("reader", model="m", template="a", prompt="b")


def test_empty_prompt_is_reported_clearly():
    with pytest.raises(ValueError, match="has no prompt; set template:"):
        LLM("reader", model="m").get_pattern_matches()


def test_pre_run():
    mock_self = MagicMock(spec=LLM)
    mock_self.messages = [MagicMock(), MagicMock(), MagicMock()]  # system + extras
    mock_self.get_pattern_matches.return_value = ("Intro\n\nHello {name}", r"\{(.*?)\}")

    prompt, pattern = LLM.pre_run(mock_self)

    assert len(mock_self.messages) == 1
    assert prompt == "Intro\n\nHello {name}"
    assert pattern == r"\{(.*?)\}"


def test_check_inputs():
    mock_self = MagicMock(spec=LLM)
    mock_self.messages = [
        MagicMock(),
        MagicMock(content=[{"type": "text", "text": "hello"}]),
    ]

    LLM.check_inputs(mock_self, MagicMock(spec=Text))  # valid - should not raise

    with pytest.raises(ValueError):
        LLM.check_inputs(
            mock_self, MagicMock(spec=Text), MagicMock(spec=Text)
        )  # too many inputs

    mock_self.messages = [MagicMock()]
    with pytest.raises(ValueError):
        LLM.check_inputs(mock_self, MagicMock(spec=Text))  # wrong message count


def test_replace_inputs():
    mock_self = MagicMock(spec=LLM)
    mock_self._add_text.side_effect = lambda t: {"type": "text", "text": t}
    mock_self._add_content.side_effect = lambda v: {"type": "text", "text": v()}

    mock_text = MagicMock(spec=Text)
    mock_text.name = "name"
    mock_text.return_value = "Alice"

    result = LLM.replace_inputs(mock_self, r"\{(.*?)\}", "Hello {name}", mock_text)
    assert {"type": "text", "text": "Alice"} in result

    result = LLM.replace_inputs(mock_self, r"\{(.*?)\}", "Hello {unknown}")
    assert any("No content provided" in r.get("text", "") for r in result)


def test_invoke():
    mock_self = MagicMock()
    mock_self.context.usage_file = None
    mock_self.parser.invoke.return_value = "hello"

    result = LLM.invoke(mock_self)

    assert result == "hello"
    mock_self.chain.invoke.assert_called_once_with(mock_self.messages)

    # empty string handling
    mock_self.parser.invoke.return_value = "''"
    result = LLM.invoke(mock_self)
    assert result == ""


def test_run_returns_text():
    mock_self = MagicMock(spec=LLM)
    mock_self._prefetched = {}
    mock_self._build_messages.return_value = [{"type": "text", "text": "Hello Alice"}]
    mock_self.invoke.return_value = "some response"
    mock_self.get_output_name.return_value = "output_name"

    result = LLM.run(mock_self, MagicMock(spec=Text))

    assert isinstance(result, Text)
    assert result.name == "output_name"
    assert result.data == "some response"


def test_run_returns_none_on_exception():
    mock_self = MagicMock(spec=LLM)
    mock_self.name = "test_task"
    mock_self._build_messages.side_effect = RuntimeError("model failed")

    result = LLM.run(mock_self)

    assert result is None


# --- batching -------------------------------------------------------------


class FakeChain:
    """Stands in for the LangChain model; records every call made to it."""

    def __init__(self, fail_on=()):
        self.invoked = []
        self.batches = []
        self.configs = []
        self.fail_on = fail_on

    @staticmethod
    def _text(messages):
        return messages[1].content[-1]["text"]

    def invoke(self, messages):
        self.invoked.append(self._text(messages))
        return AIMessage(content=f"answer to {self._text(messages)}")

    def batch(self, inputs, config=None, return_exceptions=False):
        self.batches.append([self._text(messages) for messages in inputs])
        self.configs.append(config)
        return [
            (
                RuntimeError("boom")
                if self._text(messages) in self.fail_on
                else AIMessage(content=f"batched {self._text(messages)}")
            )
            for messages in inputs
        ]


def make_task(chain, **kwargs):
    task = LLM("describe", "fake-model", template="{text}", **kwargs)
    task.__dict__["llm"] = chain  # bypasses lazy loading of the real model
    return task


def entries_for(*texts):
    return [[Text(name="text", data=text)] for text in texts]


def test_batch_size_defaults_to_one_and_is_validated():
    assert make_task(FakeChain()).batch_size == 1
    assert make_task(FakeChain(), batch_size=4).batch_size == 4
    with pytest.raises(ValueError, match="batch_size"):
        make_task(FakeChain(), batch_size=0)


def test_prefetch_sends_entries_in_batches_of_batch_size():
    chain = FakeChain()
    task = make_task(chain, batch_size=2)
    task.prefetch(entries_for("a", "b", "c"))
    assert [len(batch) for batch in chain.batches] == [2, 1]
    assert all(config == {"max_concurrency": 2} for config in chain.configs)


def test_run_reuses_prefetched_responses_without_calling_the_model_again():
    chain = FakeChain()
    task = make_task(chain, batch_size=3)
    task.prefetch(entries_for("a", "b"))
    first, second = (task.run(*entry) for entry in entries_for("a", "b"))
    assert chain.invoked == []
    assert (first.data, second.data) == ("batched a", "batched b")


def test_prefetched_response_is_matched_to_its_own_inputs():
    chain = FakeChain()
    task = make_task(chain, batch_size=3)
    task.prefetch(entries_for("a", "b"))
    # Consume out of order: each response must belong to its own prompt.
    assert task.run(*entries_for("b")[0]).data == "batched b"
    assert task.run(*entries_for("a")[0]).data == "batched a"
    assert chain.invoked == []


def test_each_prefetched_response_is_used_once_then_the_model_is_called():
    chain = FakeChain()
    task = make_task(chain, batch_size=2)
    task.prefetch(entries_for("a"))
    assert task.run(*entries_for("a")[0]).data == "batched a"
    assert task.run(*entries_for("a")[0]).data == "answer to a"
    assert chain.invoked == ["a"]


def test_failed_batch_items_fall_back_to_individual_calls():
    chain = FakeChain(fail_on=("b",))
    task = make_task(chain, batch_size=2)
    task.prefetch(entries_for("a", "b"))
    assert task.run(*entries_for("a")[0]).data == "batched a"
    assert task.run(*entries_for("b")[0]).data == "answer to b"
    assert chain.invoked == ["b"]


def test_prefetch_skips_entries_that_cannot_be_built():
    chain = FakeChain()
    task = make_task(chain, batch_size=2)
    task.prefetch([[], *entries_for("a")])
    assert chain.batches == [["a"]]


def test_prefetch_with_nothing_to_send_makes_no_calls():
    chain = FakeChain()
    task = make_task(chain, batch_size=2)
    task.prefetch([])
    assert chain.batches == []


def test_clear_prefetch_discards_cached_responses():
    chain = FakeChain()
    task = make_task(chain, batch_size=2)
    task.prefetch(entries_for("a"))
    task.clear_prefetch()
    assert task.run(*entries_for("a")[0]).data == "answer to a"


def test_usage_is_recorded_for_the_file_being_run_not_the_one_prefetched(tmp_path):
    chain = FakeChain()
    task = make_task(chain, batch_size=2)
    task.prefetch(entries_for("a"))
    usage_file = tmp_path / "usage.yaml"
    task.context = TaskContext(usage_file=usage_file)
    with patch("collectra.tasks.llms.get_llmloader") as loader:
        loader.return_value.LLMWrapper.get_token_count.return_value = {"tokens": 5}
        task.run(*entries_for("a")[0])
    assert yaml.safe_load(usage_file.read_text()) == {"describe": {"tokens": 5}}


# --- temperature ----------------------------------------------------------


def test_temperature_is_not_set_unless_asked_for():
    """Some models reject ``temperature``; the model's own default is the default."""
    assert LLM("t", "m", template="{x}").temperature is None


def test_an_explicit_temperature_is_kept():
    assert LLM("t", "m", template="{x}", temperature=0.2).temperature == 0.2
    assert LLM("t", "m", template="{x}", temperature=0).temperature == 0


def test_the_model_is_loaded_without_a_temperature_by_default():
    with patch("collectra.tasks.llms.get_llmloader") as loader:
        LLM("t", "m", template="{x}").llm
    loader.return_value.load.assert_called_once_with(
        "m", temperature=None, max_tokens=None
    )


def test_the_model_is_loaded_with_the_requested_temperature():
    with patch("collectra.tasks.llms.get_llmloader") as loader:
        LLM("t", "m", template="{x}", temperature=0.3, max_tokens=50).llm
    loader.return_value.load.assert_called_once_with(
        "m", temperature=0.3, max_tokens=50
    )
