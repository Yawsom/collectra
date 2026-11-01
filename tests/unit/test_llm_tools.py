"""An LLM can call named lookups as tools before it gives its answer."""

import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from langchain_core.messages import AIMessage, ToolMessage

from collectra import Pipeline
from collectra.commons import TaskContext
from collectra.tasks.base import Task
from collectra.tasks.llms import LLM, ToolCallingUnsupported
from collectra.types.images import Image
from collectra.types.texts import Text
from tests.unit.test_pipeline_stages import files  # noqa: F401

THIS = "tests.unit.test_llm_tools"
USAGE = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def call(query, id="c1", name="genera_lookup"):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": {"query": query}, "id": id, "type": "tool_call"}
        ],
        usage_metadata=USAGE,
    )


def answer(text):
    return AIMessage(content=text, usage_metadata=USAGE)


class ScriptedModel:
    """A chat model that replies from a script and records every call it gets."""

    def __init__(self, replies, supports_tools=True):
        self.replies = list(replies)
        self.supports_tools = supports_tools
        self.calls: list[list] = []
        self.bound: list[list[str]] = []

    def bind_tools(self, tools):
        if not self.supports_tools:
            raise NotImplementedError("no tool calling here")
        self.bound.append(
            [getattr(tool, "name", None) or tool.__name__ for tool in tools]
        )
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def build(tmp_path, replies=(), tools=("genera_lookup",), **llm):
    (tmp_path / "genera.txt").write_text("Didymograptus\nMonograptus\nClimacograptus\n")
    pipeline = Pipeline(
        "Tools",
        "tls",
        "0.1.0",
        path=str(tmp_path),
        genera_lookup={
            "type": "collectra.EntityLookup",
            "file": "genera.txt",
            "description": "Graptolite genera",
        },
        reader={
            "type": "collectra.LLM",
            "model": "m",
            "template": "{note}",
            "input": "note",
            "output": "answer",
            "tools": list(tools),
            **llm,
        },
    )
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("reader").get_task()
    model = ScriptedModel(replies)
    task.__dict__["llm"] = model
    return pipeline, task, model


def note(text="Didymograpus sp."):
    return Text(name="note", data=text)


# --- the option ---------------------------------------------------------------------


def test_tools_accept_a_name_or_a_list_and_default_to_none():
    assert LLM("t", "m", template="x", tools="a").tools == ["a"]
    assert LLM("t", "m", template="x", tools=["a", "b"]).tools == ["a", "b"]
    assert LLM("t", "m", template="x").tools == []
    assert LLM("t", "m", template="x", tools=None).tools == []


def test_the_number_of_lookups_is_limited_and_validated():
    assert LLM("t", "m", template="x").max_tool_calls == 5
    assert LLM("t", "m", template="x", max_tool_calls=2).max_tool_calls == 2
    with pytest.raises(ValueError, match="max_tool_calls"):
        LLM("t", "m", template="x", max_tool_calls=0)


def test_tools_can_be_combined_with_several_outputs():
    task = LLM("t", "m", template="x", output=["a", "b"], tools=["genera_lookup"])
    assert task.tools == ["genera_lookup"] and task.is_structured


# --- the loop -----------------------------------------------------------------------


def test_the_model_looks_things_up_then_answers(tmp_path):
    _, task, model = build(tmp_path, [call("Didymograpus"), answer("Didymograptus")])
    result = task.run(note())
    assert isinstance(result, Text) and result() == "Didymograptus"
    assert result.name == "answer"
    assert model.bound == [["genera_lookup"]]
    assert len(model.calls) == 2


def test_the_lookup_result_is_fed_back_to_the_model(tmp_path):
    _, task, model = build(
        tmp_path, [call("Didymograpus", id="abc"), answer("Didymograptus")]
    )
    task.run(note())
    second = model.calls[1]
    assert isinstance(second[-2], AIMessage) and second[-2].tool_calls
    returned = second[-1]
    assert isinstance(returned, ToolMessage)
    assert returned.tool_call_id == "abc" and returned.name == "genera_lookup"
    assert "Candidates for 'Didymograpus'" in returned.content
    assert "Didymograptus" in returned.content


def test_a_model_that_needs_no_lookup_just_answers(tmp_path):
    _, task, model = build(tmp_path, [answer("Monograptus")])
    assert task.run(note("Monograptus"))() == "Monograptus"
    assert len(model.calls) == 1


def test_several_lookups_in_one_reply_are_all_answered(tmp_path):
    both = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "genera_lookup",
                "args": {"query": "didymograpus"},
                "id": "a",
                "type": "tool_call",
            },
            {
                "name": "genera_lookup",
                "args": {"query": "monograptus"},
                "id": "b",
                "type": "tool_call",
            },
        ],
        usage_metadata=USAGE,
    )
    _, task, model = build(tmp_path, [both, answer("done")])
    task.run(note())
    replies = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in replies] == ["a", "b"]
    assert "didymograpus" in replies[0].content and "monograptus" in replies[1].content


def test_the_tasks_own_messages_are_not_polluted_by_the_conversation(tmp_path):
    _, task, _ = build(tmp_path, [call("x"), answer("y")])
    task.run(note())
    assert len(task.messages) == 2


def test_an_unknown_tool_name_gets_an_error_the_model_can_read(tmp_path):
    _, task, model = build(tmp_path, [call("x", name="made_up"), answer("fine")])
    assert task.run(note())() == "fine"
    message = [m for m in model.calls[1] if isinstance(m, ToolMessage)][0]
    assert (
        "Unknown tool 'made_up'" in message.content
        and "genera_lookup" in message.content
    )


def test_a_lookup_that_fails_is_reported_to_the_model_not_raised(tmp_path):
    pipeline, task, model = build(tmp_path, [call("x"), answer("fine")])
    with patch.object(
        type(pipeline.resources["genera_lookup"]),
        "search",
        side_effect=RuntimeError("index broke"),
    ):
        assert task.run(note())() == "fine"
    message = [m for m in model.calls[1] if isinstance(m, ToolMessage)][0]
    assert "Lookup failed" in message.content and "index broke" in message.content


def test_lookups_beyond_the_limit_are_refused_and_the_model_must_answer(tmp_path):
    pipeline, task, model = build(
        tmp_path,
        [call("a", "1"), call("b", "2"), call("c", "3"), answer("final")],
        max_tool_calls=2,
    )
    searches = []
    original = type(pipeline.resources["genera_lookup"]).search

    def spy(self, query, limit=None):
        searches.append(query)
        return original(self, query, limit)

    with patch.object(type(pipeline.resources["genera_lookup"]), "search", spy):
        assert task.run(note())() == "final"
    assert searches == ["a", "b"]  # the third was not run
    refusal = [m for m in model.calls[3] if isinstance(m, ToolMessage)][-1]
    assert (
        "limit" in refusal.content.lower() and "final answer" in refusal.content.lower()
    )


def test_a_model_that_never_stops_calling_tools_gives_no_answer(tmp_path, caplog):
    _, task, _ = build(
        tmp_path, [call("x", str(i)) for i in range(20)], max_tool_calls=2
    )
    assert task.run(note()) is None
    assert any("final answer" in r.getMessage() for r in caplog.records)


def test_lookups_are_logged_with_the_task_tool_and_query(tmp_path, caplog):
    _, task, _ = build(tmp_path, [call("Didymograpus"), answer("x")])
    with caplog.at_level("INFO", logger="collectra"):
        task.run(note())
    line = next(
        r.getMessage() for r in caplog.records if "genera_lookup" in r.getMessage()
    )
    assert "reader" in line and "Didymograpus" in line


# --- configuration problems --------------------------------------------------------


def test_a_model_without_tool_calling_raises_a_clear_error(tmp_path):
    _, task, model = build(tmp_path, [])
    model.supports_tools = False
    with pytest.raises(ToolCallingUnsupported, match=r"reader.*tool"):
        task.run(note())


def test_a_task_not_in_a_pipeline_says_so(tmp_path):
    task = LLM("loose", "m", template="{note}", input="note", tools=["genera_lookup"])
    task.__dict__["llm"] = ScriptedModel([])
    with pytest.raises(RuntimeError, match="loose.*pipeline"):
        task.run(note())


# --- usage ---------------------------------------------------------------------------


def test_usage_is_recorded_for_every_turn_of_the_conversation(tmp_path):
    from collectra.commons import TaskContext

    _, task, _ = build(tmp_path, [call("x"), answer("y")])
    task.context = TaskContext(usage_file=tmp_path / "usage.yaml")
    with patch("collectra.tasks.llms.get_llmloader") as loader:
        loader.return_value.LLMWrapper.get_token_count.return_value = {"tokens": 5}
        task.run(note())
    assert loader.return_value.LLMWrapper.get_token_count.call_count == 2
    assert yaml.safe_load((tmp_path / "usage.yaml").read_text()) == {
        "reader": {"tokens": 10}
    }


# --- batching ----------------------------------------------------------------------


class RuleModel:
    """Looks up the text once, then answers; waits at a barrier if given one."""

    def __init__(self, barrier=None, fail_on=None):
        self.barrier = barrier
        self.fail_on = fail_on
        self.turns = 0
        self.lock = threading.Lock()

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        with self.lock:
            self.turns += 1
        text = messages[1].content[-1]["text"]
        if (
            self.fail_on
            and text == self.fail_on
            and not any(isinstance(m, ToolMessage) for m in messages)
        ):
            raise RuntimeError("rate limited")
        if any(isinstance(m, ToolMessage) for m in messages):
            return answer(f"answer:{text}")
        if self.barrier is not None:
            self.barrier.wait(timeout=5)  # only passes if the loops run together
        return call(text)


def entries(*texts):
    return [[Text(name="note", data=t)] for t in texts]


def batched(tmp_path, model, size=3):
    _, task, _ = build(tmp_path, [], batch_size=size)
    task.__dict__["llm"] = model
    return task


def test_prefetch_runs_the_conversations_concurrently(tmp_path):
    model = RuleModel(barrier=threading.Barrier(3))
    task = batched(tmp_path, model)
    task.prefetch(entries("a", "b", "c"))
    assert model.turns == 6  # two turns each
    assert sorted(len(v) for v in task._prefetched.values()) == [1, 1, 1]


def test_run_reuses_the_conversations_without_asking_the_model_again(tmp_path):
    model = RuleModel()
    task = batched(tmp_path, model)
    task.prefetch(entries("a", "b", "c"))
    turns = model.turns
    assert [task.run(*e)() for e in entries("c", "a", "b")] == [
        "answer:c",
        "answer:a",
        "answer:b",
    ]
    assert model.turns == turns


def test_usage_for_prefetched_conversations_is_recorded_when_they_are_used(tmp_path):
    from collectra.commons import TaskContext

    model = RuleModel()
    task = batched(tmp_path, model)
    task.prefetch(entries("a"))
    task.context = TaskContext(usage_file=tmp_path / "usage.yaml")
    with patch("collectra.tasks.llms.get_llmloader") as loader:
        loader.return_value.LLMWrapper.get_token_count.return_value = {"tokens": 5}
        task.run(*entries("a")[0])
    assert loader.return_value.LLMWrapper.get_token_count.call_count == 2


def test_a_conversation_that_fails_in_prefetch_is_retried_alone(tmp_path):
    model = RuleModel(fail_on="b")
    task = batched(tmp_path, model)
    task.prefetch(entries("a", "b"))
    assert len(task._prefetched) == 1  # only "a" was kept
    assert task.run(*entries("a")[0])() == "answer:a"
    assert task.run(*entries("b")[0]) is None  # fails again alone: logged, no output
    assert model.turns >= 3


def test_prefetch_without_tool_support_surfaces_the_error(tmp_path):
    _, task, model = build(tmp_path, [], batch_size=2)
    model.supports_tools = False
    with pytest.raises(ToolCallingUnsupported):
        task.prefetch(entries("a", "b"))


# --- the whole pipeline ----------------------------------------------------------------


class Describe(Task):
    def run(self, image: Image) -> Text:
        return Text(name=self.get_output_name(), data="Didymograpus sp.")


def test_a_pipeline_task_uses_the_lookup_and_saves_the_final_answer(tmp_path, files):
    (tmp_path / "genera.txt").write_text("Didymograptus\nMonograptus\n")
    model = ScriptedModel([call("Didymograpus"), answer("Didymograptus")])
    pipeline = Pipeline(
        "Tools",
        "tls",
        "0.1.0",
        path=str(tmp_path),
        genera_lookup={
            "type": "collectra.EntityLookup",
            "file": "genera.txt",
            "description": "Genera",
        },
        first={"type": f"{THIS}.Describe", "input": ["page"], "output": ["note"]},
        reader={
            "type": "collectra.LLM",
            "model": "m",
            "template": "{note}",
            "input": ["note"],
            "output": "answer",
            "tools": ["genera_lookup"],
        },
    )
    with patch.object(LLM, "llm", property(lambda self: model)):
        pipeline.run(files=files[:1])
    saved = yaml.safe_load((tmp_path / "input" / "a.tls" / "results.yaml").read_text())
    assert saved["answer"]["data"] == "Didymograptus"
    assert saved["answer"]["id"] == "answer"
    assert "genera_lookup" not in saved
    assert model.bound == [["genera_lookup"]]


# --- tools with several outputs ---------------------------------------------------------


def submit(id="s1", **fields):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "LLMOutputs", "args": fields, "id": id, "type": "tool_call"}
        ],
        usage_metadata=USAGE,
    )


def build_structured(tmp_path, replies, **llm):
    (tmp_path / "genera.txt").write_text("Didymograptus\nMonograptus\n")
    pipeline = Pipeline(
        "Tools",
        "tls",
        "0.1.0",
        path=str(tmp_path),
        genera_lookup={
            "type": "collectra.EntityLookup",
            "file": "genera.txt",
            "description": "Graptolite genera",
        },
        reader={
            "type": "collectra.LLM",
            "model": "m",
            "template": "{note}",
            "input": "note",
            "output": ["genus", "age"],
            "tools": ["genera_lookup"],
            **llm,
        },
    )
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("reader").get_task()
    model = ScriptedModel(replies)
    task.__dict__["llm"] = model
    return task, model


def test_the_answer_schema_is_offered_as_a_tool_next_to_the_lookups(tmp_path):
    task, model = build_structured(tmp_path, [submit(genus="G", age="A")])
    task.run(note())
    assert model.bound == [["genera_lookup", "LLMOutputs"]]


def test_the_model_looks_things_up_then_submits_every_field(tmp_path):
    task, model = build_structured(
        tmp_path,
        [call("Didymograpus"), submit(genus="Didymograptus", age="Ordovician")],
    )
    result = task.run(note())
    assert [(t.name, t()) for t in result] == [
        ("genus", "Didymograptus"),
        ("age", "Ordovician"),
    ]
    assert len(model.calls) == 2
    fed_back = model.calls[1][-1]
    assert isinstance(fed_back, ToolMessage) and "Didymograptus" in fed_back.content


def test_a_field_left_out_is_an_empty_text(tmp_path):
    task, _ = build_structured(tmp_path, [submit(genus="Monograptus")])
    result = task.run(note())
    assert [(t.name, t()) for t in result] == [("genus", "Monograptus"), ("age", "")]


def test_a_submit_in_the_same_reply_as_a_lookup_ends_the_conversation(tmp_path):
    both = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "genera_lookup",
                "args": {"query": "x"},
                "id": "l1",
                "type": "tool_call",
            },
            {
                "name": "LLMOutputs",
                "args": {"genus": "G", "age": "A"},
                "id": "s1",
                "type": "tool_call",
            },
        ],
        usage_metadata=USAGE,
    )
    task, model = build_structured(tmp_path, [both])
    assert [t() for t in task.run(note())] == ["G", "A"]
    assert len(model.calls) == 1


def test_a_plain_text_reply_is_told_to_submit_the_answer(tmp_path):
    task, model = build_structured(
        tmp_path, [answer("It is Didymograptus"), submit(genus="G", age="A")]
    )
    assert [t() for t in task.run(note())] == ["G", "A"]
    nudge = model.calls[1][-1]
    assert "LLMOutputs" in nudge.content


def test_a_model_that_never_submits_gives_no_answer(tmp_path, caplog):
    task, _ = build_structured(tmp_path, [answer("hmm")] * 10, max_tool_calls=1)
    assert task.run(note()) is None
    assert "did not give a final answer" in caplog.text


def test_structured_conversations_are_prefetched_and_reused(tmp_path):
    task, model = build_structured(
        tmp_path,
        [call("Didymograpus"), submit(genus="Didymograptus", age="Ordovician")],
        batch_size=2,
    )
    task.prefetch([[note()]])
    assert len(model.calls) == 2
    result = task.run(note())
    assert [t() for t in result] == ["Didymograptus", "Ordovician"]
    assert len(model.calls) == 2


def test_usage_is_recorded_for_each_turn_of_a_structured_conversation(tmp_path):
    task, _ = build_structured(
        tmp_path, [call("Didymograpus"), submit(genus="G", age="A")]
    )
    task.context = TaskContext(usage_file=tmp_path / "usage.yaml")
    with patch("collectra.tasks.llms.get_llmloader") as loader:
        loader.return_value.LLMWrapper.get_token_count.return_value = {"tokens": 5}
        task.run(note())
    assert yaml.safe_load((tmp_path / "usage.yaml").read_text()) == {
        "reader": {"tokens": 10}
    }
