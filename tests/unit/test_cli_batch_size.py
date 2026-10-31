"""``task NAME run --batch-size N`` sets how many inputs a task batches together."""

import pytest

from collectra.cli import invoke
from tests.unit.test_pipeline_stages import build


def task_and_calls(tmp_path, monkeypatch):
    pipeline = build(tmp_path)
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("second").get_task()
    calls = []
    monkeypatch.setattr(
        pipeline, "cli_run", lambda inputs, **kwargs: calls.append((inputs, kwargs))
    )
    return pipeline, task, calls


def test_batch_size_defaults_to_the_tasks_own_setting(tmp_path, monkeypatch):
    pipeline, task, calls = task_and_calls(tmp_path, monkeypatch)
    invoke(pipeline, ["task", "second", "run", "x.stg"], name="livy")
    assert task.batch_size == 2  # from the pipeline definition
    assert calls[0][1].get("batch_size") is None


def test_batch_size_option_overrides_the_task(tmp_path, monkeypatch):
    pipeline, task, calls = task_and_calls(tmp_path, monkeypatch)
    invoke(
        pipeline, ["task", "second", "run", "x.stg", "--batch-size", "6"], name="livy"
    )
    assert task.batch_size == 6


def test_batch_size_must_be_positive(tmp_path, monkeypatch):
    pipeline, task, calls = task_and_calls(tmp_path, monkeypatch)
    with pytest.raises(Exception, match="batch_size|batch-size"):
        invoke(
            pipeline,
            ["task", "second", "run", "x.stg", "--batch-size", "0"],
            name="livy",
        )
    assert task.batch_size == 2
    assert calls == []


def test_many_inputs_are_accepted_in_one_command(tmp_path, monkeypatch):
    pipeline, task, calls = task_and_calls(tmp_path, monkeypatch)
    folders = [f"Page{n:03d}.stg" for n in range(300)]
    invoke(
        pipeline,
        ["task", "second", "run", *folders, "--force", "--batch-size", "8"],
        name="livy",
    )
    assert calls[0][0] == folders
    assert calls[0][1]["force"] is True
    assert task.batch_size == 8
