"""Tasks whose outputs already exist are skipped unless ``force`` is given."""

from pathlib import Path

import pytest
import yaml

from collectra import Collectra
from collectra.cli import invoke
from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text
from tests.unit.test_pipeline_stages import (  # noqa: F401  (files is a fixture)
    EVENTS,
    Describe,
    build,
    files,
    saved,
)


@pytest.fixture
def done(tmp_path, files):
    """Three .stg folders with every task already run once."""
    build(tmp_path, batch_size=1).run(files=files)
    EVENTS.clear()
    return [str(tmp_path / "input" / f"{name}.stg") for name in "abc"]


def stamp(monkeypatch, label):
    """Make Describe produce different output on each call, as an LLM might."""

    def run(self, image: Image) -> Text:
        stem = Path(image.get_path()).stem
        EVENTS.append(f"describe:{stem}")
        return Text(name=self.get_output_name(), data=f"page {stem} {label}")

    monkeypatch.setattr(Describe, "run", run)


def test_rerunning_existing_results_skips_tasks_that_already_have_output(
    tmp_path, done
):
    build(tmp_path, batch_size=1).run(files=done)
    assert EVENTS == []
    assert saved(tmp_path / "input" / "a.stg", "shouted") == ["PAGE A"]


def test_skipped_tasks_still_let_downstream_tasks_run(tmp_path, done):
    folder = tmp_path / "input" / "a.stg"
    results = yaml.safe_load((folder / "results.yaml").read_text())
    del results["shouted"]
    (folder / "results.yaml").write_text(yaml.safe_dump(results))
    build(tmp_path, batch_size=1).run(files=done)
    assert EVENTS == ["shout:a"]
    assert saved(folder, "shouted") == ["PAGE A"]


def test_force_reruns_tasks_and_replaces_their_output(tmp_path, done, monkeypatch):
    stamp(monkeypatch, "again")
    build(tmp_path, batch_size=1).run(files=done, force=True)
    assert EVENTS[:2] == ["describe:a", "shout:again"]
    # Replaced, not appended: one item per node, and downstream was refreshed.
    assert saved(tmp_path / "input" / "a.stg", "described") == ["page a again"]
    assert saved(tmp_path / "input" / "a.stg", "shouted") == ["PAGE A AGAIN"]


def test_force_keeps_the_old_output_when_the_rerun_produces_nothing(
    tmp_path, done, monkeypatch
):
    def fail(self, image: Image) -> Text:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(Describe, "run", fail)
    build(tmp_path, batch_size=1).run(files=done, force=True)
    assert saved(tmp_path / "input" / "a.stg", "described") == ["page a"]


def test_skipped_tasks_are_not_prefetched_in_a_batch(tmp_path, done):
    build(tmp_path).run(files=done)
    assert EVENTS == ["clear", "clear"]  # no prefetch, one clear per chunk


def test_force_prefetches_batches_again(tmp_path, done, monkeypatch):
    stamp(monkeypatch, "again")
    build(tmp_path).run(files=done, force=True)
    assert [e for e in EVENTS if e.startswith("prefetch")] == [
        "prefetch:2",
        "prefetch:1",
    ]
    assert saved(tmp_path / "input" / "c.stg", "shouted") == ["PAGE C AGAIN"]


def test_force_applies_to_a_single_task_run(tmp_path, done, monkeypatch):
    stamp(monkeypatch, "again")
    pipeline = build(tmp_path, batch_size=1)
    pipeline.run("first", single=True, files=done)
    assert EVENTS == []
    pipeline.run("first", single=True, files=done, force=True)
    assert EVENTS == ["describe:a", "describe:b", "describe:c"]
    assert saved(tmp_path / "input" / "b.stg", "described") == ["page b again"]
    assert saved(tmp_path / "input" / "b.stg", "shouted") == ["PAGE B"]


def test_force_is_a_run_option_not_a_task_override(tmp_path, done, monkeypatch):
    seen = []
    monkeypatch.setattr(
        Collectra,
        "_apply_task_run_overrides",
        lambda self, task, overrides: seen.append(dict(overrides)),
    )
    build(tmp_path, batch_size=1).run(files=done, force=True)
    assert all("force" not in overrides for overrides in seen)


def test_cli_run_passes_force_to_the_pipeline(tmp_path, monkeypatch):
    pipeline = build(tmp_path)
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: calls.append(kwargs))
    folder = tmp_path / "x.stg"
    folder.mkdir()
    (folder / "results.yaml").write_text("{}")
    pipeline.cli_run([str(folder)])
    pipeline.cli_run([str(folder)], force=True)
    assert [call["force"] for call in calls] == [False, True]


def test_task_run_command_accepts_force(tmp_path, monkeypatch):
    pipeline = build(tmp_path)
    pipeline.connect()
    calls = []
    monkeypatch.setattr(
        pipeline, "cli_run", lambda inputs, **kwargs: calls.append(kwargs)
    )
    invoke(pipeline, ["task", "first", "run", "x.stg", "--force"])
    invoke(pipeline, ["task", "first", "run", "x.stg"])
    assert [call["force"] for call in calls] == [True, False]
    assert all(call["single"] for call in calls)


def test_detector_run_commands_accept_force(capsys):
    from collectra.tasks.object_detection.rfdetr import ObjectDetectionRFDETR
    from collectra.tasks.object_detection.yolo import ObjectDetectionYOLO

    for task in (ObjectDetectionYOLO("y"), ObjectDetectionRFDETR("r")):
        with pytest.raises(SystemExit):
            invoke(task, ["run", "--help"], name="task")
        assert "--force" in capsys.readouterr().out


def test_skipping_explains_which_task_file_and_output_and_how_to_rerun(
    tmp_path, done, caplog
):
    with caplog.at_level("WARNING", logger="collectra"):
        build(tmp_path, batch_size=1).run(files=done[:1])
    messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    first = next(m for m in messages if "'first'" in m)
    assert "a.stg" in first
    assert "described" in first
    assert "already" in first
    assert "--force" in first
    assert any("'second'" in m and "shouted" in m for m in messages)


def test_forcing_does_not_print_skip_messages(tmp_path, done, caplog):
    with caplog.at_level("WARNING", logger="collectra"):
        build(tmp_path, batch_size=1).run(files=done[:1], force=True)
    assert not [r for r in caplog.records if "--force" in r.getMessage()]


class Joiner(Task):
    """Needs two inputs; only ``described`` ever has data in these folders."""

    def run(self, described: Text, extra: Text) -> Text:
        EVENTS.append("join")
        return Text(name=self.get_output_name(), data="joined")


def test_a_requested_task_that_lacks_inputs_says_which(tmp_path, done, caplog):
    from tests.unit.test_pipeline_stages import MODULE

    pipeline = build(tmp_path, batch_size=1)
    pipeline.data["joiner"] = {
        "type": "tests.unit.test_pipeline_skip.Joiner",
        "input": ["described", "extra"],
        "output": ["joined"],
    }
    with caplog.at_level("WARNING", logger="collectra"):
        pipeline.run("joiner", single=True, files=done[:1])
    assert EVENTS == []
    message = next(
        r.getMessage() for r in caplog.records if "'joiner'" in r.getMessage()
    )
    assert "did not run" in message
    assert "a.stg" in message
    # Each missing input is on its own coloured line; inputs with data are not listed.
    assert "\n  • [bold red]extra[/bold red]" in message
    assert "described" not in message.split("Missing")[-1]
    assert "only runs once all of its inputs exist" in message


def test_long_lists_of_missing_inputs_are_shortened(tmp_path, done, caplog):
    pipeline = build(tmp_path, batch_size=1)
    names = [f"missing_{n:02d}" for n in range(14)]
    pipeline.data["joiner_many"] = {
        "type": "tests.unit.test_pipeline_skip.Joiner",
        "input": ["described", *names],
        "output": ["joined"],
    }
    with caplog.at_level("WARNING", logger="collectra"):
        pipeline.run("joiner_many", single=True, files=done[:1])
    message = next(
        r.getMessage() for r in caplog.records if "did not run" in r.getMessage()
    )
    assert message.count("•") == 10
    assert "and 4 more" in message


def test_names_that_look_like_markup_are_shown_literally(tmp_path, done, caplog):
    pipeline = build(tmp_path, batch_size=1)
    pipeline.data["joiner_odd"] = {
        "type": "tests.unit.test_pipeline_skip.Joiner",
        "input": ["described", "[red]odd"],
        "output": ["joined"],
    }
    with caplog.at_level("WARNING", logger="collectra"):
        pipeline.run("joiner_odd", single=True, files=done[:1])
    message = next(
        r.getMessage() for r in caplog.records if "did not run" in r.getMessage()
    )
    assert "\\[red]odd" in message


def test_tasks_reached_before_their_other_inputs_are_ready_stay_quiet(
    tmp_path, files, caplog
):
    pipeline = build(tmp_path, batch_size=1)
    pipeline.data["joiner"] = {
        "type": "tests.unit.test_pipeline_skip.Joiner",
        "input": ["described", "extra"],
        "output": ["joined"],
    }
    with caplog.at_level("WARNING", logger="collectra"):
        pipeline.run(files=files[:1])
    assert not [r for r in caplog.records if "did not run" in r.getMessage()]
