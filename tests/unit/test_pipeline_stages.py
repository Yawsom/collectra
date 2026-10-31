"""Batchable tasks anywhere in the DAG are prefetched across a chunk of files.

The pipeline runs a chunk of files up to each batchable task, lets that task
prefetch every file's inputs at once, then resumes each file from saved results.
"""

import shutil
from pathlib import Path

import pytest
import yaml

from collectra import Pipeline
from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text

EVENTS: list[str] = []
MODULE = "tests.unit.test_pipeline_stages"


class Describe(Task):
    def run(self, image: Image) -> Text:
        stem = Path(image.get_path()).stem
        EVENTS.append(f"describe:{stem}")
        return Text(name=self.get_output_name(), data=f"page {stem}")


class Shout(Task):
    """Batchable: ``prefetch`` upper-cases many texts at once for ``run`` to reuse."""

    fail_prefetch: bool = False

    def prefetch(self, entries):
        if self.fail_prefetch:
            raise RuntimeError("batch failed")
        EVENTS.append(f"prefetch:{len(entries)}")
        self.cache = {entry[0]().strip(): entry[0]().upper() for entry in entries}

    def clear_prefetch(self):
        EVENTS.append("clear")
        self.cache = {}

    def run(self, text: Text) -> Text:
        value = text().strip()
        EVENTS.append(f"shout:{value.split()[-1]}")
        cache = getattr(self, "cache", {})
        return Text(name=self.get_output_name(), data=cache.pop(value, value.upper()))


class Count(Task):
    def run(self, text: Text) -> Text:
        EVENTS.append(f"count:{text().split()[-1].lower()}")
        return Text(name=self.get_output_name(), data=str(len(text())))


def build(tmp_path, batch_size=2, with_tail=False, **shout):
    config = {
        "first": {
            "type": f"{MODULE}.Describe",
            "input": ["page"],
            "output": ["described"],
        },
        "second": {
            "type": f"{MODULE}.Shout",
            "input": ["described"],
            "output": ["shouted"],
            "batch_size": batch_size,
            **shout,
        },
    }
    if with_tail:
        config["third"] = {
            "type": f"{MODULE}.Count",
            "input": ["shouted"],
            "output": ["counted"],
        }
    return Pipeline("Staged", "stg", "0.1.0", path=str(tmp_path), **config)


@pytest.fixture
def files(tmp_path, raw_img_path):
    EVENTS.clear()
    directory = tmp_path / "input"
    directory.mkdir()
    paths = []
    for name in "abc":
        path = directory / f"{name}.jpg"
        shutil.copy(raw_img_path, path)
        paths.append(str(path))
    return paths


def saved(folder: Path, key: str) -> list[str]:
    data = yaml.safe_load((folder / "results.yaml").read_text())[key]
    return [item["data"] for item in (data if isinstance(data, list) else [data])]


def test_downstream_task_prefetches_a_whole_chunk_after_its_upstream_ran(
    tmp_path, files
):
    build(tmp_path).run(files=files)
    assert EVENTS == [
        "describe:a", "describe:b", "prefetch:2", "shout:a", "shout:b", "clear",
        "describe:c", "prefetch:1", "shout:c", "clear",
    ]  # fmt: skip
    for name in "abc":
        folder = tmp_path / "input" / f"{name}.stg"
        assert saved(folder, "described") == [f"page {name}"]
        assert saved(folder, "shouted") == [f"PAGE {name.upper()}"]


def test_batching_leaves_per_file_order_unchanged_when_batch_size_is_one(
    tmp_path, files
):
    build(tmp_path, batch_size=1).run(files=files)
    assert EVENTS == [
        "describe:a", "shout:a", "describe:b", "shout:b", "describe:c", "shout:c",
    ]  # fmt: skip


def test_results_go_to_the_output_folder_and_inputs_are_untouched(tmp_path, files):
    output = tmp_path / "out"
    build(tmp_path).run(files=files, output=str(output))
    for name in "abc":
        assert saved(output / f"{name}.stg", "shouted") == [f"PAGE {name.upper()}"]
        assert not (tmp_path / "input" / f"{name}.stg").exists()


def test_a_failed_prefetch_falls_back_to_running_each_file(tmp_path, files):
    build(tmp_path, fail_prefetch=True).run(files=files)
    assert [event for event in EVENTS if event.startswith("shout")] == [
        "shout:a", "shout:b", "shout:c",
    ]  # fmt: skip
    assert saved(tmp_path / "input" / "b.stg", "shouted") == ["PAGE B"]


def test_tasks_after_a_batched_task_still_run_for_every_file(tmp_path, files):
    build(tmp_path, with_tail=True).run(files=files)
    assert [event for event in EVENTS if event.startswith("count")] == [
        "count:a", "count:b", "count:c",
    ]  # fmt: skip
    assert saved(tmp_path / "input" / "c.stg", "counted") == ["6"]


def test_single_runs_do_not_stage(tmp_path, files):
    pipeline = build(tmp_path)
    pipeline.run("first", single=True, files=files)
    assert EVENTS == ["describe:a", "describe:b", "describe:c"]


def test_a_file_that_fails_upstream_does_not_block_the_others(
    tmp_path, files, monkeypatch
):
    original = Describe.run

    def flaky(self, image: Image) -> Text:
        if Path(image.get_path()).stem == "b":
            raise RuntimeError("cannot describe")
        return original(self, image)

    monkeypatch.setattr(Describe, "run", flaky)
    build(tmp_path).run(files=files)
    assert saved(tmp_path / "input" / "a.stg", "shouted") == ["PAGE A"]
    assert saved(tmp_path / "input" / "c.stg", "shouted") == ["PAGE C"]
    # Only the file that survived upstream reaches each chunk's prefetch.
    assert [e for e in EVENTS if e.startswith("prefetch")] == [
        "prefetch:1",
        "prefetch:1",
    ]


class Plain(Task):
    """Declares a batch_size but cannot batch, so it must not be staged."""

    def run(self, text: Text) -> Text:
        EVENTS.append(f"plain:{text().split()[-1].lower()}")
        return Text(name=self.get_output_name(), data=text())


def test_tasks_without_prefetch_are_not_staged(tmp_path, files):
    pipeline = build(tmp_path)
    pipeline.data["second"]["type"] = f"{MODULE}.Plain"
    pipeline.run(files=files)
    assert EVENTS == [
        "describe:a", "plain:a", "describe:b", "plain:b", "describe:c", "plain:c",
    ]  # fmt: skip


def test_consecutive_batched_tasks_each_prefetch_a_whole_chunk(tmp_path, files):
    pipeline = build(tmp_path, with_tail=True)
    pipeline.data["third"] = {
        "type": f"{MODULE}.Shout",
        "input": ["shouted"],
        "output": ["counted"],
        "batch_size": 2,
    }
    pipeline.run(files=files)
    prefetches = [event for event in EVENTS if event.startswith("prefetch")]
    assert prefetches == ["prefetch:2", "prefetch:2", "prefetch:1", "prefetch:1"]
    assert saved(tmp_path / "input" / "a.stg", "counted") == ["PAGE A"]


def test_a_file_that_cannot_be_set_up_is_skipped_but_the_rest_batch(
    tmp_path, files, monkeypatch
):
    pipeline = build(tmp_path)
    original = pipeline._setup_file

    def setup(nodes, file_path, options):
        if Path(file_path).stem == "b":
            raise RuntimeError("cannot open")
        return original(nodes, file_path, options)

    monkeypatch.setattr(pipeline, "_setup_file", setup)
    pipeline.run(files=files)
    assert saved(tmp_path / "input" / "a.stg", "shouted") == ["PAGE A"]
    assert saved(tmp_path / "input" / "c.stg", "shouted") == ["PAGE C"]
