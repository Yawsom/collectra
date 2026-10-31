"""A multi-file run shows one overall bar and says which file it is up to."""

from pathlib import Path

import pytest

from tests.unit.test_pipeline_stages import Describe, build, files  # noqa: F401


class FakeProgress:
    """Records what the pipeline asks the progress bar to do."""

    instances: list["FakeProgress"] = []

    def __init__(self, *columns, **kwargs):
        self.descriptions: list[str] = []
        self.total = None
        self.advanced = 0
        self.entered = self.exited = False
        FakeProgress.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True

    def add_task(self, description, total=None, **fields):
        self.descriptions.append(description)
        self.total = total
        return 0

    def update(self, task_id, **fields):
        if "description" in fields:
            self.descriptions.append(fields["description"])

    def advance(self, task_id, advance=1):
        self.advanced += advance


@pytest.fixture
def progress(monkeypatch):
    FakeProgress.instances = []
    monkeypatch.setattr("collectra.pipelines.base.Progress", FakeProgress)

    def no_inner_bars(*args, **kwargs):
        raise AssertionError("per-task bars must not appear inside the overall bar")

    monkeypatch.setattr("collectra.pipelines.base.track", no_inner_bars)
    return FakeProgress


def test_one_overall_bar_counts_every_file(tmp_path, files, progress):
    build(tmp_path, batch_size=1).run(files=files)
    (bar,) = progress.instances
    assert bar.total == 3
    assert bar.advanced == 3
    assert bar.entered and bar.exited


def test_it_says_which_file_it_is_up_to(tmp_path, files, progress):
    build(tmp_path, batch_size=1).run(files=files)
    (bar,) = progress.instances
    shown = [
        d
        for d in bar.descriptions
        if "a.stg" in d or "a.jpg" in d or "b." in d or "c." in d
    ]
    names = [next(n for n in ("a.", "b.", "c.") if n in d) for d in shown]
    assert names[:3] == ["a.", "b.", "c."]
    assert all("first" in d or "second" in d for d in bar.descriptions)


def test_the_label_names_the_task_being_run(tmp_path, files, progress):
    build(tmp_path, batch_size=1).run("first", single=True, files=files)
    (bar,) = progress.instances
    assert all(d.startswith("first") for d in bar.descriptions)


def test_waiting_on_a_batch_is_visible(tmp_path, files, progress):
    build(tmp_path, batch_size=2).run(files=files)
    (bar,) = progress.instances
    assert any("batch of 2 files" in d for d in bar.descriptions)
    assert any("batch of 1 file" in d and "files" not in d for d in bar.descriptions)


def test_files_that_pass_through_several_stages_are_counted_once(
    tmp_path, files, progress
):
    build(tmp_path, batch_size=2, with_tail=True).run(files=files)
    (bar,) = progress.instances
    assert bar.advanced == 3


def test_a_failing_file_still_counts_as_done(tmp_path, files, progress, monkeypatch):
    original = Describe.run

    def flaky(self, image):
        if Path(image.get_path()).stem == "b":
            raise RuntimeError("cannot describe")
        return original(self, image)

    from collectra.types.images import Image
    from collectra.types.texts import Text

    flaky.__annotations__ = {"image": Image, "return": Text}
    monkeypatch.setattr(Describe, "run", flaky)
    build(tmp_path, batch_size=1).run(files=files)
    (bar,) = progress.instances
    assert bar.advanced == 3


def test_a_single_file_keeps_the_plain_task_bar(tmp_path, files, monkeypatch):
    FakeProgress.instances = []
    monkeypatch.setattr("collectra.pipelines.base.Progress", FakeProgress)
    bars = []
    from collectra.pipelines import base

    real = base.track
    monkeypatch.setattr(
        base, "track", lambda *a, **k: bars.append(k["description"]) or real(*a, **k)
    )
    build(tmp_path, batch_size=1).run(files=files[:1])
    assert FakeProgress.instances == []
    assert bars == ["Running first", "Running second"]


def test_the_bar_is_closed_even_when_the_run_fails(
    tmp_path, files, progress, monkeypatch
):
    pipeline = build(tmp_path, batch_size=1)
    monkeypatch.setattr(
        pipeline,
        "_process_single_file",
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        pipeline._process_files(
            pipeline._get_root_starting_nodes(),
            {
                "files": files,
                "single": False,
                "force": False,
                "output": None,
                "task_overrides": {},
                "usage": False,
                "render": False,
            },
        )
    (bar,) = progress.instances
    assert bar.exited
