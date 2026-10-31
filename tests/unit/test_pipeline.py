import shutil
from pathlib import Path
from types import SimpleNamespace

from collectra import Pipeline, Task, TaskNode


def build_pipeline(pipeline: dict) -> Pipeline:
    metadata: dict = pipeline.pop("collectra_pipeline_metadata")
    name = metadata["name"]
    ext = metadata["ext"]
    version = metadata["version"]
    path = pipeline.pop("pipeline_path", "")
    return Pipeline(name, ext, version, path=path, **pipeline)


def test_pipeline_init(pipeline, debug):
    try:
        name = pipeline["collectra_pipeline_metadata"]["name"]
        ext = pipeline["collectra_pipeline_metadata"]["ext"]
        version = pipeline["collectra_pipeline_metadata"]["version"]
        pipeline = build_pipeline(pipeline)
        assert pipeline.name == name
        assert pipeline.ext == ext
        assert pipeline.version == version
        assert pipeline.path
        assert (
            isinstance(pipeline.data, dict) and len(pipeline.data) > 0
        ), "Pipeline data should be a non-empty dictionary"
    except Exception as e:
        debug(e)


def test_run_task_reports_rich_progress(tmp_path, monkeypatch):
    pipeline = Pipeline("progress", "collectra", "1", path=tmp_path)
    task = Task("line_reader")
    task.prepare_inputs = lambda parents: [["first"], ["second"], ["third"]]
    task_node = TaskNode("line_reader", task)
    monkeypatch.setattr(
        pipeline.node_manager, "get_parents_artefact", lambda node: []
    )
    monkeypatch.setattr(pipeline, "_execute_entries", lambda entry, task: [])
    progress_call = {}

    def fake_track(iterable, *, total, description):
        progress_call.update(total=total, description=description)
        yield from iterable

    monkeypatch.setattr("collectra.pipelines.base.track", fake_track)

    pipeline._run_task(task_node)

    assert progress_call == {
        "total": 3,
        "description": "Running line_reader",
    }


def test_pipeline_init_nodes(pipeline, debug, tmpdir):
    try:
        pipeline = build_pipeline(pipeline)
        pipeline.connect()
        pipeline.render(Path(tmpdir) / "pipeline_diagram")
    except Exception as e:
        debug(e)


def test_get_task(pipeline, debug):
    try:
        pipeline = build_pipeline(pipeline)
        task_name = "object_detector"
        task = pipeline.task(task_name)
        assert (
            isinstance(task, dict) and len(task) > 0
        ), f"Task '{task_name}' should be a non-empty dictionary"
        assert (
            task.get("type", None) == "collectra.ObjectDetectionYOLO"
        ), f"Task '{task_name}' should have type 'collectra.ObjectDetectionYOLO'"
    except Exception as e:
        debug(e)


def test_train(pipeline, tmp_path, monkeypatch):
    from collectra.utils import change_dir

    pipeline = build_pipeline(pipeline)
    task_name = "object_detector"
    # One Collectra folder is sufficient to exercise pipeline loading and the
    # training lifecycle; the backend boundary is faked below.
    subset = tmp_path / "inputs"
    subset.mkdir()
    shutil.copytree(Path.cwd() / "tests/data/images/bar1.arb", subset / "bar1.arb")
    input_data = [subset]
    training_output = tmp_path / "chosen-training-output"
    with change_dir(tmp_path):
        pipeline.connect()
        task = pipeline.cli_tasks()[task_name]

        def fake_train(*images, **kwargs):
            assert images
            save_dir = kwargs["base_folder"] / kwargs["log"]
            weights = save_dir / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").touch()
            return SimpleNamespace(save_dir=save_dir, results_dict={"mock": True})

        monkeypatch.setattr(task, "_train", fake_train)
        monkeypatch.setattr(
            "collectra.tasks.object_detection.yolo.prepare_object_detection_inputs",
            lambda processed_inputs, *args: processed_inputs,
        )
        monkeypatch.setattr(
            "collectra.tasks.machine_learning.training.save_training_result",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(pipeline, "save", lambda *args, **kwargs: None)
        result = task.train(
            input_data,
            validation="true",
            output=training_output,
        )

    assert result.results_dict == {"mock": True}
    assert result.save_dir == training_output


def test_run_full(pipeline, debug, tmpdir, raw_img_path):
    try:
        pipeline = build_pipeline(pipeline)
        shutil.copy(raw_img_path, tmpdir / "bar1.jpg")
        pipeline.run(bar_img1=tmpdir / "bar1.jpg")
    except Exception as e:
        debug(e)


def test_run_one_task(pipeline, debug, tmpdir, raw_img_path):
    try:
        pipeline = build_pipeline(pipeline)
        file_path = "tests/data/images/bar1.arb"
        tempfile = shutil.copytree(file_path, tmpdir / "bar1.arb")
        shutil.copy(raw_img_path, tmpdir / "bar1.jpg")
        pipeline.run(
            "markdown_converter",
            bar_img1=tmpdir / "bar1.jpg",
            file=tempfile,
        )
    except Exception as e:
        debug(e)


def test_run_single_task(pipeline, debug, tmpdir):
    try:
        pipeline = build_pipeline(pipeline)
        file_path = "tests/data/images/bar1.arb"
        tempfile = shutil.copytree(file_path, tmpdir / "bar1.arb")
        pipeline.run("object_detector", single=True, file=tempfile)
    except Exception as e:
        debug(e)
