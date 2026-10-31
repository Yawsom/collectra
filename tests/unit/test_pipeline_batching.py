"""Starting tasks with batch_size > 1 prefetch per chunk of files, then save per file."""

import shutil
from pathlib import Path

from collectra import Pipeline


def build_pipeline(pipeline: dict) -> Pipeline:
    metadata: dict = pipeline.pop("collectra_pipeline_metadata")
    path = pipeline.pop("pipeline_path", "")
    return Pipeline(
        metadata["name"], metadata["ext"], metadata["version"], path=path, **pipeline
    )


def test_files_are_prefetched_in_chunks(pipeline, tmpdir, raw_img_path, monkeypatch):
    pipeline = build_pipeline(pipeline)
    pipeline.connect()
    start = pipeline._get_root_starting_nodes()[0]
    task = start.get_task()
    task.batch_size = 2

    input_dir = Path(tmpdir) / "input"
    input_dir.mkdir()
    files = []
    for i in range(5):
        path = input_dir / f"image_{i}.jpg"
        shutil.copy(raw_img_path, path)
        files.append(str(path))

    events = []
    monkeypatch.setattr(
        type(task), "prefetch",
        lambda self, entries: events.append(("prefetch", len(entries))),
    )
    monkeypatch.setattr(
        type(task), "clear_prefetch", lambda self: events.append(("clear",))
    )
    monkeypatch.setattr(
        pipeline, "_process_single_file",
        lambda nodes, file_path, options: events.append(("file", Path(file_path).name)),
    )

    pipeline.run(start.name, single=True, files=files, output=str(Path(tmpdir) / "out"))

    # The fixture task has two entries per file, so prefetch sees 2 per file.
    assert events == [
        ("prefetch", 4), ("file", "image_0.jpg"), ("file", "image_1.jpg"), ("clear",),
        ("prefetch", 4), ("file", "image_2.jpg"), ("file", "image_3.jpg"), ("clear",),
        ("prefetch", 2), ("file", "image_4.jpg"), ("clear",),
    ]
