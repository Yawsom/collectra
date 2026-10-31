"""Hugging Face image-to-text OCR tests using a tiny real Transformers model.

Model tests use the tiny random DeepSeek-OCR-2 checkpoint from the local
Hugging Face cache and are skipped when it has not been downloaded.
"""

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image as PILImage

from collectra import Image, OCRHuggingFace, Text
from collectra.tasks.ocr_huggingface import _rgb, character_error_rate

TINY_MODEL = "optimum-intel-internal-testing/tiny-random-deepseek-ocr-2"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """Save the cached tiny model locally so tests also cover local directories."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(TINY_MODEL, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            TINY_MODEL, local_files_only=True
        )
    except OSError:
        pytest.skip(f"{TINY_MODEL} is not in the Hugging Face cache")
    directory = tmp_path_factory.mktemp("tiny-ocr")
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield directory
    torch.set_num_threads(threads)


def test_character_error_rate():
    assert character_error_rate(["abc"], ["abc"]) == 0
    assert character_error_rate(["abd", ""], ["abc", "de"]) == pytest.approx(3 / 5)
    assert character_error_rate([""], [""]) == 0


def test_transparent_regions_are_flattened_onto_white():
    image = PILImage.new("RGBA", (2, 1), (0, 0, 0, 0))
    image.putpixel((1, 0), (0, 0, 0, 255))
    flattened = _rgb(image)
    assert flattened.mode == "RGB"
    assert flattened.getpixel((0, 0)) == (255, 255, 255)
    assert flattened.getpixel((1, 0)) == (0, 0, 0)


def test_missing_local_model_is_rejected(tmp_path):
    task = OCRHuggingFace("ocr", model=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        task._source()


def test_run_and_exact_training_labels(tiny_model, tmp_path):
    task = OCRHuggingFace("ocr", model=str(tiny_model), max_new_tokens=3, device="cpu")
    task._load()
    assert task._prompt_text() == "<image>\nFree OCR."
    path = tmp_path / "line.png"
    PILImage.new("RGB", (60, 12), "white").save(path)
    texts = task.run(Image(name="line", data=path))
    assert len(texts) == 1 and type(texts[0]) is Text
    assert texts[0].name == "ocr_output"

    images = [PILImage.new("RGB", (60, 12), "white"), PILImage.new("RGB", (30, 8))]
    # "x" directly after "OCR." would merge into one token if tokenized jointly.
    batch = task._training_batch(images, ["hello world", "x"])
    tokenizer = task._tokenizer
    for row, text in zip(batch["labels"].tolist(), ["hello world", "x"]):
        labelled = [token for token in row if token != -100]
        expected = tokenizer(text, add_special_tokens=False).input_ids
        assert labelled == [*expected, tokenizer.eos_token_id]
    assert task._model(**task._to_device(batch)).loss.isfinite()


def make_pipeline(tmp_path, model):
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for index, (text, partition) in enumerate(
        [("ab", "train"), ("ba", "train"), ("ab", "validation"), ("", "train"), ("cd", "test")]
    ):
        path = data_dir / f"{index}.png"
        PILImage.new("RGB", (40, 10), "white").save(path)
        sample_dir = data_dir / f"{index}.htr"
        sample_dir.mkdir()
        (sample_dir / "results.yaml").write_text(
            yaml.safe_dump(
                {
                    "collectra_results_metadata": {"partition": partition},
                    "line_image": {
                        "type": "collectra.Image",
                        "id": f"image{index}",
                        "data": str(path),
                    },
                    "line_text": {
                        "type": "collectra.Text",
                        "id": f"text{index}",
                        "data": text,
                        "parents": f"image{index}",
                    },
                }
            )
        )
    pipeline_file = pipeline_dir / "pipeline.yaml"
    pipeline_file.write_text(
        yaml.safe_dump(
            {
                "collectra_pipeline_metadata": {
                    "name": "htr",
                    "ext": "htr",
                    "version": "1.0",
                },
                "line_reader": {
                    "type": "collectra.OCRHuggingFace",
                    "model": str(model),
                    "input": "line_image",
                    "output": "line_text",
                    "max_new_tokens": 3,
                    "device": "cpu",
                },
            }
        )
    )
    return pipeline_file, data_dir


def test_train_exports_saves_and_reloads(tiny_model, tmp_path):
    from collectra import Pipeline

    pipeline_file, data_dir = make_pipeline(tmp_path, tiny_model)
    pipeline = Pipeline.from_file(pipeline_file)
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("line_reader").get_task()
    output = tmp_path / "run"
    destination = task.train(
        [str(data_dir)],
        output=output,
        validation="validation",
        exclude="test",
        epochs=1,
        batch=2,
    )

    labels = {
        split: [
            json.loads(line)["text"]
            for line in (output / "dataset" / split / "labels.jsonl").read_text().splitlines()
        ]
        for split in ("train", "validation")
    }
    # Empty transcriptions and the excluded partition are left out.
    assert sorted(labels["train"]) == ["ab", "ba"]
    assert labels["validation"] == ["ab"]
    assert (output / "metrics.csv").read_text().splitlines()[0] == "epoch,train_loss,val_loss,cer"

    saved = yaml.safe_load(pipeline_file.read_text())["line_reader"]["model"]
    assert destination == pipeline_file.parent / saved
    assert (destination / "config.json").is_file()

    reloaded = Pipeline.from_file(pipeline_file)
    reloaded.connect()
    reader = reloaded.node_manager.resolve_node("line_reader").get_task()
    texts = reader.run(Image(name="line_image", data=data_dir / "0.png"))
    assert type(texts[0]) is Text


def test_reuse_data_requires_output(tiny_model, tmp_path):
    from collectra import Pipeline

    pipeline_file, data_dir = make_pipeline(tmp_path, tiny_model)
    pipeline = Pipeline.from_file(pipeline_file)
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("line_reader").get_task()
    with pytest.raises(ValueError, match="--reuse-data requires --output"):
        task.train([str(data_dir)], reuse_data=True)


def test_lora_adapters_are_merged_into_saved_model(tiny_model, tmp_path):
    pytest.importorskip("peft")
    from safetensors import safe_open

    from collectra import Pipeline

    pipeline_file, data_dir = make_pipeline(tmp_path, tiny_model)
    pipeline = Pipeline.from_file(pipeline_file)
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("line_reader").get_task()
    destination = task.train(
        [str(data_dir)],
        output=tmp_path / "run",
        validation="validation",
        epochs=1,
        batch=2,
        lora=True,
        lora_rank=2,
        keep_log=False,
    )
    assert not (tmp_path / "run").exists()
    with safe_open(destination / "model.safetensors", "pt") as weights:
        assert not any("lora" in name for name in weights.keys())


def test_wandb_logs_epoch_metrics(tiny_model, tmp_path, monkeypatch):
    import sys
    import types

    from collectra import Pipeline

    class FakeRun:
        def __init__(self, **kwargs):
            self.init, self.logged, self.summary = kwargs, [], {}
            self.finished = False

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, metrics):
            self.logged.append(metrics)

        def finish(self):
            self.finished = True

    runs = []
    fake = types.ModuleType("wandb")
    fake.init = lambda **kwargs: runs.append(FakeRun(**kwargs)) or runs[-1]
    monkeypatch.setitem(sys.modules, "wandb", fake)

    pipeline_file, data_dir = make_pipeline(tmp_path, tiny_model)
    pipeline = Pipeline.from_file(pipeline_file)
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("line_reader").get_task()
    task.train(
        [str(data_dir)],
        output=tmp_path / "run",
        validation="validation",
        epochs=2,
        batch=2,
        early_stop=0,
        wandb=True,
    )
    (run,) = runs
    assert run.init["project"] == "pipeline-line_reader"
    assert run.init["name"] == "run"
    assert run.init["config"]["epochs"] == 2
    assert [metrics["epoch"] for metrics in run.logged] == [1, 2]
    assert {"train_loss", "val_loss", "cer"} <= set(run.logged[0])
    assert "best_cer" in run.summary
    assert run.finished
