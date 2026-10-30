"""Image-to-text recognition with Transformers vision-language models.

Any model that loads natively with ``AutoModelForImageTextToText`` and
``AutoProcessor`` is supported, without remote code: for example
DeepSeek-OCR-2, GLM-OCR, LightOnOCR, PaddleOCR-VL and Qianfan-OCR.
Models with a chat template receive the image and prompt as a user message;
others receive the prompt directly, with the image token prepended if absent.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

from cappa import Arg
from PIL import Image as PILImage
from rich.progress import track

from collectra.cli import command
from collectra.tasks.base import Task
from collectra.types.images import Image, open_oriented
from collectra.types.texts import Text
from collectra.utils import threading_locked

__all__ = ["OCRHuggingFace", "character_error_rate"]

DEFAULT_MODEL = "deepseek-community/DeepSeek-OCR-2"
DEFAULT_PROMPT = "Free OCR."
SPLITS = ("train", "validation")


def training_progress(message):
    """Keep progress visible in terminals and redirected cluster job logs."""
    print(f"[OCR] {message}", flush=True)


def _edit_distance(prediction: str, reference: str) -> int:
    previous = list(range(len(reference) + 1))
    for i, predicted in enumerate(prediction, 1):
        current = [i]
        for j, expected in enumerate(reference, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (predicted != expected),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    """Total character edits divided by total reference characters."""
    edits = sum(map(_edit_distance, predictions, references))
    return edits / max(1, sum(len(reference) for reference in references))


def _rgb(image: PILImage.Image) -> PILImage.Image:
    """Flatten masked (transparent) regions onto white, as on a page."""
    if image.mode in {"RGBA", "LA", "P"}:
        image = image.convert("RGBA")
        background = PILImage.new("RGBA", image.size, "white")
        return PILImage.alpha_composite(background, image).convert("RGB")
    return image.convert("RGB")


@dataclass
class OCRSample:
    image: Image
    text: str
    partition: str = ""


class _ExportedLines:
    """Exported line images and transcriptions of one dataset split."""

    def __init__(self, directory: Path):
        self.directory = directory
        with open(directory / "labels.jsonl", encoding="utf-8") as stream:
            self.items = [json.loads(line) for line in stream if line.strip()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return _rgb(open_oriented(self.directory / item["file"])), item["text"]


class OCRHuggingFace(Task):
    """Transcribe images with a Transformers image-text-to-text model."""

    def __init__(
        self,
        name: str,
        model: str | Path = DEFAULT_MODEL,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 256,
        device: str = "",
        dtype: str = "auto",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.model, self.prompt = model, prompt
        self.max_new_tokens, self.device, self.dtype = max_new_tokens, device, dtype
        self._model = None
        self._processor = None

    def attributes_to_ignore(self):
        return super().attributes_to_ignore() | {"_model", "_processor"}

    def input_type(self) -> type:
        return Image

    def output_type(self) -> type:
        return Text

    @staticmethod
    def _select_device(device: str = "") -> str:
        import torch

        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _source(self, model: str | Path | None = None) -> str:
        """Resolve local directories relative to the pipeline, else a Hub model ID."""
        model = str(model or self.model)
        path = Path(model).expanduser()
        if not path.is_absolute() and self.pipeline is not None:
            if (Path(self.pipeline.path) / path).is_dir():
                path = Path(self.pipeline.path) / path
        if path.is_dir():
            return str(path.resolve())
        if model.startswith(("/", "./", "../", "~")):
            raise FileNotFoundError(f"Model directory does not exist: {path}")
        return model

    def _load(self, model: str | Path | None = None) -> None:
        if self._model is not None and model is None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        source = self._source(model)
        device = self._select_device(self.device)
        dtype = self.dtype if self.dtype == "auto" else getattr(torch, self.dtype)
        self._processor = AutoProcessor.from_pretrained(
            source, trust_remote_code=False
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            source, dtype=dtype, trust_remote_code=False
        ).to(device)
        self._model.eval()

    @property
    def _tokenizer(self):
        return self._processor.tokenizer

    def _prompt_text(self) -> str:
        """The text preceding the transcription, for inference and training alike."""
        processor = self._processor
        if getattr(processor, "chat_template", None):
            content = [{"type": "image"}]
            if self.prompt:
                content.append({"type": "text", "text": self.prompt})
            return processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
            )
        image_token = getattr(processor, "image_token", None) or "<image>"
        if image_token in self.prompt:
            return self.prompt
        return f"{image_token}\n{self.prompt}" if self.prompt else image_token

    def _encode_prompts(self, images: list[PILImage.Image]):
        """Left-padded prompts so every row ends where generation begins."""
        tokenizer = self._tokenizer
        side, tokenizer.padding_side = tokenizer.padding_side, "left"
        try:
            return self._processor(
                images=images,
                text=[self._prompt_text()] * len(images),
                padding=True,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = side

    def _to_device(self, inputs: dict) -> dict:
        import torch

        dtype = self._model.dtype
        return {
            name: (
                value.to(self._model.device, dtype=dtype)
                if torch.is_floating_point(value)
                else value.to(self._model.device)
            )
            for name, value in inputs.items()
        }

    def _generate(self, images: list[PILImage.Image]) -> list[str]:
        import torch

        inputs = self._encode_prompts(images)
        with torch.inference_mode():
            generated = self._model.generate(
                **self._to_device(inputs),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        texts = self._processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return [text.strip() for text in texts]

    def _training_batch(self, images: list[PILImage.Image], texts: list[str]):
        """Append each tokenized transcription to its prompt and mask the prompt.

        Tokenizing transcriptions separately keeps the loss boundary exact:
        joint tokenization could merge the last prompt and first text tokens.
        """
        import torch

        tokenizer = self._tokenizer
        prompts = self._encode_prompts(images)
        input_ids, mask = prompts["input_ids"], prompts["attention_mask"].bool()
        # Sequence-aligned extras (e.g. token type ids) are extended with zeros.
        aligned = [
            name
            for name, value in prompts.items()
            if name not in {"input_ids", "attention_mask"}
            and torch.is_tensor(value)
            and value.shape[:2] == input_ids.shape
        ]
        rows = []
        for index, text in enumerate(texts):
            target = tokenizer(text, add_special_tokens=False).input_ids
            target = torch.tensor([*target, tokenizer.eos_token_id])
            prompt = input_ids[index][mask[index]]
            rows.append((index, prompt, target))
        length = max(len(prompt) + len(target) for _, prompt, target in rows)
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        batch = {
            "input_ids": torch.full((len(rows), length), pad_id),
            "attention_mask": torch.zeros((len(rows), length), dtype=torch.long),
            "labels": torch.full((len(rows), length), -100),
        }
        for name in aligned:
            batch[name] = torch.zeros(
                (len(rows), length, *prompts[name].shape[2:]),
                dtype=prompts[name].dtype,
            )
        for index, prompt, target in rows:
            size = len(prompt) + len(target)
            batch["input_ids"][index, :size] = torch.cat([prompt, target])
            batch["attention_mask"][index, :size] = 1
            batch["labels"][index, len(prompt) : size] = target
            for name in aligned:
                batch[name][index, : len(prompt)] = prompts[name][index][mask[index]]
        for name, value in prompts.items():
            if name not in batch and name != "attention_mask":
                batch[name] = value
        return batch

    @threading_locked()
    def run(self, *images: Image) -> list[Text]:
        """Transcribe each image or crop and return one Text per input."""
        if not images or any(not isinstance(image, Image) for image in images):
            raise TypeError("OCRHuggingFace requires one or more Image artefacts")
        self._load()
        texts = self._generate([_rgb(image.pil()) for image in images])
        return [Text(name=self.get_output_name(), data=text) for text in texts]

    def _samples(
        self, inputs: list[str], exclude: str = "", max_items: int | None = None
    ):
        """Pair each transcribed Text output with the image it was read from."""
        from collectra.tasks.machine_learning.training import (
            _prepare_data,
            _training_files,
        )

        if self.pipeline is None:
            raise RuntimeError(
                "Attach the task to a pipeline before preparing training data"
            )
        self.pipeline._ensure_workflow_connected()
        node = self.pipeline.node_manager.resolve_node(self.name)
        children = self.pipeline.node_manager.get_children_artefact(node)
        ancestors = self.pipeline.node_manager.get_ancestor_artefacts(node)
        files = []
        for value in inputs:
            path = Path(value).expanduser().resolve()
            if path.name == "results.yaml" and path.is_file():
                path = path.parent
            if path.is_dir() and (path / "results.yaml").is_file():
                files.append(path)
            else:
                files.extend(_training_files(self, [str(path)]))
        files = list(dict.fromkeys(files))
        samples = []
        for path in track(files, description="Loading transcriptions"):
            # Read each result independently: artefact ids may repeat across datasets.
            texts, _ = _prepare_data(self, [path], children)
            parents, _ = _prepare_data(self, [path], ancestors)
            images = {item.id: item for item in parents if isinstance(item, Image)}
            for text in texts:
                if not isinstance(text, Text) or not str(text.data).strip():
                    continue
                parent_ids = text.parents or []
                parent_ids = [parent_ids] if isinstance(parent_ids, str) else parent_ids
                image = next((images[i] for i in parent_ids if i in images), None)
                if image is None:
                    continue
                partition = text.partition or ""
                if exclude and partition == exclude:
                    continue
                samples.append(OCRSample(image, str(text.data).strip(), partition))
                if max_items is not None and len(samples) >= max_items:
                    return samples
        return samples

    def prepare_training_data(
        self,
        inputs: list[str],
        output: Path,
        validation: str = "",
        exclude: str = "",
        max_items: int | None = None,
    ) -> Path:
        """Export line images to OUTPUT/{train,validation} with labels.jsonl files."""
        samples = self._samples(inputs, exclude, max_items)
        if not samples:
            raise ValueError(
                f"No transcribed images found for the outputs of {self.name!r}"
            )
        for split in SPLITS:
            if (output / split).exists():
                shutil.rmtree(output / split)
            (output / split).mkdir(parents=True)
        counts = dict.fromkeys(SPLITS, 0)
        labels = {
            split: open(output / split / "labels.jsonl", "w", encoding="utf-8")
            for split in SPLITS
        }
        try:
            for sample in track(samples, description="Exporting images"):
                split = (
                    "validation"
                    if validation and sample.partition == validation
                    else "train"
                )
                name = f"{counts[split]:07d}.png"
                _rgb(sample.image.pil()).save(output / split / name)
                labels[split].write(
                    json.dumps({"file": name, "text": sample.text}, ensure_ascii=False)
                    + "\n"
                )
                counts[split] += 1
        finally:
            for stream in labels.values():
                stream.close()
        training_progress(
            f"Exported {counts['train']} training and {counts['validation']} validation images"
        )
        if not counts["train"]:
            raise ValueError("No training images remain after partitioning")
        return output

    def _evaluate(self, dataset, batch: int) -> dict:
        import torch

        self._model.eval()
        losses, predictions, references = [], [], []
        for start in track(range(0, len(dataset), batch), description="Validating"):
            images, texts = zip(*(dataset[i] for i in range(start, min(start + batch, len(dataset)))))
            with torch.inference_mode():
                inputs = self._to_device(self._training_batch(list(images), list(texts)))
                losses.append(self._model(**inputs).loss.item())
            predictions.extend(self._generate(list(images)))
            references.extend(texts)
        return {
            "val_loss": sum(losses) / len(losses),
            "cer": character_error_rate(predictions, references),
        }

    @command
    def train(
        self,
        inputs: Annotated[
            list[str],
            Arg(
                required=False,
                default=[],
                help="Results YAML files, result directories, or parent directories containing transcribed images. May be omitted when reusing an existing export.",
            ),
        ],
        model: Annotated[
            str,
            Arg(
                help="Hugging Face model ID or local save_pretrained directory to fine-tune. Omit to continue from the configured model."
            ),
        ] = "",
        output: Annotated[
            Path | None,
            Arg(
                help="Directory for exported images, metrics.csv and the best model. Defaults to a timestamped directory; required with --reuse-data."
            ),
        ] = None,
        validation: Annotated[
            str,
            Arg(
                help="Partition used for validation and best-model selection by character error rate. Omit to train without validation."
            ),
        ] = "",
        exclude: Annotated[
            str,
            Arg(help="Partition excluded from both training and validation."),
        ] = "",
        epochs: Annotated[int, Arg(help="Number of training epochs.")] = 5,
        batch: Annotated[int, Arg(help="Number of images per batch.")] = 4,
        accumulate: Annotated[
            int,
            Arg(help="Number of batches to accumulate before each optimizer step."),
        ] = 1,
        learning_rate: Annotated[
            float, Arg(help="Initial learning rate for AdamW.")
        ] = 0.00001,
        weight_decay: Annotated[
            float, Arg(help="Weight decay used for regularization.")
        ] = 0.0,
        lora: Annotated[
            bool,
            Arg(
                help="Train low-rank adapters (requires peft) instead of all weights; adapters are merged into the saved model."
            ),
        ] = False,
        lora_rank: Annotated[int, Arg(help="Rank of the LoRA adapters.")] = 16,
        freeze_vision: Annotated[
            bool,
            Arg(
                help="Freeze vision-encoder parameters and train only the language model and projector."
            ),
        ] = False,
        gradient_checkpointing: Annotated[
            bool,
            Arg(help="Recompute activations in the backward pass to save memory."),
        ] = False,
        early_stop: Annotated[
            int,
            Arg(
                help="Stop after this many epochs without character-error-rate improvement; 0 disables."
            ),
        ] = 3,
        max_items: Annotated[
            int | None,
            Arg(
                help="Maximum transcribed images to load across both splits, after exclusions. Omit for all."
            ),
        ] = None,
        reuse_data: Annotated[
            bool,
            Arg(
                help="Reuse OUTPUT/dataset when both splits exist instead of exporting again. Requires --output."
            ),
        ] = False,
        keep_log: Annotated[
            bool,
            Arg(
                help="Keep exported data, metrics and checkpoints after copying the best model into the pipeline."
            ),
        ] = True,
        seed: Annotated[
            int, Arg(help="Random seed for shuffling and initialization.")
        ] = 0,
        wandb: Annotated[
            bool,
            Arg(
                help="Log epoch metrics to Weights & Biases. Project: <pipeline-directory>-<task>; run: output directory name; entity: W&B settings or WANDB_ENTITY."
            ),
        ] = False,
    ) -> Path:
        """Fine-tune the model on transcribed images and save the best model to the pipeline."""
        import torch

        if min(epochs, batch, accumulate, lora_rank) < 1:
            raise ValueError("epochs, batch, accumulate and lora_rank must be positive")
        if reuse_data and output is None:
            raise ValueError("--reuse-data requires --output")
        if self.pipeline is None:
            raise RuntimeError(f"Task {self.name!r} is not attached to a pipeline")
        output = (
            Path(output).expanduser().resolve()
            if output is not None
            else Path.cwd() / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        dataset_dir = output / "dataset"
        if reuse_data and all(
            (dataset_dir / split / "labels.jsonl").is_file() for split in SPLITS
        ):
            training_progress(f"Reusing exported dataset: {dataset_dir}")
        else:
            if not inputs:
                raise ValueError("Supply training inputs to export the dataset")
            self.prepare_training_data(
                inputs, dataset_dir, validation, exclude, max_items=max_items
            )
        train_set = _ExportedLines(dataset_dir / "train")
        val_set = _ExportedLines(dataset_dir / "validation")
        if not len(train_set):
            raise ValueError(f"No training images in {dataset_dir / 'train'}")

        random.seed(seed)
        torch.manual_seed(seed)
        training_progress(f"Loading {model or self.model}")
        self._load(model or self.model)
        network = self._model
        if gradient_checkpointing:
            network.gradient_checkpointing_enable()
            if lora:
                network.enable_input_require_grads()
        if lora:
            try:
                from peft import (
                    LoraConfig,
                    get_peft_model,
                    get_peft_model_state_dict,
                    set_peft_model_state_dict,
                )
            except ModuleNotFoundError as error:
                raise ImportError("--lora requires peft: pip install peft") from error
            network = get_peft_model(
                network,
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=2 * lora_rank,
                    target_modules="all-linear",
                    lora_dropout=0.05,
                ),
            )
        if freeze_vision:
            frozen = 0
            for name, parameter in network.named_parameters():
                if any(part in name for part in ("vision", "visual", "sam_model")):
                    parameter.requires_grad_(False)
                    frozen += 1
            if not frozen:
                raise ValueError(
                    f"No vision parameters found to freeze in {type(self._model).__name__}"
                )
        parameters = [p for p in network.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters remain")
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
        steps = epochs * -(-len(train_set) // (batch * accumulate))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.0, 1 - step / max(1, steps))
        )
        training_progress(
            f"Training {self.name}: {len(train_set)} training and {len(val_set)} validation images, "
            f"epochs={epochs}, batch={batch}x{accumulate}, device={self._model.device}"
        )

        best_dir = output / "best"
        metrics_path = output / "metrics.csv"
        metrics_path.write_text("epoch,train_loss,val_loss,cer\n")
        best_cer, stale, best_adapters = float("inf"), 0, None
        started = time.monotonic()
        run = None
        if wandb:
            try:
                import wandb as sdk
            except ModuleNotFoundError as error:
                raise ImportError("--wandb requires wandb: pip install wandb") from error
            run = sdk.init(
                project=f"{self.pipeline.name}-{self.name}",
                name=output.name,
                dir=str(output),
                config={
                    "model": str(model or self.model),
                    "prompt": self.prompt,
                    "train_images": len(train_set),
                    "validation_images": len(val_set),
                    "epochs": epochs,
                    "batch": batch,
                    "accumulate": accumulate,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "lora": lora,
                    "lora_rank": lora_rank if lora else None,
                    "freeze_vision": freeze_vision,
                    "gradient_checkpointing": gradient_checkpointing,
                    "seed": seed,
                },
            )
            run.define_metric("epoch")
            run.define_metric("*", step_metric="epoch")
        try:
            for epoch in range(1, epochs + 1):
                network.train()
                order = list(range(len(train_set)))
                random.shuffle(order)
                total, count = 0.0, 0
                optimizer.zero_grad()
                batches = [order[i : i + batch] for i in range(0, len(order), batch)]
                for step, indices in enumerate(
                    track(batches, description=f"Epoch {epoch}/{epochs}"), 1
                ):
                    images, texts = zip(*(train_set[i] for i in indices))
                    inputs_ = self._to_device(
                        self._training_batch(list(images), list(texts))
                    )
                    loss = network(**inputs_).loss
                    (loss / accumulate).backward()
                    total, count = total + loss.item(), count + 1
                    if step % accumulate == 0 or step == len(batches):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                metrics = {"train_loss": total / count}
                if len(val_set):
                    metrics |= self._evaluate(val_set, batch)
                training_progress(
                    f"Epoch {epoch}: "
                    + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
                )
                if run is not None:
                    run.log({"epoch": epoch, **metrics})
                with open(metrics_path, "a") as stream:
                    stream.write(
                        f"{epoch},{metrics['train_loss']},{metrics.get('val_loss', '')},{metrics.get('cer', '')}\n"
                    )
                # Without validation the latest epoch is kept.
                cer = metrics.get("cer", -epoch)
                if cer < best_cer:
                    best_cer, stale = cer, 0
                    if run is not None and "cer" in metrics:
                        run.summary["best_cer"] = cer
                        run.summary["best_epoch"] = epoch
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    if lora:
                        # Adapters are small: keep the best in memory and merge once at the end.
                        best_adapters = {
                            name: value.detach().cpu().clone()
                            for name, value in get_peft_model_state_dict(network).items()
                        }
                    else:
                        network.save_pretrained(best_dir)
                        self._processor.save_pretrained(best_dir)
                else:
                    stale += 1
                    if early_stop and stale >= early_stop:
                        training_progress(f"Stopping early after {epoch} epochs")
                        break
        finally:
            if run is not None:
                run.finish()

        if lora:
            set_peft_model_state_dict(network, best_adapters)
            network.merge_and_unload().save_pretrained(best_dir)
            self._processor.save_pretrained(best_dir)

        destination = self.pipeline.path / f"{self.name}-{output.name}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(best_dir, destination)
        self.model = destination.name
        self._model = self._processor = None
        self.pipeline.data[self.name]["model"] = destination.name
        self.pipeline.save(self.name)
        training_progress(
            f"Saved model to {destination} ({time.monotonic() - started:.1f}s)"
        )
        if not keep_log:
            shutil.rmtree(output, ignore_errors=True)
        return destination
