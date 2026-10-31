"""Transformers image classifiers with portable Collectra checkpoints."""

from __future__ import annotations

import copy
import json
import tempfile
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict

from collectra.commons import PositionalFields
from pathlib import Path

from cappa import Arg

from collectra.cli import command
from collectra.types.images import Image
from collectra.types.links import Link
from collectra.utils import threading_locked
from ..machine_learning.training import run_training_command
from .training import (
    ClassificationTrainResult,
    TorchClassifierTask,
    prepare_classification_inputs,
)

__all__ = ["ImageClassifierHuggingFace"]


class _ProcessorTransform(PositionalFields, BaseModel):
    """A picklable per-image transform for DataLoader worker processes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    processor: Any
    fliplr: float = 0.0

    def __call__(self, image):
        import torch
        from PIL import ImageOps

        image = image.convert("RGB")
        if self.fliplr and torch.rand(()).item() < self.fliplr:
            image = ImageOps.mirror(image)
        inputs = self.processor(images=image, return_tensors="pt")
        return {name: value.squeeze(0) for name, value in inputs.items()}


def _class_names(config) -> list[str]:
    return [
        config.id2label.get(index, config.id2label.get(str(index), str(index)))
        for index in range(config.num_labels)
    ]


def _set_classes(config, classes: list[str]) -> None:
    config.num_labels = len(classes)
    config.id2label = dict(enumerate(classes))
    config.label2id = {name: index for index, name in enumerate(classes)}
    config.problem_type = "single_label_classification"


def _head_modules(model):
    """Identify task-specific layers outside the Transformers base model."""
    backbone = model.base_model
    if backbone is model:
        raise ValueError(
            f"{type(model).__name__} does not expose a separate classification backbone"
        )
    backbone_modules = {id(module) for module in backbone.modules()}
    return [
        module
        for module in model.modules()
        if module is not model and id(module) not in backbone_modules
    ]


def _reset_head(model) -> None:
    # Also reset when the new labels have the same count but different meanings.
    reset = False
    for module in _head_modules(model):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
            reset = True
    if not reset:
        raise ValueError(
            f"Cannot reset the classification head of {type(model).__name__}"
        )


class ImageClassifierHuggingFace(TorchClassifierTask):
    """Fine-tune Transformers image classifiers and return predicted class Links."""

    prepare_training_inputs = staticmethod(prepare_classification_inputs)

    def __init__(
        self,
        name: str,
        model: str | Path = "google/vit-base-patch16-224-in21k",
        **kwargs,
    ):
        super().__init__(name, model=model, **kwargs)
        self.original_model_path: Path | None = None
        self._categories: list[str] = []
        self._processor = None
        self._device = "cpu"

    @staticmethod
    def _config_from_dict(data):
        from transformers import AutoConfig

        data = copy.deepcopy(data)
        model_type = data.pop("model_type")
        return AutoConfig.for_model(model_type, **data)

    def _from_state(self, config, state, classes=None):
        from transformers import AutoModelForImageClassification

        changed = classes is not None and classes != _class_names(config)
        if classes is not None:
            _set_classes(config, classes)
        model = AutoModelForImageClassification.from_config(
            config, trust_remote_code=False
        )
        if changed:
            # Keep the trained backbone, but never reuse output rows for different labels.
            backbone = model.base_model
            if backbone is model:
                raise ValueError(
                    f"Cannot adapt the classification head of {type(model).__name__}"
                )
            prefix = next(
                name for name, module in model.named_modules() if module is backbone
            )
            prefix += "."
            backbone.load_state_dict(
                {
                    name[len(prefix) :]: value
                    for name, value in state.items()
                    if name.startswith(prefix)
                }
            )
        else:
            model.load_state_dict(state)
        return model

    def _load(
        self, *, pretrained: bool = True, classes: list[str] | None = None
    ) -> None:
        import torch
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForImageClassification,
        )

        if isinstance(self.model, torch.nn.Module):
            if classes is not None and classes != self._categories:
                self.model = self._from_state(
                    copy.deepcopy(self.model.config), self.model.state_dict(), classes
                )
            if classes is not None:
                _set_classes(self.model.config, classes)
            self._categories = _class_names(self.model.config)
            self.model.to(self._device).eval()
            return
        if not isinstance(self.model, (str, Path)):
            raise ValueError(
                "Model must be a Hugging Face model ID, local model directory, or Collectra checkpoint"
            )
        path = Path(self.model).expanduser()
        if path.is_file():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            required = {"backend", "config", "processor", "state_dict", "classes"}
            if (
                not isinstance(checkpoint, dict)
                or not required.issubset(checkpoint)
                or checkpoint["backend"] != "huggingface"
            ):
                raise ValueError(
                    "Expected a Collectra Hugging Face checkpoint with model config, processor, classes, and state_dict; a bare weight file is not sufficient"
                )
            config = self._config_from_dict(checkpoint["config"])
            if checkpoint["classes"] != _class_names(config):
                raise ValueError(
                    "Checkpoint class names do not match its model configuration"
                )
            # AutoImageProcessor resolves built-in processor types from its standard JSON file.
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "preprocessor_config.json").write_text(
                    json.dumps(checkpoint["processor"])
                )
                self._processor = AutoImageProcessor.from_pretrained(
                    directory,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            model = self._from_state(config, checkpoint["state_dict"], classes)
            self.original_model_path = path.resolve()
        else:
            if isinstance(self.model, Path) and not path.is_dir():
                raise ValueError(
                    f"Model directory or checkpoint does not exist: {path}"
                )
            if not path.exists() and (
                str(self.model).startswith(("/", "./", "../", "~"))
                or path.suffix in {".pt", ".pth", ".bin", ".safetensors"}
            ):
                raise ValueError(
                    f"Model directory or checkpoint does not exist: {path}"
                )
            source = str(path) if path.is_dir() else str(self.model)
            config = AutoConfig.from_pretrained(source, trust_remote_code=False)
            self._processor = AutoImageProcessor.from_pretrained(
                source, trust_remote_code=False
            )
            changed = classes is not None and classes != _class_names(config)
            if classes is not None:
                _set_classes(config, classes)
            # Local checkpoints always retain their learned weights, as with torchvision.
            if pretrained or path.is_dir():
                model = AutoModelForImageClassification.from_pretrained(
                    source,
                    config=config,
                    ignore_mismatched_sizes=changed,
                    trust_remote_code=False,
                )
                if changed:
                    _reset_head(model)
            else:
                model = AutoModelForImageClassification.from_config(
                    config, trust_remote_code=False
                )
            self.original_model_path = path.resolve() if path.is_dir() else None
        self.model = model.to(self._device)
        self._categories = _class_names(self.model.config)
        self.model.eval()

    def _prepare_training_model(self, classes, pretrained, freeze):
        self._load(pretrained=pretrained, classes=classes)
        self.model.requires_grad_(True)
        if freeze:
            _head_modules(
                self.model
            )  # Check that this architecture exposes a separate backbone.
            self.model.base_model.requires_grad_(False)
            if not any(
                parameter.requires_grad for parameter in self.model.parameters()
            ):
                raise ValueError(
                    "No trainable classification head remains after freezing the backbone"
                )

    def _transforms(self, *, training=False, fliplr=0.0):
        return _ProcessorTransform(self._processor, fliplr if training else 0.0)

    def _forward(self, pixels):
        inputs = {name: tensor.to(self._device) for name, tensor in pixels.items()}
        return self.model(**inputs).logits

    @threading_locked()
    def run(self, *args: Image) -> Link:
        """Classify one image or crop and link its predicted class to the input."""
        import torch

        if len(args) != 1 or not isinstance(args[0], Image):
            raise ValueError("This task requires a single Image or ImageCrop input")
        self._device = self._select_device(getattr(self, "device", ""))
        self._load()
        image = args[0]
        with image.pil() as pixels:
            inputs = {
                name: tensor.unsqueeze(0)
                for name, tensor in self._transforms()(pixels).items()
            }
        with torch.inference_mode():
            class_index = self._forward(inputs).argmax(dim=1).item()
        class_name = self._categories[class_index]
        outputs = getattr(self, "output", [])
        outputs = [outputs] if isinstance(outputs, str) else outputs
        if outputs and class_name not in outputs:
            raise ValueError(
                f"Predicted class {class_name!r} is not a configured output: {outputs}"
            )
        return Link(name=class_name, target=image)

    def _checkpoint(self, epoch, metrics):
        return {
            "format_version": 1,
            "backend": "huggingface",
            "config": json.loads(self.model.config.to_json_string(use_diff=False)),
            "processor": json.loads(self._processor.to_json_string()),
            "classes": self._categories,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
            },
            "epoch": epoch,
            "metrics": metrics,
        }

    def _train(self, *images: Image, **kwargs) -> ClassificationTrainResult:
        result = super()._train(*images, **kwargs)
        directory = result.save_dir / "weights" / "huggingface"
        self.model.save_pretrained(directory)
        self._processor.save_pretrained(directory)
        return result

    @command
    def train(
        self,
        inputs: Annotated[
            list[str],
            Arg(
                help="Pipeline data files or directories containing classification labels."
            ),
        ],
        model: Annotated[
            str,
            Arg(
                help="Hugging Face model ID (e.g. google/vit-base-patch16-224-in21k), local save_pretrained directory, or Collectra Hugging Face checkpoint. Omit to keep the configured model."
            ),
        ] = "",
        output: Annotated[
            Path | None,
            Arg(
                help="Directory for extracted train/val images, metrics, and checkpoints."
            ),
        ] = None,
        keep_log: Annotated[
            bool,
            Arg(
                help="Keep extracted images and training logs after saving the model to the pipeline."
            ),
        ] = True,
        validation: Annotated[
            str,
            Arg(
                help="Partition value identifying validation images; omit to train without validation."
            ),
        ] = "",
        exclude: Annotated[
            str, Arg(help="Partition value identifying images to exclude.")
        ] = "",
        epochs: Annotated[int, Arg(help="Number of training epochs.")] = 10,
        batch: Annotated[int, Arg(help="Number of images per training batch.")] = 16,
        learning_rate: Annotated[
            float, Arg(help="Initial learning rate for AdamW.")
        ] = 0.00005,
        weight_decay: Annotated[
            float, Arg(help="Weight decay used for regularization.")
        ] = 0.0001,
        pretrained: Annotated[
            bool,
            Arg(
                help="Load pretrained Hub weights; local directories and checkpoints always retain their saved weights. Disable to initialize from a Hub model configuration."
            ),
        ] = True,
        freeze_backbone: Annotated[
            bool,
            Arg(help="Train the classification head and freeze the model backbone."),
        ] = False,
        early_stop: Annotated[
            int,
            Arg(
                help="Stop after this many epochs without validation-loss improvement; 0 disables."
            ),
        ] = 10,
        min_size: Annotated[
            int,
            Arg(
                help="Skip images or crops with either dimension below this pixel threshold; 0 disables."
            ),
        ] = 0,
        fliplr: Annotated[
            float,
            Arg(
                help="Probability of horizontal flipping during training, from 0 to 1."
            ),
        ] = 0.5,
        workers: Annotated[
            int, Arg(help="Number of data-loading worker processes.")
        ] = 0,
        wandb: Annotated[
            bool,
            Arg(help="Log training metrics to Weights & Biases."),
        ] = False,
        device: Annotated[
            str,
            Arg(
                help="Torch device such as cpu, cuda, cuda:0, or mps; omit for automatic selection."
            ),
        ] = "",
        seed: Annotated[
            int, Arg(help="Random seed for model initialization and data loading.")
        ] = 0,
    ):
        """Fine-tune a Hugging Face image classifier using its saved image processor."""
        result = run_training_command(
            self,
            inputs,
            self._train,
            keep_log=keep_log,
            output=output,
            validation=validation,
            exclude=exclude,
            prepare_inputs=self.prepare_training_inputs,
            model=model,
            epochs=epochs,
            batch=batch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            early_stop=early_stop,
            min_size=min_size,
            fliplr=fliplr,
            workers=workers,
            wandb=wandb,
            device=device,
            seed=seed,
        )
        if self.pipeline is not None:
            saved_model = self.pipeline.data[self.name]["model"]
            self.original_model_path = (self.pipeline.path / saved_model).resolve()
        return result
