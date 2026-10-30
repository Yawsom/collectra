"""Ultralytics training on bitmap masks, imported only by the training runner."""

from copy import copy, deepcopy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.utils import __version__

from collectra.types.images import open_oriented

from .mask_metrics import mask_overlap
from .sam3_metrics import EpochReporter


def decode_mask(rle):
    """Decode Collectra's lossless COCO RLE without requiring pycocotools."""
    counts = rle["counts"]
    if not isinstance(counts, list):
        from pycocotools.mask import decode

        return decode(rle).astype(bool)
    pixels = np.zeros(math.prod(rle["size"]), dtype=np.uint8)
    position = 0
    for index, count in enumerate(counts):
        if count < 0 or position + count > len(pixels):
            raise ValueError("Invalid COCO mask run length")
        pixels[position:position + count] = index % 2
        position += count
    if position != len(pixels):
        raise ValueError("COCO mask runs do not match the mask dimensions")
    return pixels.reshape(rle["size"], order="F")


class BitmapDataset(Dataset):
    """Resize/pad image and masks together, retaining holes and separate islands.

    Uses horizontal/vertical flips for training. Polygon-based mosaic, random
    perspective and mixup are deliberately bypassed, so instance masks remain
    bitmaps throughout preparation and training.
    """

    rect = False
    mosaic = False
    collate_fn = staticmethod(YOLODataset.collate_fn)

    def __init__(self, directory, imgsz, *, augment=False, fliplr=.5, flipud=0):
        self.directory = Path(directory)
        data = json.loads((self.directory / "_annotations.coco.json").read_text())
        self.images = data["images"]
        self.categories = {item["id"]: index for index, item in enumerate(sorted(data["categories"], key=lambda item: item["id"]))}
        self.annotations = {}
        for annotation in data["annotations"]:
            self.annotations.setdefault(annotation["image_id"], []).append(annotation)
        self.imgsz, self.augment, self.fliplr, self.flipud = imgsz, augment, fliplr, flipud
        self.im_files = [str(self.directory / image["file_name"]) for image in self.images]
        self.labels = []
        for image, path in zip(self.images, self.im_files):
            annotations = self.annotations.get(image["id"], [])
            boxes = np.array([a["bbox"] for a in annotations], dtype=np.float32).reshape(-1, 4)
            boxes[:, :2] += boxes[:, 2:] / 2
            boxes /= [image["width"], image["height"], image["width"], image["height"]]
            self.labels.append({"im_file": path, "shape": (image["height"], image["width"]),
                                "cls": np.array([self.categories[a["category_id"]] for a in annotations], dtype=np.float32).reshape(-1, 1),
                                "bboxes": boxes, "normalized": True, "bbox_format": "xywh"})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        metadata = self.images[index]
        image = open_oriented(self.im_files[index]).convert("RGB")
        width, height = image.size
        scale = self.imgsz / max(width, height)
        new_width, new_height = round(width * scale), round(height * scale)
        left, top = (self.imgsz - new_width) // 2, (self.imgsz - new_height) // 2
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        canvas.paste(image.resize((new_width, new_height), Image.Resampling.BILINEAR), (left, top))
        pixels = np.asarray(canvas).copy()
        masks, classes = [], []
        for annotation in self.annotations.get(metadata["id"], []):
            mask = decode_mask(annotation["segmentation"])
            if mask.shape != (height, width):
                raise ValueError("Annotation mask dimensions differ from the dataset image")
            resized = Image.fromarray(mask.astype(np.uint8)).resize((new_width, new_height), Image.Resampling.NEAREST)
            canvas = np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
            canvas[top:top + new_height, left:left + new_width] = np.asarray(resized)
            if canvas.any():
                masks.append(canvas)
                classes.append(self.categories[annotation["category_id"]])
        masks = np.asarray(masks, dtype=np.uint8).reshape(-1, self.imgsz, self.imgsz)
        if self.augment and np.random.random() < self.fliplr:
            pixels, masks = pixels[:, ::-1], masks[:, :, ::-1]
        if self.augment and np.random.random() < self.flipud:
            pixels, masks = pixels[::-1], masks[:, ::-1]
        boxes = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        return {
            "img": torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))),
            "masks": torch.from_numpy(np.ascontiguousarray(masks)),
            "cls": torch.tensor(classes, dtype=torch.float32).reshape(-1, 1),
            "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4) / self.imgsz,
            "batch_idx": torch.zeros(len(classes)), "im_file": self.im_files[index],
            "ori_shape": (height, width), "resized_shape": (self.imgsz, self.imgsz),
            "ratio_pad": ((new_height / height, new_width / width), (left, top)),
        }


class MaskValidator(SegmentationValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.overlap = torch.zeros(2, dtype=torch.int64, device=self.device)

    def _process_batch(self, preds, batch):
        self.overlap += mask_overlap(
            {"masks": preds["masks"], "labels": preds["cls"], "scores": preds["conf"]},
            {"masks": batch["masks"], "labels": batch["cls"]}, self.mask_threshold,
        )
        return super()._process_batch(preds, batch)

    def gather_stats(self):
        super().gather_stats()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.overlap)

    def get_stats(self):
        stats = super().get_stats()
        stats["val/mask_iou"] = float(self.overlap[0] / self.overlap[1].clamp_min(1))
        stats["fitness"] = stats["val/mask_iou"]
        return stats


class MaskSegmentationTrainer(SegmentationTrainer):
    """Train against bitmap targets and keep only the best EMA weights."""

    def get_dataset(self):
        self.collectra = json.loads(Path(self.args.data).read_text())
        directory = Path(self.collectra["dataset"])
        annotations = json.loads((directory / "train/_annotations.coco.json").read_text())
        names = {index: item["name"] for index, item in enumerate(sorted(annotations["categories"], key=lambda item: item["id"]))}
        return {"train": str(directory / "train"),
                "val": str(directory / "valid"),
                "names": names, "nc": len(names), "channels": 3, "path": directory}

    def build_dataset(self, img_path, mode="train", batch=None):
        return BitmapDataset(img_path, self.args.imgsz, augment=mode == "train",
                             fliplr=self.args.fliplr, flipud=self.args.flipud)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "val" and not self.collectra["has_validation"]:
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            return DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def get_validator(self):
        validator = MaskValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
        validator.mask_threshold = self.collectra["threshold"]
        return validator

    def validate(self):
        if not self.collectra["has_validation"]:
            loss = torch.tensor(sum(float(value) for value in self.tloss.values()), device=self.device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(loss)
                loss /= torch.distributed.get_world_size()
            fitness = -float(loss)
            self.best_fitness = fitness if self.best_fitness is None else max(self.best_fitness, fitness)
            return {}, fitness
        return super().validate()

    def save_model(self):
        if not math.isfinite(float(self.fitness)) or self.fitness != self.best_fitness:
            return False
        model = deepcopy(self.ema.ema).float().cpu().to(memory_format=torch.contiguous_format)
        if hasattr(model, "criterion"):
            model.criterion = None
        self.wdir.mkdir(parents=True, exist_ok=True)
        temporary = self.best.with_suffix(".tmp")
        torch.save({"model": model, "epoch": self.epoch, "train_args": vars(self.args),
                    "version": __version__, "optimizer": None}, temporary)
        temporary.replace(self.best)
        (self.save_dir / "best.json").write_text(json.dumps({
            "epoch": self.epoch + 1,
            "metric": "val/mask_iou" if self.collectra["has_validation"] else "train/loss",
            "value": float(self.fitness) if self.collectra["has_validation"] else -float(self.fitness),
        }))
        return True

    def save_metrics(self, metrics):
        super().save_metrics(metrics)
        if not hasattr(self, "reporter"):
            self.reporter = EpochReporter(self.save_dir, wandb=self.collectra["wandb"],
                                          project=self.collectra["project"], run_name=self.collectra["run_name"],
                                          config=vars(self.args))
        self.reporter.record(self.epoch + 1, "", metrics)

    def final_eval(self):
        # No last checkpoint or second validation pass; the selected weights
        # and their epoch's validation score already agree.
        if hasattr(self, "reporter"):
            self.reporter.close()
            self.reporter.run = None


def launch(config_path):
    from collectra.tasks.machine_learning.yolo import YOLOTask, get_yolo_class

    config = json.loads(Path(config_path).read_text())
    model = get_yolo_class()(config["model"])
    if model.task != "segment":
        raise ValueError("ImageSegmentationYOLO requires a YOLO segmentation model (-seg)")
    (Path(config["params"]["project"]) / config["params"]["name"] / "weights" / "best.pt").unlink(missing_ok=True)
    with YOLOTask("train", model=model)._wandb_logging(False):
        try:
            model.train(trainer=MaskSegmentationTrainer, **config["params"])
        finally:
            reporter = getattr(getattr(model, "trainer", None), "reporter", None)
            if reporter is not None and reporter.run is not None:
                reporter.close()
