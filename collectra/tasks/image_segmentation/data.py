"""Shared image/mask geometry and lossless COCO preparation for segmentation backends."""

from __future__ import annotations

import json
import time
from pydantic import BaseModel

from collectra.commons import PositionalFields
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from collectra.types.images import Image, ImageSegmentation


def training_progress(message):
    print(f"[Segmentation] {message}", flush=True)


def _bounds(image: Image):
    return tuple(round(value) for value in image.display_bounds())


def mask_in_view(image: Image, mask: PILImage.Image):
    """Apply the same crop, region and rotation as image.pil()."""
    if mask.size != (image.raw_width, image.raw_height):
        raise ValueError("Annotation mask dimensions differ from the source image")
    if isinstance(image, ImageSegmentation):
        pixels = (np.asarray(mask) != 0) & (np.asarray(image._mask_image()) != 0)
        mask = PILImage.fromarray(pixels.astype("uint8"))
    return mask.crop(_bounds(image)).rotate(image.orientation.to_degree(), expand=True)


def mask_to_source(image: Image, mask: np.ndarray):
    """Invert the displayed image's rotation/crop into source-sized 0/1 pixels."""
    local = PILImage.fromarray(mask.astype("uint8"))
    local = local.rotate(-image.orientation.to_degree(), expand=True)
    left, top, right, bottom = _bounds(image)
    if local.size != (right - left, bottom - top):
        raise ValueError("Predicted mask dimensions differ from the input image view")
    source = PILImage.new("L", (image.raw_width, image.raw_height))
    source.paste(local, (left, top))
    pixels = np.asarray(source) != 0
    if isinstance(image, ImageSegmentation):
        pixels &= np.asarray(image._mask_image()) != 0
    return pixels.astype("uint8")


def encode_rle(mask: np.ndarray):
    """Lossless COCO uncompressed RLE, including holes and disconnected pixels."""
    values = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    boundaries = np.concatenate(
        ([0], np.flatnonzero(values[1:] != values[:-1]) + 1, [len(values)])
    )
    counts = np.diff(boundaries).tolist()
    if len(values) and values[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


class SegmentationSample(PositionalFields, BaseModel):
    image: Image
    instances: list[ImageSegmentation]


def prepare_segmentation_inputs(
    processed_inputs, processed_parents, input_maps, parent_input_maps
):
    samples = []
    for filename, instances in input_maps.items():
        parents = {
            image.id: image
            for image in parent_input_maps.get(filename, [])
            if isinstance(image, Image)
        }
        grouped = {}
        for instance in instances:
            if not isinstance(instance, ImageSegmentation):
                raise TypeError(
                    "Segmentation training targets must be ImageSegmentation instances"
                )
            ids = (
                [instance.parents]
                if isinstance(instance.parents, str)
                else instance.parents
            )
            candidates = [parents[key] for key in ids if key in parents]
            if len(candidates) != 1:
                raise ValueError(
                    f"Mask {instance.id!r} must identify exactly one input image parent"
                )
            parent = candidates[0]
            if parent.get_path().resolve() != instance.get_path().resolve():
                raise ValueError(
                    f"Mask {instance.id!r} refers to a different source image"
                )
            grouped.setdefault(
                parent.id, SegmentationSample(parent, [])
            ).instances.append(instance)
        samples.extend(grouped.values())
    return samples


def _sample_partition(sample):
    partitions = {item.partition for item in sample.instances if item.partition}
    if sample.image.partition:
        partitions.add(sample.image.partition)
    if len(partitions) > 1:
        raise ValueError("Instances of one image cannot belong to different partitions")
    return next(iter(partitions), "")


def export_dataset(
    samples, directory: Path, *, prompts=None, validation="", exclude="", max_items=None,
    progress=None,
):
    progress = progress or training_progress
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    started = time.monotonic()
    directory = Path(directory).resolve()
    selected = []
    for sample in samples:
        partition = _sample_partition(sample)
        if exclude and partition == exclude:
            continue
        selected.append(
            (sample, "valid" if validation and partition == validation else "train")
        )
        if max_items is not None and len(selected) >= max_items:
            break
    labels = sorted({item.name for sample, _ in selected for item in sample.instances})
    if not labels or not any(split == "train" for _, split in selected):
        raise ValueError("No labelled training instances were found")
    categories = [
        {"id": index + 1, "name": (prompts or {}).get(label, label.replace("_", " "))}
        for index, label in enumerate(labels)
    ]
    label_ids = {label: index + 1 for index, label in enumerate(labels)}
    progress(
        f"Exporting {len(selected)} images and "
        f"{sum(len(sample.instances) for sample, _ in selected)} masks to {directory} "
        f"({sum(split == 'train' for _, split in selected)} train, "
        f"{sum(split == 'valid' for _, split in selected)} validation images)"
    )
    datasets = {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in ("train", "valid")
    }
    for split in datasets:
        (directory / split).mkdir(parents=True, exist_ok=True)
    skipped_masks, skipped_images = [], []
    image_id = 0
    for index, (sample, split) in enumerate(selected, 1):
        progress(
            f"Export {index}/{len(selected)} [{split}]: {sample.image.id} "
            f"({len(sample.instances)} masks; {time.monotonic() - started:.1f}s elapsed)"
        )
        image = sample.image.pil().convert("RGB")
        annotations = []
        for mask_index, instance in enumerate(sample.instances, 1):
            mask = np.asarray(mask_in_view(sample.image, instance._mask_image())) != 0
            ys, xs = np.nonzero(mask)
            if not len(xs):
                # Usually an annotation lying outside its parent region; skip it
                # rather than abandon the whole export.
                skipped_masks.append(instance.id)
                progress(
                    f"  WARNING: skipping mask {instance.id!r}: no pixels inside "
                    f"its input image {sample.image.id!r}"
                )
                continue
            if mask.shape != (image.height, image.width):
                raise ValueError(
                    "Image subclass crop differs from its declared display bounds"
                )
            annotations.append(
                {
                    "category_id": label_ids[instance.name],
                    "iscrowd": 0,
                    "bbox": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    ],
                    "area": int(mask.sum()),
                    "segmentation": encode_rle(mask),
                }
            )
            if mask_index % 25 == 0:
                progress(
                    f"  Image {index}/{len(selected)}: encoded mask "
                    f"{mask_index}/{len(sample.instances)}"
                )
        if sample.instances and not annotations:
            # Without its masks the image would wrongly teach that it has no instances.
            skipped_images.append(sample.image.id)
            progress(
                f"  WARNING: skipping image {sample.image.id!r}: none of its "
                f"{len(sample.instances)} masks lie inside it"
            )
            continue
        image_id += 1
        filename = f"{image_id:08d}.png"
        image.save(directory / split / filename)
        data = datasets[split]
        data["images"].append(
            {
                "id": image_id,
                "file_name": filename,
                "width": image.width,
                "height": image.height,
            }
        )
        for annotation in annotations:
            data["annotations"].append(
                {"id": len(data["annotations"]) + 1, "image_id": image_id, **annotation}
            )
    if skipped_masks:
        progress(
            f"WARNING: skipped {len(skipped_masks)} masks with no pixels inside their "
            f"input images and {len(skipped_images)} images left without masks"
        )
    if not datasets["train"]["images"]:
        raise ValueError("No labelled training instances remain after skipping empty masks")
    for split, data in datasets.items():
        progress(
            f"Writing {split} annotations: {len(data['images'])} images, "
            f"{len(data['annotations'])} masks"
        )
        (directory / split / "_annotations.coco.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
    progress(f"Dataset export complete ({time.monotonic() - started:.1f}s): {directory}")
    return directory


class SegmentationTrainingData:
    """Backend-independent annotated instance collection and export."""

    def _samples(self, inputs, max_items=None, exclude=""):
        from collectra.tasks.machine_learning.training import (
            _training_files,
            _prepare_data,
        )

        if self.pipeline is None:
            raise RuntimeError(
                "Attach the task to a pipeline before preparing training data"
            )
        started = time.monotonic()
        self._progress("Discovering training result files (CPU/I/O preparation)")
        self.pipeline._ensure_workflow_connected()
        node = self.pipeline.node_manager.resolve_node(self.name)
        files = []
        for value in inputs:
            path = Path(value).expanduser().resolve()
            self._progress(f"Scanning {path}")
            if path.name == "results.yaml" and path.is_file():
                path = path.parent
            if path.is_dir() and (path / "results.yaml").is_file():
                files.append(path)
            else:
                files.extend(_training_files(self, [str(path)]))
        samples = []
        files = list(dict.fromkeys(files))
        self._progress(f"Found {len(files)} result files; loading images and validating masks")
        for index, path in enumerate(files, 1):
            self._progress(
                f"Load {index}/{len(files)}: {path} "
                f"({time.monotonic() - started:.1f}s elapsed)"
            )
            # Read each results directory independently. Different datasets can
            # have the same page basename and even the same artefact ids.
            targets, target_maps = _prepare_data(
                self, [path], self.pipeline.node_manager.get_children_artefact(node)
            )
            parents, parent_maps = _prepare_data(
                self, [path], self.pipeline.node_manager.get_ancestor_artefacts(node)
            )
            for sample in prepare_segmentation_inputs(targets, parents, target_maps, parent_maps):
                if max_items is not None and exclude and _sample_partition(sample) == exclude:
                    continue
                samples.append(sample)
                if max_items is not None and len(samples) >= max_items:
                    break
            if max_items is not None and len(samples) >= max_items:
                self._progress(f"Reached max_items={max_items}; stopping source loading")
                break
        self._progress(
            f"Loaded {len(samples)} labelled images and "
            f"{sum(len(sample.instances) for sample in samples)} masks "
            f"({time.monotonic() - started:.1f}s)"
        )
        return samples

    def prepare_training_data(
        self, inputs: list[str], output: Path, validation: str = "", exclude: str = "",
        max_items: int | None = None,
    ) -> Path:
        """Export annotated masks and input images as lossless COCO datasets."""
        samples = self._samples(inputs, max_items=max_items, exclude=exclude)
        return export_dataset(
            samples,
            output,
            prompts=self._label_prompts(samples),
            progress=self._progress,
            validation=validation,
            exclude=exclude,
            max_items=max_items,
        )

    def _label_prompts(self, samples):
        return self._prompts()

