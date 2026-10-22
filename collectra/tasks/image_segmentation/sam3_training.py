"""Adapt Meta's installed image-training template for mask-supervised COCO data."""

import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

import yaml


def training_config(dataset, output, checkpoint, *, epochs, batch, num_gpus,
                    wandb=False, run_name=None, project=None, confidence_threshold=0.5):
    from .sam3 import _sam3

    package = Path(_sam3().__file__).parent
    template = (
        package / "train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml"
    )
    if not template.is_file():
        raise FileNotFoundError(
            f"Install SAM3's training sources; template missing: {template}"
        )
    config = yaml.safe_load(template.read_text())
    config["paths"].update(
        roboflow_vl_100_root=str(dataset),
        experiment_log_dir=str(output / "training"),
        bpe_path=str(package / "assets/bpe_simple_vocab_16e6.txt.gz"),
    )
    settings = config["roboflow_train"]
    settings.update(supercategory="collectra", num_images=None)
    losses = settings["loss"]["loss_fns_find"]
    losses.append(
        {
            "_target_": "sam3.train.loss.loss_fns.Masks",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "weight_dict": {"loss_mask": 200.0, "loss_dice": 10.0},
            "compute_aux": False,
        }
    )
    # Validation masks must be decoded too; Meta's detection template only
    # decodes them in the training transform sequence.
    settings["val_transforms"][0]["transforms"].insert(
        0,
        {
            "_target_": "sam3.train.transforms.segmentation.DecodeRle",
        },
    )
    config["scratch"].update(
        enable_segmentation=True,
        train_batch_size=batch,
        num_train_workers=0,
        num_val_workers=0,
    )
    trainer = config["trainer"]
    trainer.update(
        _target_="collectra.tasks.image_segmentation.sam3_metrics.create_trainer",
        max_epochs=epochs, skip_saving_ckpts=True, val_epoch_freq=1, skip_first_val=False,
        collectra_output=str(output), collectra_wandb=wandb,
        collectra_run_name=run_name,
        collectra_project=project,
    )
    trainer["model"] = {
        "_target_": "collectra.tasks.image_segmentation.sam3_training.build_training_model",
        "checkpoint": str(checkpoint),
        "bpe_path": config["paths"]["bpe_path"],
    }
    for split, folder in (("train", "train"), ("val", "valid")):
        data = trainer["data"][split]
        data["dataset"].update(
            img_folder=str(dataset / folder),
            ann_file=str(dataset / folder / "_annotations.coco.json"),
            load_segmentation=True,
        )
        data["drop_last"] = False
    # After every epoch the training images are also scored like validation
    # (no augmentation, eval mode), so train/mask_iou is comparable to val/mask_iou.
    train_eval = copy.deepcopy(trainer["data"]["val"])
    train_eval["dataset"].update(
        img_folder=str(dataset / "train"),
        ann_file=str(dataset / "train/_annotations.coco.json"),
    )
    trainer.update(
        collectra_train_eval=train_eval,
        collectra_mask_threshold=confidence_threshold,
    )
    import json

    valid = json.loads((dataset / "valid/_annotations.coco.json").read_text())
    if valid["images"]:
        evaluator = trainer["meters"]["val"]["roboflow100"]["detection"][
            "pred_file_evaluators"
        ][0]
        evaluator["gt_path"] = str(dataset / "valid/_annotations.coco.json")
        trainer["meters"]["val"]["roboflow100"]["mask_iou"] = {
            "_target_": "collectra.tasks.image_segmentation.sam3_metrics.MaskIoU",
            "confidence_threshold": confidence_threshold,
        }
    else:
        # Training without held-out images must not silently evaluate on training data.
        trainer["data"].pop("val", None)
        trainer["meters"] = {}
        trainer["mode"] = "train_only"
    config["launcher"].update(gpus_per_node=num_gpus, num_nodes=1)
    config["submitit"].update(use_cluster=False, job_array=None)
    return config


def load_image_state(model, state):
    """Load image weights, adapting SAM3.1's three-level detection neck."""
    if any(key.startswith("detector.") for key in state):
        state = {
            key.removeprefix("detector."): value
            for key, value in state.items()
            if key.startswith("detector.")
        }
    prefix = "backbone.vision_backbone."
    tracking = (prefix + "interactive_convs.", prefix + "propagation_convs.")
    if any(key.startswith(tracking) for key in state):
        state = {key: value for key, value in state.items() if not key.startswith(tracking)}
        # SAM3.1 stores three detection scales. The image builder creates four,
        # then SAM3VLBackbone(scalp=1) discards the fourth. Initialize only that
        # unused scale from the equivalent stride-1 convolutions; never tolerate
        # missing weights in any feature level actually consumed by the model.
        neck = model.backbone.vision_backbone
        if model.backbone.scalp != 1 or tuple(neck.scale_factors) != (4.0, 2.0, 1.0, 0.5):
            raise ValueError("Unsupported SAM3 image backbone for SAM3.1 checkpoint")
        for suffix in ("conv_1x1.weight", "conv_1x1.bias", "conv_3x3.weight", "conv_3x3.bias"):
            target = prefix + "convs.3." + suffix
            if target not in state:
                state[target] = state[prefix + "convs.2." + suffix].clone()
    model.load_state_dict(state, strict=True)


def _training_mlp_forward(self, x):
    """SAM3 ViT MLP with ordinary autograd-capable PyTorch operations."""
    x = self.act(self.fc1(x))
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    return self.drop2(x)


def enable_training_mlps(model):
    from sam3.model.vitdet import Mlp

    # Upstream Mlp.forward uses addmm_act, an inference-only fused kernel.
    # Bind only this model's MLPs; keep parameter names/checkpoints intact and
    # leave SAM3's inference implementation untouched.
    for module in model.modules():
        if isinstance(module, Mlp):
            module.forward = MethodType(_training_mlp_forward, module)


def freeze_unused_semantic_head(model):
    """Instance-mask losses do not supervise the separate semantic output."""
    segmentation = getattr(model, "segmentation_head", None)
    semantic = getattr(segmentation, "semantic_seg_head", None)
    if semantic is not None:
        # DDP sees this head in forward's outputs, so find_unused_parameters
        # cannot detect that the loss never consumes it. Freeze its parameters
        # before DDP/optimizer construction, preserving checkpoint compatibility.
        semantic.requires_grad_(False)


def build_training_model(checkpoint, bpe_path):
    import torch
    from sam3.model_builder import build_sam3_image_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        load_from_HF=False,
        enable_segmentation=True,
        device="cpu",
        eval_mode=False,
    )
    load_image_state(model, state)
    enable_training_mlps(model)
    freeze_unused_semantic_head(model)
    return model


def launch(config_path):
    from hydra import initialize_config_dir
    from sam3.train.train import main, register_omegaconf_resolvers

    config_path = Path(config_path).resolve()
    register_omegaconf_resolvers()
    with initialize_config_dir(str(config_path.parent), version_base="1.2"):
        main(
            SimpleNamespace(
                config=config_path.name, use_cluster=False, num_gpus=None, num_nodes=1
            )
        )
