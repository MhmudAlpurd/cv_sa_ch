#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train, validate, evaluate, and run inference with Mas   k2Former on Oxford-IIIT Pet.

The program performs the full pipeline:
1. Downloads Oxford-IIIT Pet automatically.
2. Downloads a pretrained Mask2Former checkpoint automatically.
3. Splits trainval into training and validation sets.
4. Applies paired image/mask augmentation.
5. Fine-tunes Mask2Former for three semantic classes:
       0 = pet
       1 = background
       2 = border
6. Prints progress in the terminal and writes the same logs to a file.
7. Saves the best model according to validation mIoU.
8. Evaluates the best model on the official test set.
9. Saves training curves, confusion matrix, augmentation examples,
   and sample predictions.
10. Can later predict a single custom image.

Example commands
----------------
Quick pipeline test:
    python mask2former_oxford_pet.py --mode train --quick-test

Full training:
    python mask2former_oxford_pet.py --mode train --epochs 15 --image-size 384

Predict one image:
    python mask2former_oxford_pet.py \
        --mode predict \
        --image ./my_pet.jpg \
        --checkpoint ./outputs_mask2former_pet/best_model
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

# Makes plotting work on servers and terminal-only systems.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import ColorJitter, InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)


ID2LABEL = {
    0: "pet",
    1: "background",
    2: "border",
}
LABEL2ID = {name: index for index, name in ID2LABEL.items()}
NUM_CLASSES = len(ID2LABEL)


@dataclass
class TrainConfig:
    data_dir: str = "./data"
    output_dir: str = "./outputs_mask2former_pet"
    model_name: str = "facebook/mask2former-swin-tiny-ade-semantic"

    image_size: int = 384
    epochs: int = 15
    batch_size: int = 2
    accumulation_steps: int = 2

    learning_rate: float = 5e-5
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 0.05

    validation_ratio: float = 0.15
    num_workers: int = 4
    seed: int = 42

    use_amp: bool = True
    freeze_backbone_epochs: int = 0
    early_stopping_patience: int = 5
    gradient_clip_norm: float = 1.0

    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    visual_samples: int = 6
    log_every: int = 10


# ---------------------------------------------------------------------------
# Logging and reproducibility
# ---------------------------------------------------------------------------

def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    logger = logging.getLogger("mask2former_pet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Log file: %s", log_path.resolve())
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is preferred over maximum benchmark speed here.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    # Each DataLoader worker receives a deterministic but distinct seed.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Oxford-IIIT Pet mask handling
# ---------------------------------------------------------------------------

def convert_oxford_trimap(mask: Image.Image) -> Image.Image:
    """
    Oxford-IIIT Pet trimap:
        1 = pet
        2 = background
        3 = border

    Internal zero-based labels:
        0 = pet
        1 = background
        2 = border
    """
    source = np.asarray(mask, dtype=np.uint8)
    converted = np.full(source.shape, 255, dtype=np.uint8)

    converted[source == 1] = 0
    converted[source == 2] = 1
    converted[source == 3] = 2

    if np.any(converted == 255):
        raise ValueError(
            f"Unexpected Oxford-IIIT Pet trimap values: {np.unique(source).tolist()}"
        )

    return Image.fromarray(converted, mode="L")


# ---------------------------------------------------------------------------
# Paired augmentation: geometric operations must be identical for image/mask.
# ---------------------------------------------------------------------------

class JointTrainTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size
        self.color_jitter = ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.20,
            hue=0.03,
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        top, left, height, width = RandomResizedCrop.get_params(
            image,
            scale=(0.70, 1.00),
            ratio=(0.85, 1.15),
        )

        image = TF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resized_crop(
            mask,
            top,
            left,
            height,
            width,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.30:
            angle = random.uniform(-10.0, 10.0)
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=1,  # Newly introduced pixels are background.
            )

        image = self.color_jitter(image)
        return image, mask


class JointEvalTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        return image, mask


# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------

class OxfordPetMask2FormerDataset(Dataset):
    def __init__(
        self,
        base_dataset: OxfordIIITPet,
        indices: list[int],
        processor: Any,
        joint_transform: Any,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.processor = processor
        self.joint_transform = joint_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        real_index = self.indices[item]
        image, trimap = self.base_dataset[real_index]

        image = image.convert("RGB")
        mask = convert_oxford_trimap(trimap)
        image, mask = self.joint_transform(image, mask)

        semantic_mask = np.asarray(mask, dtype=np.int64)

        # The image processor converts the semantic map into the set-prediction
        # targets expected by Mask2Former: mask_labels and class_labels.
        encoded = self.processor(
            images=image,
            segmentation_maps=semantic_mask,
            return_tensors="pt",
        )

        sample: dict[str, Any] = {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "mask_labels": encoded["mask_labels"][0],
            "class_labels": encoded["class_labels"][0],
            "semantic_mask": torch.from_numpy(semantic_mask.copy()).long(),
            "display_image": np.asarray(image, dtype=np.uint8),
            "dataset_index": real_index,
        }

        if "pixel_mask" in encoded:
            sample["pixel_mask"] = encoded["pixel_mask"].squeeze(0)

        return sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        # The number of target masks can differ per image, so these stay lists.
        "mask_labels": [item["mask_labels"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
        "semantic_masks": torch.stack([item["semantic_mask"] for item in batch]),
        "display_images": [item["display_image"] for item in batch],
        "dataset_indices": [item["dataset_index"] for item in batch],
    }

    if "pixel_mask" in batch[0]:
        result["pixel_mask"] = torch.stack(
            [item["pixel_mask"] for item in batch]
        )

    return result


def truncate_indices(indices: list[int], maximum: int | None) -> list[int]:
    if maximum is None:
        return indices
    return indices[: min(maximum, len(indices))]


def build_dataloaders(
    processor: Any,
    cfg: TrainConfig,
    logger: logging.Logger,
) -> tuple[DataLoader, DataLoader, DataLoader, Dataset, Dataset]:
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing Oxford-IIIT Pet dataset in %s", data_dir.resolve())
    logger.info("If missing, torchvision will download the dataset automatically.")

    trainval_base = OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="segmentation",
        download=True,
    )
    test_base = OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types="segmentation",
        download=True,
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    permutation = torch.randperm(
        len(trainval_base),
        generator=generator,
    ).tolist()

    val_count = max(1, int(len(permutation) * cfg.validation_ratio))
    val_indices = permutation[:val_count]
    train_indices = permutation[val_count:]
    test_indices = list(range(len(test_base)))

    train_indices = truncate_indices(train_indices, cfg.max_train_samples)
    val_indices = truncate_indices(val_indices, cfg.max_val_samples)
    test_indices = truncate_indices(test_indices, cfg.max_test_samples)

    train_dataset = OxfordPetMask2FormerDataset(
        trainval_base,
        train_indices,
        processor,
        JointTrainTransform(cfg.image_size),
    )
    val_dataset = OxfordPetMask2FormerDataset(
        trainval_base,
        val_indices,
        processor,
        JointEvalTransform(cfg.image_size),
    )
    test_dataset = OxfordPetMask2FormerDataset(
        test_base,
        test_indices,
        processor,
        JointEvalTransform(cfg.image_size),
    )

    common_loader_args = {
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
        "collate_fn": collate_fn,
        "worker_init_fn": worker_init_fn,
    }

    train_generator = torch.Generator().manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=train_generator,
        drop_last=False,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )

    logger.info(
        "Dataset sizes | train=%d | validation=%d | test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    return train_loader, val_loader, test_loader, train_dataset, test_dataset


# ---------------------------------------------------------------------------
# Model, optimizer, and AMP
# ---------------------------------------------------------------------------

def build_model(
    cfg: TrainConfig,
    logger: logging.Logger,
) -> tuple[Any, Mask2FormerForUniversalSegmentation]:
    logger.info("Loading image processor: %s", cfg.model_name)
    processor = AutoImageProcessor.from_pretrained(
        cfg.model_name,
        do_resize=False,
        do_reduce_labels=False,
        ignore_index=255,
    )

    logger.info("Loading pretrained Mask2Former checkpoint: %s", cfg.model_name)
    model_config = Mask2FormerConfig.from_pretrained(cfg.model_name)
    model_config.num_labels = NUM_CLASSES
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID
    model_config.ignore_value = 255

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg.model_name,
        config=model_config,
        ignore_mismatched_sizes=True,
    )

    logger.info(
        "Model loaded. The ADE20K classification head is replaced by a %d-class head.",
        NUM_CLASSES,
    )
    logger.info(
        "A warning about newly initialized or mismatched class-predictor weights is expected."
    )
    return processor, model


def is_backbone_parameter(name: str) -> bool:
    return "pixel_level_module.encoder" in name


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if is_backbone_parameter(name):
            parameter.requires_grad = trainable


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> AdamW:
    backbone_parameters: list[nn.Parameter] = []
    other_parameters: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_backbone_parameter(name):
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    return AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": cfg.backbone_learning_rate,
            },
            {
                "params": other_parameters,
                "lr": cfg.learning_rate,
            },
        ],
        weight_decay=cfg.weight_decay,
    )


def create_grad_scaler(enabled: bool) -> Any:
    # Supports both newer and older PyTorch APIs.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool) -> Any:
    if not enabled:
        return contextlib.nullcontext()

    try:
        return torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=True,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def move_targets_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "pixel_values": batch["pixel_values"].to(
            device,
            non_blocking=True,
        ),
        "mask_labels": [
            item.to(device, non_blocking=True)
            for item in batch["mask_labels"]
        ],
        "class_labels": [
            item.to(device, non_blocking=True)
            for item in batch["class_labels"]
        ],
    }

    if "pixel_mask" in batch:
        inputs["pixel_mask"] = batch["pixel_mask"].to(
            device,
            non_blocking=True,
        )

    return inputs


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    prediction = prediction.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)

    valid = (
        (target >= 0)
        & (target < NUM_CLASSES)
        & (prediction >= 0)
        & (prediction < NUM_CLASSES)
    )

    encoded = target[valid] * NUM_CLASSES + prediction[valid]
    counts = torch.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    )
    confusion_matrix += counts.reshape(NUM_CLASSES, NUM_CLASSES).cpu()


def calculate_metrics(confusion_matrix: torch.Tensor) -> dict[str, Any]:
    matrix = confusion_matrix.to(torch.float64)
    true_positive = torch.diag(matrix)
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive

    iou_denominator = true_positive + false_positive + false_negative
    iou = torch.where(
        iou_denominator > 0,
        true_positive / iou_denominator,
        torch.nan,
    )

    class_total = matrix.sum(dim=1)
    class_accuracy = torch.where(
        class_total > 0,
        true_positive / class_total,
        torch.nan,
    )

    total_pixels = matrix.sum()
    pixel_accuracy = (
        true_positive.sum() / total_pixels
        if total_pixels > 0
        else torch.tensor(float("nan"), dtype=torch.float64)
    )

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "mean_accuracy": float(torch.nanmean(class_accuracy)),
        "mean_iou": float(torch.nanmean(iou)),
        "class_iou": {
            ID2LABEL[index]: float(iou[index])
            for index in range(NUM_CLASSES)
        },
        "class_accuracy": {
            ID2LABEL[index]: float(class_accuracy[index])
            for index in range(NUM_CLASSES)
        },
    }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    logger: logging.Logger,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    amp_enabled = cfg.use_amp and device.type == "cuda"
    running_loss = 0.0
    batch_count = 0

    progress = tqdm(
        enumerate(loader, start=1),
        total=len(loader),
        desc=f"Train {epoch:02d}",
        dynamic_ncols=True,
        leave=True,
    )

    for step, batch in progress:
        model_inputs = move_targets_to_device(batch, device)

        with autocast_context(device, amp_enabled):
            outputs = model(**model_inputs)
            raw_loss = outputs.loss
            scaled_loss = raw_loss / cfg.accumulation_steps

        if not torch.isfinite(raw_loss):
            raise FloatingPointError(
                f"Non-finite loss detected at epoch={epoch}, step={step}: "
                f"{raw_loss.item()}"
            )

        scaler.scale(scaled_loss).backward()

        update_now = (
            step % cfg.accumulation_steps == 0
            or step == len(loader)
        )
        if update_now:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_loss = float(raw_loss.detach().cpu())
        running_loss += batch_loss
        batch_count += 1
        average_loss = running_loss / batch_count

        progress.set_postfix(
            loss=f"{batch_loss:.4f}",
            avg=f"{average_loss:.4f}",
            lr=f"{optimizer.param_groups[-1]['lr']:.2e}",
        )

        if step == 1 or step % cfg.log_every == 0 or step == len(loader):
            logger.info(
                "TRAIN | epoch=%d/%d | step=%d/%d | batch_loss=%.6f | "
                "average_loss=%.6f | lr=%.3e",
                epoch,
                cfg.epochs,
                step,
                len(loader),
                batch_loss,
                average_loss,
                optimizer.param_groups[-1]["lr"],
            )

    return running_loss / max(batch_count, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    phase: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    model.eval()

    amp_enabled = cfg.use_amp and device.type == "cuda"
    total_loss = 0.0
    batch_count = 0
    confusion_matrix = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64,
    )

    progress = tqdm(
        enumerate(loader, start=1),
        total=len(loader),
        desc=phase,
        dynamic_ncols=True,
        leave=True,
    )

    for step, batch in progress:
        model_inputs = move_targets_to_device(batch, device)

        with autocast_context(device, amp_enabled):
            outputs = model(**model_inputs)

        batch_loss = float(outputs.loss.detach().cpu())
        total_loss += batch_loss
        batch_count += 1

        batch_size = batch["pixel_values"].shape[0]
        target_sizes = [
            (int(batch["semantic_masks"][i].shape[-2]),
             int(batch["semantic_masks"][i].shape[-1]))
            for i in range(batch_size)
        ]

        predicted_maps = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=target_sizes,
        )

        for index, prediction in enumerate(predicted_maps):
            update_confusion_matrix(
                confusion_matrix,
                prediction.cpu(),
                batch["semantic_masks"][index].cpu(),
            )

        running_loss = total_loss / batch_count
        progress.set_postfix(loss=f"{running_loss:.4f}")

        if step == 1 or step % cfg.log_every == 0 or step == len(loader):
            logger.info(
                "%s | step=%d/%d | average_loss=%.6f",
                phase.upper(),
                step,
                len(loader),
                running_loss,
            )

    metrics = calculate_metrics(confusion_matrix)
    metrics["loss"] = total_loss / max(batch_count, 1)
    metrics["confusion_matrix"] = confusion_matrix.tolist()
    return metrics


# ---------------------------------------------------------------------------
# Saving models, tables, and visual outputs
# ---------------------------------------------------------------------------

MASK_PALETTE = np.array(
    [
        [220, 70, 70],   # pet
        [40, 40, 40],    # background
        [255, 220, 70],  # border
    ],
    dtype=np.uint8,
)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.min() < 0 or mask.max() >= NUM_CLASSES:
        raise ValueError(
            f"Mask contains labels outside [0, {NUM_CLASSES - 1}]."
        )
    return MASK_PALETTE[mask]


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    colored = colorize_mask(mask).astype(np.float32)
    image_float = image.astype(np.float32)
    overlay = (1.0 - alpha) * image_float + alpha * colored
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_checkpoint(
    model: nn.Module,
    processor: Any,
    checkpoint_dir: Path,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    processor.save_pretrained(checkpoint_dir)

    with (checkpoint_dir / "training_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "epoch": epoch,
                "metrics": metrics,
                "id2label": ID2LABEL,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_checkpoint(
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[Any, Mask2FormerForUniversalSegmentation]:
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint_dir
    ).to(device)
    model.eval()
    return processor, model


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_training_curves(
    history: list[dict[str, float]],
    output_dir: Path,
) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "training_history.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.plot(frame["epoch"], frame["train_loss"], marker="o", label="Train")
    plt.plot(frame["epoch"], frame["val_loss"], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(frame["epoch"], frame["val_miou"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.title("Validation mean IoU")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "miou_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(frame["epoch"], frame["val_pixel_accuracy"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel accuracy")
    plt.title("Validation pixel accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pixel_accuracy_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(frame["epoch"], frame["learning_rate"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning-rate schedule")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate_curve.png", dpi=180)
    plt.close()


def save_confusion_matrix(
    confusion_matrix: list[list[int]],
    output_path: Path,
) -> None:
    matrix = np.asarray(confusion_matrix, dtype=np.int64)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)
    plt.title("Test confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Ground-truth class")
    plt.xticks(range(NUM_CLASSES), [ID2LABEL[i] for i in range(NUM_CLASSES)])
    plt.yticks(range(NUM_CLASSES), [ID2LABEL[i] for i in range(NUM_CLASSES)])

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            plt.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_augmentation_examples(
    dataset: Dataset,
    output_path: Path,
    count: int = 4,
) -> None:
    count = min(count, len(dataset))
    figure, axes = plt.subplots(count, 2, figsize=(10, 4 * count))

    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(count):
        sample = dataset[row]
        axes[row, 0].imshow(sample["display_image"])
        axes[row, 0].set_title("Augmented image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(
            colorize_mask(sample["semantic_mask"].numpy())
        )
        axes[row, 1].set_title("Augmented mask")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


@torch.inference_mode()
def save_test_predictions(
    model: nn.Module,
    processor: Any,
    dataset: Dataset,
    device: torch.device,
    count: int,
    output_path: Path,
) -> None:
    model.eval()
    count = min(count, len(dataset))
    selected = np.linspace(
        0,
        len(dataset) - 1,
        num=count,
        dtype=int,
    )

    figure, axes = plt.subplots(count, 4, figsize=(16, 4 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, dataset_index in enumerate(selected):
        sample = dataset[int(dataset_index)]
        inputs: dict[str, Any] = {
            "pixel_values": sample["pixel_values"].unsqueeze(0).to(device)
        }

        if "pixel_mask" in sample:
            inputs["pixel_mask"] = (
                sample["pixel_mask"].unsqueeze(0).to(device)
            )

        outputs = model(**inputs)
        target_height, target_width = sample["semantic_mask"].shape[-2:]

        prediction = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(int(target_height), int(target_width))],
        )[0].cpu().numpy()

        image = sample["display_image"]
        ground_truth = sample["semantic_mask"].numpy()
        overlay = create_overlay(image, prediction)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Input")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(colorize_mask(ground_truth))
        axes[row, 1].set_title("Ground truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(colorize_mask(prediction))
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Overlay")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# Full train workflow
# ---------------------------------------------------------------------------

def run_training(cfg: TrainConfig) -> None:
    output_dir = Path(cfg.output_dir)
    logger = configure_logging(output_dir)
    set_seed(cfg.seed)

    logger.info("=" * 78)
    logger.info("Mask2Former Oxford-IIIT Pet training started")
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("PyTorch: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Selected device: %s", device)

    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "Initial CUDA memory allocated: %.2f MB",
            torch.cuda.memory_allocated() / (1024**2),
        )
    else:
        logger.warning(
            "CUDA was not detected. Training on CPU is possible but will be very slow."
        )

    save_json(asdict(cfg), output_dir / "run_config.json")
    logger.info("Configuration:\n%s", json.dumps(asdict(cfg), indent=2))

    processor, model = build_model(cfg, logger)
    model = model.to(device)

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        test_dataset,
    ) = build_dataloaders(processor, cfg, logger)

    logger.info("Saving paired augmentation examples.")
    save_augmentation_examples(
        train_dataset,
        output_dir / "augmentation_examples.png",
        count=4,
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=1e-7,
    )

    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = create_grad_scaler(amp_enabled)
    logger.info("Automatic mixed precision enabled: %s", amp_enabled)
    logger.info(
        "Effective batch size: %d",
        cfg.batch_size * cfg.accumulation_steps,
    )

    history: list[dict[str, float]] = []
    best_miou = -math.inf
    epochs_without_improvement = 0
    best_model_dir = output_dir / "best_model"
    started_at = time.time()

    for epoch in range(1, cfg.epochs + 1):
        logger.info("=" * 78)
        logger.info("Starting epoch %d/%d", epoch, cfg.epochs)

        freeze_backbone = epoch <= cfg.freeze_backbone_epochs
        set_backbone_trainable(model, trainable=not freeze_backbone)
        logger.info("Backbone frozen: %s", freeze_backbone)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            epoch,
            logger,
        )

        validation_metrics = evaluate(
            model,
            processor,
            val_loader,
            device,
            cfg,
            phase=f"Validation {epoch:02d}",
            logger=logger,
        )

        scheduler.step()

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(validation_metrics["loss"]),
            "val_miou": float(validation_metrics["mean_iou"]),
            "val_pixel_accuracy": float(
                validation_metrics["pixel_accuracy"]
            ),
            "val_mean_accuracy": float(
                validation_metrics["mean_accuracy"]
            ),
            "learning_rate": float(
                optimizer.param_groups[-1]["lr"]
            ),
        }
        history.append(row)
        save_training_curves(history, output_dir)

        logger.info(
            "EPOCH RESULT | epoch=%d | train_loss=%.6f | val_loss=%.6f | "
            "val_mIoU=%.6f | val_pixel_accuracy=%.6f | val_mean_accuracy=%.6f",
            epoch,
            train_loss,
            validation_metrics["loss"],
            validation_metrics["mean_iou"],
            validation_metrics["pixel_accuracy"],
            validation_metrics["mean_accuracy"],
        )

        for class_name, class_iou in validation_metrics["class_iou"].items():
            logger.info(
                "VALIDATION CLASS IoU | class=%s | IoU=%.6f",
                class_name,
                class_iou,
            )

        current_miou = float(validation_metrics["mean_iou"])
        if current_miou > best_miou:
            best_miou = current_miou
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                processor,
                best_model_dir,
                epoch,
                validation_metrics,
            )
            logger.info(
                "New best model saved | epoch=%d | validation mIoU=%.6f | path=%s",
                epoch,
                best_miou,
                best_model_dir.resolve(),
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "Validation mIoU did not improve for %d epoch(s).",
                epochs_without_improvement,
            )

        if torch.cuda.is_available():
            logger.info(
                "CUDA memory | allocated=%.2f MB | reserved=%.2f MB | "
                "max_allocated=%.2f MB",
                torch.cuda.memory_allocated() / (1024**2),
                torch.cuda.memory_reserved() / (1024**2),
                torch.cuda.max_memory_allocated() / (1024**2),
            )

        if epochs_without_improvement >= cfg.early_stopping_patience:
            logger.info(
                "Early stopping activated after %d epoch(s) without improvement.",
                epochs_without_improvement,
            )
            break

    total_minutes = (time.time() - started_at) / 60.0
    logger.info("Training completed in %.2f minutes.", total_minutes)

    if not best_model_dir.exists():
        raise RuntimeError(
            "No best checkpoint was created. Review training.log for errors."
        )

    logger.info("Loading best checkpoint for final test evaluation.")
    best_processor, best_model = load_checkpoint(best_model_dir, device)

    test_metrics = evaluate(
        best_model,
        best_processor,
        test_loader,
        device,
        cfg,
        phase="Final test",
        logger=logger,
    )

    save_json(test_metrics, output_dir / "test_metrics.json")
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        output_dir / "test_confusion_matrix.png",
    )

    logger.info(
        "FINAL TEST | loss=%.6f | mIoU=%.6f | pixel_accuracy=%.6f | "
        "mean_accuracy=%.6f",
        test_metrics["loss"],
        test_metrics["mean_iou"],
        test_metrics["pixel_accuracy"],
        test_metrics["mean_accuracy"],
    )
    for class_name, class_iou in test_metrics["class_iou"].items():
        logger.info(
            "TEST CLASS IoU | class=%s | IoU=%.6f",
            class_name,
            class_iou,
        )

    logger.info("Saving sample test predictions.")
    save_test_predictions(
        best_model,
        best_processor,
        test_dataset,
        device,
        cfg.visual_samples,
        output_dir / "test_sample_predictions.png",
    )

    logger.info("=" * 78)
    logger.info("All stages completed successfully.")
    logger.info("Best model: %s", best_model_dir.resolve())
    logger.info("Test metrics: %s", (output_dir / "test_metrics.json").resolve())
    logger.info("Full log: %s", (output_dir / "training.log").resolve())
    logger.info("Visual outputs are in: %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_single_image(
    image_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_path.parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading checkpoint from %s", checkpoint_dir.resolve())
    processor, model = load_checkpoint(checkpoint_dir, device)

    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size

    # The saved processor contains the preprocessing configuration used by
    # the checkpoint. It can process arbitrary image dimensions for inference.
    inputs = processor(images=image, return_tensors="pt")
    model_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    outputs = model(**model_inputs)
    prediction = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[(original_height, original_width)],
    )[0].cpu().numpy()

    image_array = np.asarray(image, dtype=np.uint8)
    colored_mask = colorize_mask(prediction)
    overlay = create_overlay(image_array, prediction)

    figure, axes = plt.subplots(1, 3, figsize=(17, 6))
    axes[0].imshow(image_array)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(colored_mask)
    axes[1].set_title("Predicted mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    raw_mask_path = output_path.with_name(
        f"{output_path.stem}_class_ids.png"
    )
    Image.fromarray(prediction.astype(np.uint8), mode="L").save(raw_mask_path)

    logger.info("Prediction figure saved to %s", output_path.resolve())
    logger.info("Raw class-ID mask saved to %s", raw_mask_path.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask2Former training on Oxford-IIIT Pet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="Run full training/evaluation or predict one image.",
    )

    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--output-dir",
        default="./outputs_mask2former_pet",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/mask2former-swin-tiny-ade-semantic",
    )

    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
    )
    parser.add_argument("--visual-samples", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run one epoch on a small subset to verify the full pipeline.",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image path for predict mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs_mask2former_pet/best_model",
        help="Saved checkpoint directory for predict mode.",
    )
    parser.add_argument(
        "--prediction-output",
        type=str,
        default="./outputs_mask2former_pet/custom_prediction.png",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if args.image_size % 32 != 0:
        raise ValueError(
            "--image-size should be divisible by 32, for example 320, 384, or 512."
        )
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation-steps must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1.")

    if args.mode == "predict" and not args.image:
        raise ValueError("--image is required when --mode predict is used.")


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        use_amp=not args.no_amp,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        early_stopping_patience=args.early_stopping_patience,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        visual_samples=args.visual_samples,
        log_every=args.log_every,
    )

    if args.quick_test:
        cfg.image_size = 320
        cfg.epochs = 1
        cfg.batch_size = 1
        cfg.accumulation_steps = 1
        cfg.max_train_samples = 20
        cfg.max_val_samples = 10
        cfg.max_test_samples = 10
        cfg.visual_samples = 4
        cfg.early_stopping_patience = 1
        cfg.log_every = 1

    return cfg


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.mode == "train":
        cfg = config_from_args(args)
        run_training(cfg)
    else:
        predict_single_image(
            image_path=Path(args.image),
            checkpoint_dir=Path(args.checkpoint),
            output_path=Path(args.prediction_output),
        )


if __name__ == "__main__":
    main()
