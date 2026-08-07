#!/usr/bin/env python3
"""
End-to-end Mask R-CNN instance segmentation on Penn-Fudan Pedestrian.

Features
--------
1. Downloads and extracts Penn-Fudan automatically.
2. Builds train/validation/test splits.
3. Fine-tunes torchvision Mask R-CNN ResNet50-FPN.
4. Uses low-resource defaults: batch size 1, resized inputs, optional AMP.
5. Saves checkpoints, CSV history, loss curves, JSON metrics.
6. Evaluates mask IoU, Dice, Precision and Recall at IoU=0.5.
7. Saves predictions on several test images.

Example
-------
python train_mask_rcnn.py
python train_mask_rcnn.py --epochs 3 --max-samples 80
python train_mask_rcnn.py --device cpu --epochs 1
python train_mask_rcnn.py --predict-only --checkpoint outputs/checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as TF
from tqdm import tqdm


DATASET_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
CLASS_NAMES = {0: "background", 1: "person"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Mask R-CNN on Penn-Fudan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--lr-step-size", type=int, default=4)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--min-size", type=int, default=400)
    parser.add_argument("--max-size", type=int, default=600)
    parser.add_argument(
        "--trainable-backbone-layers",
        type=int,
        default=2,
        choices=range(0, 6),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision when CUDA is active.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use only this many images; useful for a quick educational run.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-predictions", type=int, default=6)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used by --resume or --predict-only.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load COCO pretrained weights. Not recommended.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"

    logger = logging.getLogger("mask_rcnn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = 100.0 * downloaded / total_size
        print(
            f"\rDownloading: {downloaded / 1024**2:.1f}/"
            f"{total_size / 1024**2:.1f} MB ({percent:.1f}%)",
            end="",
            flush=True,
        )

    urllib.request.urlretrieve(url, destination, reporthook=reporthook)
    print()


def ensure_dataset(data_dir: Path, logger: logging.Logger) -> Path:
    dataset_root = data_dir / "PennFudanPed"
    image_dir = dataset_root / "PNGImages"
    mask_dir = dataset_root / "PedMasks"

    if image_dir.is_dir() and mask_dir.is_dir():
        logger.info("Dataset already exists at %s", dataset_root)
        return dataset_root

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "PennFudanPed.zip"

    if not zip_path.exists():
        logger.info("Downloading Penn-Fudan from %s", DATASET_URL)
        try:
            download_with_progress(DATASET_URL, zip_path)
        except Exception as exc:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Dataset download failed. Check the internet connection or "
                f"download manually from {DATASET_URL} and place the ZIP at "
                f"{zip_path}."
            ) from exc

    logger.info("Extracting %s", zip_path)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(data_dir)
    except zipfile.BadZipFile as exc:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded ZIP is invalid. Delete it and run the script again."
        ) from exc

    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise RuntimeError(
            f"Expected folders were not found after extraction: {dataset_root}"
        )

    logger.info("Dataset is ready at %s", dataset_root)
    return dataset_root


class DetectionTransform:
    def __init__(self, train: bool) -> None:
        self.train = train

    def __call__(
        self,
        image: Image.Image,
        target: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        image_tensor = TF.convert_image_dtype(TF.pil_to_tensor(image), torch.float32)

        if self.train and random.random() < 0.5:
            image_tensor = TF.hflip(image_tensor)
            width = image_tensor.shape[-1]

            boxes = target["boxes"].clone()
            old_xmin = boxes[:, 0].clone()
            old_xmax = boxes[:, 2].clone()
            boxes[:, 0] = width - old_xmax
            boxes[:, 2] = width - old_xmin
            target["boxes"] = boxes
            target["masks"] = torch.flip(target["masks"], dims=[2])

        return image_tensor, target


class PennFudanDataset(Dataset):
    def __init__(self, root: Path, train: bool) -> None:
        self.root = root
        self.transforms = DetectionTransform(train=train)
        self.images = sorted((root / "PNGImages").glob("*.png"))
        self.masks = sorted((root / "PedMasks").glob("*.png"))

        if not self.images or not self.masks:
            raise RuntimeError(f"No images or masks found under {root}")
        if len(self.images) != len(self.masks):
            raise RuntimeError(
                f"Image/mask count mismatch: {len(self.images)} vs {len(self.masks)}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        image = Image.open(self.images[index]).convert("RGB")
        mask_image = Image.open(self.masks[index])
        mask_np = np.asarray(mask_image, dtype=np.int64)

        object_ids = np.unique(mask_np)
        object_ids = object_ids[object_ids != 0]

        masks_np = mask_np == object_ids[:, None, None]
        masks = torch.as_tensor(masks_np, dtype=torch.uint8)

        boxes_list: List[List[float]] = []
        valid_masks: List[Tensor] = []

        for mask in masks:
            positions = torch.where(mask > 0)
            if positions[0].numel() == 0:
                continue
            ymin = int(positions[0].min())
            ymax = int(positions[0].max()) + 1
            xmin = int(positions[1].min())
            xmax = int(positions[1].max()) + 1

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes_list.append([xmin, ymin, xmax, ymax])
            valid_masks.append(mask)

        if not boxes_list:
            raise RuntimeError(f"No valid instances in {self.masks[index]}")

        boxes = torch.tensor(boxes_list, dtype=torch.float32)
        masks = torch.stack(valid_masks).to(torch.uint8)
        labels = torch.ones((len(boxes_list),), dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes_list),), dtype=torch.int64)

        target: Dict[str, Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        image_tensor, target = self.transforms(image, target)
        return image_tensor, target


def collate_fn(
    batch: Sequence[Tuple[Tensor, Dict[str, Tensor]]],
) -> Tuple[Tuple[Tensor, ...], Tuple[Dict[str, Tensor], ...]]:
    return tuple(zip(*batch))


def split_indices(
    dataset_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    max_samples: Optional[int],
) -> Tuple[List[int], List[int], List[int]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    if max_samples is not None:
        if max_samples < 3:
            raise ValueError("--max-samples must be at least 3.")
        indices = indices[: min(max_samples, dataset_size)]

    n = len(indices)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    return train_indices, val_indices, test_indices


def build_loaders(
    dataset_root: Path,
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, Subset]:
    train_dataset_full = PennFudanDataset(dataset_root, train=True)
    eval_dataset_full = PennFudanDataset(dataset_root, train=False)

    train_idx, val_idx, test_idx = split_indices(
        dataset_size=len(train_dataset_full),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    train_dataset = Subset(train_dataset_full, train_idx)
    val_dataset = Subset(eval_dataset_full, val_idx)
    test_dataset = Subset(eval_dataset_full, test_idx)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, test_dataset


def build_model(
    num_classes: int,
    min_size: int,
    max_size: int,
    trainable_backbone_layers: int,
    pretrained: bool,
) -> nn.Module:
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    model = maskrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
    )

    box_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        box_in_features,
        num_classes,
    )

    mask_in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in_features,
        hidden_layer,
        num_classes,
    )

    return model


def move_targets_to_device(
    targets: Iterable[Dict[str, Tensor]],
    device: torch.device,
) -> List[Dict[str, Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(enabled: bool):
    # torch.amp.GradScaler is the current API. The fallback supports older PyTorch.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any,
    amp_enabled: bool,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running: Dict[str, float] = defaultdict(float)
    sample_count = 0

    progress = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for images, targets in progress:
        images = [image.to(device, non_blocking=True) for image in images]
        targets_device = move_targets_to_device(targets, device)
        batch_size = len(images)

        optimizer.zero_grad(set_to_none=True)

        with amp_context(device, amp_enabled):
            loss_dict = model(images, targets_device)
            total_loss = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite loss detected: {float(total_loss.detach().cpu())}"
            )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["total_loss"] += float(total_loss.detach().cpu()) * batch_size
        for name, loss_value in loss_dict.items():
            running[name] += float(loss_value.detach().cpu()) * batch_size
        sample_count += batch_size

        progress.set_postfix(loss=f"{float(total_loss.detach().cpu()):.4f}")

    return {name: value / max(sample_count, 1) for name, value in running.items()}


@torch.inference_mode()
def calculate_validation_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
) -> Dict[str, float]:
    # Detection models return losses only in train mode. FrozenBatchNorm in the
    # default Mask R-CNN backbone makes this practical for validation.
    model.train()
    running: Dict[str, float] = defaultdict(float)
    sample_count = 0

    progress = tqdm(loader, desc=f"Val loss {epoch}", leave=False)
    for images, targets in progress:
        images = [image.to(device, non_blocking=True) for image in images]
        targets_device = move_targets_to_device(targets, device)
        batch_size = len(images)

        with amp_context(device, amp_enabled):
            loss_dict = model(images, targets_device)
            total_loss = sum(loss for loss in loss_dict.values())

        running["total_loss"] += float(total_loss.detach().cpu()) * batch_size
        for name, loss_value in loss_dict.items():
            running[name] += float(loss_value.detach().cpu()) * batch_size
        sample_count += batch_size

    return {name: value / max(sample_count, 1) for name, value in running.items()}


def binary_mask_iou(mask_a: Tensor, mask_b: Tensor) -> float:
    mask_a = mask_a.bool()
    mask_b = mask_b.bool()
    intersection = torch.logical_and(mask_a, mask_b).sum().item()
    union = torch.logical_or(mask_a, mask_b).sum().item()
    return float(intersection / union) if union > 0 else 0.0


def binary_mask_dice(mask_a: Tensor, mask_b: Tensor) -> float:
    mask_a = mask_a.bool()
    mask_b = mask_b.bool()
    intersection = torch.logical_and(mask_a, mask_b).sum().item()
    denominator = mask_a.sum().item() + mask_b.sum().item()
    return float(2.0 * intersection / denominator) if denominator > 0 else 0.0


def greedy_match_masks(
    predicted_masks: Tensor,
    true_masks: Tensor,
    match_iou_threshold: float,
) -> Tuple[int, int, int, List[float], List[float]]:
    num_pred = len(predicted_masks)
    num_true = len(true_masks)

    if num_pred == 0:
        return 0, 0, num_true, [], []
    if num_true == 0:
        return 0, num_pred, 0, [], []

    candidates: List[Tuple[float, int, int]] = []
    for pred_index in range(num_pred):
        for true_index in range(num_true):
            iou = binary_mask_iou(predicted_masks[pred_index], true_masks[true_index])
            candidates.append((iou, pred_index, true_index))

    candidates.sort(reverse=True, key=lambda item: item[0])
    used_pred = set()
    used_true = set()
    matched_ious: List[float] = []
    matched_dices: List[float] = []

    for iou, pred_index, true_index in candidates:
        if iou < match_iou_threshold:
            break
        if pred_index in used_pred or true_index in used_true:
            continue

        used_pred.add(pred_index)
        used_true.add(true_index)
        matched_ious.append(iou)
        matched_dices.append(
            binary_mask_dice(
                predicted_masks[pred_index],
                true_masks[true_index],
            )
        )

    true_positive = len(used_pred)
    false_positive = num_pred - true_positive
    false_negative = num_true - true_positive
    return true_positive, false_positive, false_negative, matched_ious, matched_dices


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    mask_threshold: float,
    match_iou_threshold: float,
) -> Dict[str, float]:
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious: List[float] = []
    all_dices: List[float] = []
    inference_times: List[float] = []

    progress = tqdm(loader, desc="Evaluating", leave=False)
    for images, targets in progress:
        images_device = [image.to(device, non_blocking=True) for image in images]

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(images_device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed / max(len(images), 1))

        for output, target in zip(outputs, targets):
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()
            keep = (scores >= score_threshold) & (labels == 1)

            predicted_masks = (
                output["masks"].detach().cpu()[keep, 0] >= mask_threshold
            )
            true_masks = target["masks"].detach().cpu().bool()

            tp, fp, fn, ious, dices = greedy_match_masks(
                predicted_masks,
                true_masks,
                match_iou_threshold,
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_ious.extend(ious)
            all_dices.extend(dices)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "true_positives": float(total_tp),
        "false_positives": float(total_fp),
        "false_negatives": float(total_fn),
        "precision_at_iou_0_5": float(precision),
        "recall_at_iou_0_5": float(recall),
        "f1_at_iou_0_5": float(f1),
        "mean_iou_of_matched_instances": float(np.mean(all_ious)) if all_ious else 0.0,
        "mean_dice_of_matched_instances": float(np.mean(all_dices)) if all_dices else 0.0,
        "mean_inference_seconds_per_image": (
            float(np.mean(inference_times)) if inference_times else 0.0
        ),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    history: List[Dict[str, float]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "args": vars(args),
            "class_names": CLASS_NAMES,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def save_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    fieldnames = sorted({key for row in history for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_loss_plot(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [row["train_total_loss"] for row in history]
    val_loss = [row["val_total_loss"] for row in history]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train total loss")
    plt.plot(epochs, val_loss, marker="o", label="Validation total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mask R-CNN training history")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def tensor_to_pil(image: Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    return TF.to_pil_image(image)


def blend_mask(
    image_np: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    result = image_np.astype(np.float32).copy()
    color_array = np.asarray(color, dtype=np.float32)
    result[mask] = (1.0 - alpha) * result[mask] + alpha * color_array
    return np.clip(result, 0, 255).astype(np.uint8)


@torch.inference_mode()
def save_predictions(
    model: nn.Module,
    dataset: Subset,
    device: torch.device,
    output_dir: Path,
    score_threshold: float,
    mask_threshold: float,
    num_images: int,
    seed: int,
) -> None:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    count = min(num_images, len(dataset))
    rng = random.Random(seed)
    chosen_indices = rng.sample(range(len(dataset)), k=count)

    palette = [
        (239, 83, 80),
        (66, 165, 245),
        (102, 187, 106),
        (255, 202, 40),
        (171, 71, 188),
        (38, 198, 218),
    ]

    for output_index, dataset_index in enumerate(chosen_indices, start=1):
        image, target = dataset[dataset_index]
        prediction = model([image.to(device)])[0]

        scores = prediction["scores"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        keep = (scores >= score_threshold) & (labels == 1)

        boxes = prediction["boxes"].detach().cpu()[keep]
        kept_scores = scores[keep]
        masks = prediction["masks"].detach().cpu()[keep, 0] >= mask_threshold

        original = np.asarray(tensor_to_pil(image)).copy()
        overlay = original.copy()

        for instance_index, mask in enumerate(masks):
            color = palette[instance_index % len(palette)]
            overlay = blend_mask(
                overlay,
                mask.numpy().astype(bool),
                color=color,
            )

        visual = Image.fromarray(overlay)
        draw = ImageDraw.Draw(visual)

        for instance_index, (box, score) in enumerate(zip(boxes, kept_scores)):
            color = palette[instance_index % len(palette)]
            x1, y1, x2, y2 = [float(value) for value in box]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            text = f"person {float(score):.2f}"
            text_box = draw.textbbox((x1, y1), text)
            draw.rectangle(text_box, fill=color)
            draw.text((x1, y1), text, fill=(255, 255, 255))

        # Ground-truth boxes are drawn in white for visual comparison.
        for true_box in target["boxes"]:
            x1, y1, x2, y2 = [float(value) for value in true_box]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255), width=1)

        destination = output_dir / f"prediction_{output_index:02d}.png"
        visual.save(destination)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, float]], Path]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)

    checkpoint_dir = args.output_dir / "checkpoints"
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"

    start_epoch = 1
    best_val_loss = math.inf
    history: List[Dict[str, float]] = []

    if args.resume:
        checkpoint_path = args.checkpoint or last_path
        checkpoint = load_checkpoint(
            checkpoint_path,
            model,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        history = list(checkpoint.get("history", []))
        logger.info("Resumed from %s at epoch %d", checkpoint_path, start_epoch)

    logger.info("AMP enabled: %s", amp_enabled)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            amp_enabled,
            epoch,
        )
        val_metrics = calculate_validation_loss(
            model,
            val_loader,
            device,
            amp_enabled,
            epoch,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "learning_rate": current_lr,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)

        val_total_loss = val_metrics["total_loss"]
        logger.info(
            "Epoch %d/%d | train loss %.4f | val loss %.4f | lr %.6f",
            epoch,
            args.epochs,
            train_metrics["total_loss"],
            val_total_loss,
            current_lr,
        )

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                history,
                args,
            )
            logger.info("Saved new best checkpoint: %s", best_path)

        scheduler.step()

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            history,
            args,
        )
        save_history_csv(
            history,
            args.output_dir / "metrics" / "training_history.csv",
        )
        save_loss_plot(
            history,
            args.output_dir / "plots" / "loss_curve.png",
        )

    return history, best_path


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.output_dir)
    seed_everything(args.seed)

    device = resolve_device(args.device)
    logger.info("PyTorch: %s", torch.__version__)
    logger.info("TorchVision: %s", torchvision.__version__)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    dataset_root = ensure_dataset(args.data_dir, logger)
    train_loader, val_loader, test_loader, test_dataset = build_loaders(
        dataset_root,
        args,
    )
    logger.info(
        "Dataset split | train=%d | validation=%d | test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model = build_model(
        num_classes=2,
        min_size=args.min_size,
        max_size=args.max_size,
        trainable_backbone_layers=args.trainable_backbone_layers,
        pretrained=not args.no_pretrained,
    )
    model.to(device)

    best_path = args.output_dir / "checkpoints" / "best_model.pth"

    if args.predict_only:
        checkpoint_path = args.checkpoint or best_path
        load_checkpoint(checkpoint_path, model, device)
        logger.info("Loaded checkpoint for prediction: %s", checkpoint_path)
    else:
        _, best_path = train(
            model,
            train_loader,
            val_loader,
            device,
            args,
            logger,
        )
        load_checkpoint(best_path, model, device)
        logger.info("Loaded best checkpoint for final evaluation: %s", best_path)

    metrics = evaluate_model(
        model,
        test_loader,
        device,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        match_iou_threshold=args.match_iou_threshold,
    )

    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    logger.info("Test metrics:")
    for key, value in metrics.items():
        logger.info("  %s: %.6f", key, value)

    save_predictions(
        model,
        test_dataset,
        device,
        output_dir=args.output_dir / "predictions",
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        num_images=args.num_predictions,
        seed=args.seed,
    )

    logger.info("Finished successfully.")
    logger.info("Best checkpoint: %s", best_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Predictions: %s", args.output_dir / "predictions")
    logger.info("Loss curve: %s", args.output_dir / "plots" / "loss_curve.png")


if __name__ == "__main__":
    main()
