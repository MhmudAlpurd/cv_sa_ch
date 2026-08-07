from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation


# ============================================================
# 1. تنظیمات
# ============================================================

@dataclass
class Config:
  
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    best_model_dir: str = "./outputs/best_model"

   
    model_name: str = "nvidia/mit-b0"
    num_classes: int = 3

  
    image_size: int = 256

    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 6e-5
    weight_decay: float = 1e-4

 
    validation_ratio: float = 0.15

    # DataLoader
    num_workers: int = min(4, os.cpu_count() or 1)
    pin_memory: bool = True

 
    patience: int = 4

    seed: int = 42

  
    num_visualizations: int = 8


CFG = Config()



ID2LABEL = {
    0: "pet",
    1: "background",
    2: "border",
}

LABEL2ID = {
    value: key
    for key, value in ID2LABEL.items()
}



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



class SegmentationTransform:

    def __init__(
        self,
        image_size: int,
        train: bool,
    ) -> None:
        self.image_size = image_size
        self.train = train

        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        image = image.convert("RGB")


        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.8:
                image = self.color_jitter(image)


        image = TF.resize(
            image,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        mask = TF.resize(
            mask,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)

        image_tensor = TF.normalize(
            image_tensor,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )


        mask_array = np.array(mask, dtype=np.int64)

        """
        ماسک Oxford-IIIT Pet دارای مقادیر زیر است:

        1 = حیوان
        2 = پس‌زمینه
        3 = مرز

        مدل CrossEntropyLoss نیاز دارد کلاس‌ها از صفر شروع شوند:

        0 = حیوان
        1 = پس‌زمینه
        2 = مرز
        """

        mask_array = mask_array - 1

     
        mask_array = np.clip(
            mask_array,
            0,
            CFG.num_classes - 1,
        )

        mask_tensor = torch.from_numpy(mask_array).long()

        return image_tensor, mask_tensor



class OxfordPetSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: SegmentationTransform,
        download: bool = True,
    ) -> None:

        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        image, mask = self.dataset[index]

        pixel_values, labels = self.transform(
            image,
            mask,
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }



def build_dataloaders(
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    print("\nDownloading/loading Oxford-IIIT Pet dataset...")

    train_transform = SegmentationTransform(
        image_size=config.image_size,
        train=True,
    )

    eval_transform = SegmentationTransform(
        image_size=config.image_size,
        train=False,
    )

    # دو Dataset جدا می‌سازیم تا Transform متفاوت داشته باشند.
    full_train_dataset = OxfordPetSegmentationDataset(
        root=config.data_dir,
        split="trainval",
        transform=train_transform,
        download=True,
    )

    full_val_dataset = OxfordPetSegmentationDataset(
        root=config.data_dir,
        split="trainval",
        transform=eval_transform,
        download=True,
    )

    test_dataset = OxfordPetSegmentationDataset(
        root=config.data_dir,
        split="test",
        transform=eval_transform,
        download=True,
    )

    dataset_size = len(full_train_dataset)

    generator = torch.Generator().manual_seed(
        config.seed
    )

    indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    validation_size = int(
        dataset_size * config.validation_ratio
    )

    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    train_dataset = Subset(
        full_train_dataset,
        train_indices,
    )

    val_dataset = Subset(
        full_val_dataset,
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    print(f"Train samples:      {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def build_model(
    config: Config,
) -> SegformerForSemanticSegmentation:

    print(f"\nLoading model: {config.model_name}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        config.model_name,
        num_labels=config.num_classes,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    return model



@torch.no_grad()
def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:

    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    valid_mask = (
        (targets >= 0)
        & (targets < num_classes)
    )

    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    encoded = (
        targets * num_classes
        + predictions
    )

    batch_matrix = torch.bincount(
        encoded,
        minlength=num_classes ** 2,
    )

    batch_matrix = batch_matrix.reshape(
        num_classes,
        num_classes,
    )

    confusion_matrix += batch_matrix.cpu()

    return confusion_matrix



def calculate_metrics(
    confusion_matrix: torch.Tensor,
) -> dict[str, object]:

    confusion_matrix = confusion_matrix.float()

    true_positive = torch.diag(confusion_matrix)

    ground_truth_total = confusion_matrix.sum(dim=1)
    prediction_total = confusion_matrix.sum(dim=0)

    union = (
        ground_truth_total
        + prediction_total
        - true_positive
    )

    class_iou = true_positive / union.clamp_min(1)

    valid_classes = union > 0

    mean_iou = class_iou[valid_classes].mean().item()

    pixel_accuracy = (
        true_positive.sum()
        / confusion_matrix.sum().clamp_min(1)
    ).item()

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "class_iou": class_iou.tolist(),
    }



def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    num_classes: int,
) -> dict[str, object]:

    model.train()

    total_loss = 0.0

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    use_amp = device.type == "cuda"

    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

      
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        predictions = logits.argmax(dim=1)

        confusion_matrix = update_confusion_matrix(
            confusion_matrix,
            predictions.detach(),
            labels.detach(),
            num_classes,
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    metrics = calculate_metrics(
        confusion_matrix
    )

    metrics["loss"] = total_loss / len(dataloader)

    return metrics



@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    description: str = "Validation",
) -> dict[str, object]:

    model.eval()

    total_loss = 0.0

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )

    progress_bar = tqdm(
        dataloader,
        desc=description,
        leave=False,
    )

    use_amp = device.type == "cuda"

    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

        total_loss += loss.item()

        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        predictions = logits.argmax(dim=1)

        confusion_matrix = update_confusion_matrix(
            confusion_matrix,
            predictions,
            labels,
            num_classes,
        )

    metrics = calculate_metrics(
        confusion_matrix
    )

    metrics["loss"] = total_loss / len(dataloader)

    return metrics



def save_history(
    history: dict[str, list],
    output_path: Path,
) -> None:

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False,
        )



def plot_training_history(
    history: dict[str, list],
    output_dir: Path,
) -> None:

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )


    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=150,
    )
    plt.close()


    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_miou"],
        label="Train mIoU",
    )
    plt.plot(
        epochs,
        history["val_miou"],
        label="Validation mIoU",
    )
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Training and Validation mIoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "miou_curve.png",
        dpi=150,
    )
    plt.close()



def denormalize_image(
    image_tensor: torch.Tensor,
) -> np.ndarray:

    image = image_tensor.detach().cpu().clone()

    mean = torch.tensor(
        IMAGENET_MEAN
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD
    ).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)

    image = image.permute(
        1,
        2,
        0,
    ).numpy()

    return image



def colorize_mask(
    mask: np.ndarray,
) -> np.ndarray:

    palette = np.array(
        [
            [255, 170, 0],   # حیوان
            [0, 0, 0],       # پس‌زمینه
            [0, 170, 255],   # مرز
        ],
        dtype=np.uint8,
    )

    mask = np.clip(
        mask,
        0,
        len(palette) - 1,
    )

    return palette[mask]



@torch.no_grad()
def save_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    number_of_images: int,
) -> None:

    model.eval()

    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_count = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        outputs = model(
            pixel_values=pixel_values
        )

        logits = F.interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        predictions = logits.argmax(dim=1).cpu()

        for index in range(pixel_values.shape[0]):
            image = denormalize_image(
                pixel_values[index]
            )

            true_mask = labels[index].numpy()
            predicted_mask = predictions[index].numpy()

            true_color = colorize_mask(true_mask)
            predicted_color = colorize_mask(predicted_mask)

            figure, axes = plt.subplots(
                1,
                3,
                figsize=(15, 5),
            )

            axes[0].imshow(image)
            axes[0].set_title("Input Image")

            axes[1].imshow(true_color)
            axes[1].set_title("Ground Truth")

            axes[2].imshow(predicted_color)
            axes[2].set_title("Prediction")

            for axis in axes:
                axis.axis("off")

            plt.tight_layout()

            output_path = (
                prediction_dir
                / f"prediction_{saved_count:03d}.png"
            )

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(figure)

            saved_count += 1

            if saved_count >= number_of_images:
                print(
                    f"Saved {saved_count} predictions "
                    f"to {prediction_dir}"
                )
                return



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Config,
) -> dict[str, list]:

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    use_amp = device.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_miou": [],
        "val_miou": [],
        "train_pixel_accuracy": [],
        "val_pixel_accuracy": [],
    }

    best_val_miou = -1.0
    epochs_without_improvement = 0

    output_dir = Path(config.output_dir)
    best_model_dir = Path(config.best_model_dir)

    for epoch in range(
        1,
        config.num_epochs + 1,
    ):
        print(
            f"\n{'=' * 60}\n"
            f"Epoch {epoch}/{config.num_epochs}\n"
            f"{'=' * 60}"
        )

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            num_classes=config.num_classes,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            num_classes=config.num_classes,
            description="Validation",
        )

        scheduler.step(
            val_metrics["mean_iou"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(
            train_metrics["loss"]
        )
        history["val_loss"].append(
            val_metrics["loss"]
        )
        history["train_miou"].append(
            train_metrics["mean_iou"]
        )
        history["val_miou"].append(
            val_metrics["mean_iou"]
        )
        history["train_pixel_accuracy"].append(
            train_metrics["pixel_accuracy"]
        )
        history["val_pixel_accuracy"].append(
            val_metrics["pixel_accuracy"]
        )

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train mIoU: {train_metrics['mean_iou']:.4f} | "
            f"Train Acc: {train_metrics['pixel_accuracy']:.4f}"
        )

        print(
            f"Val Loss:   {val_metrics['loss']:.4f} | "
            f"Val mIoU:   {val_metrics['mean_iou']:.4f} | "
            f"Val Acc:    {val_metrics['pixel_accuracy']:.4f}"
        )

        print(f"Learning Rate: {current_lr:.8f}")

        for class_id, class_iou in enumerate(
            val_metrics["class_iou"]
        ):
            print(
                f"  IoU {ID2LABEL[class_id]:>10}: "
                f"{class_iou:.4f}"
            )

        save_history(
            history,
            output_dir / "training_history.json",
        )


        if val_metrics["mean_iou"] > best_val_miou:
            best_val_miou = val_metrics["mean_iou"]
            epochs_without_improvement = 0

            best_model_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(
                best_model_dir
            )

            metadata = {
                "image_size": config.image_size,
                "num_classes": config.num_classes,
                "id2label": ID2LABEL,
                "best_validation_miou": best_val_miou,
                "epoch": epoch,
            }

            with (
                best_model_dir
                / "training_metadata.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    metadata,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            print(
                f"Best model saved. "
                f"Validation mIoU: {best_val_miou:.4f}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "No validation improvement. "
                f"Counter: {epochs_without_improvement}/"
                f"{config.patience}"
            )

        if (
            epochs_without_improvement
            >= config.patience
        ):
            print("\nEarly stopping activated.")
            break

    return history


def main() -> None:

    set_seed(CFG.seed)

    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024 ** 3
        )

        print(
            f"GPU memory: {total_memory:.2f} GB"
        )

    train_loader, val_loader, test_loader = (
        build_dataloaders(CFG)
    )

    model = build_model(CFG)
    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=CFG,
    )

    plot_training_history(
        history,
        output_dir,
    )


    print("\nLoading best saved model...")

    best_model = (
        SegformerForSemanticSegmentation
        .from_pretrained(CFG.best_model_dir)
        .to(device)
    )


    test_metrics = evaluate(
        model=best_model,
        dataloader=test_loader,
        device=device,
        num_classes=CFG.num_classes,
        description="Testing",
    )

    print("\nFinal test results")
    print("-" * 40)

    print(
        f"Test Loss:           "
        f"{test_metrics['loss']:.4f}"
    )

    print(
        f"Test Pixel Accuracy: "
        f"{test_metrics['pixel_accuracy']:.4f}"
    )

    print(
        f"Test mIoU:           "
        f"{test_metrics['mean_iou']:.4f}"
    )

    for class_id, class_iou in enumerate(
        test_metrics["class_iou"]
    ):
        print(
            f"Test IoU {ID2LABEL[class_id]:>10}: "
            f"{class_iou:.4f}"
        )

    with (
        output_dir
        / "test_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            test_metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )


    save_predictions(
        model=best_model,
        dataloader=test_loader,
        device=device,
        output_dir=output_dir,
        number_of_images=CFG.num_visualizations,
    )

    print("\nTraining completed successfully.")
    print(f"Best model: {CFG.best_model_dir}")
    print(f"Results:    {CFG.output_dir}")


if __name__ == "__main__":
    main()