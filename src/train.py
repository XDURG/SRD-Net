from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PairedUltrasoundDataset
from models import MultiTaskLoss, SRDNet
from utils import binary_classification_metrics, dice_score, ensure_dir, load_yaml, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRD-Net for multimodal ultrasound segmentation and recurrence classification.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file.")
    return parser.parse_args()


def build_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    dataset = PairedUltrasoundDataset(
        manifest_path=config["data"][split]["manifest"],
        image_size=config["data"]["image_size"],
    )
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=split == "train",
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "sample_id": batch["sample_id"],
        "bmode": batch["bmode"].to(device),
        "cdfi": batch["cdfi"].to(device),
        "mask": batch["mask"].to(device),
        "label": batch["label"].to(device),
    }


def evaluate(model: SRDNet, dataloader: DataLoader, criterion: MultiTaskLoss, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    dice_scores: list[float] = []
    classification_probabilities = []
    classification_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            outputs = model(batch["bmode"], batch["cdfi"])
            loss, _ = criterion(
                outputs.segmentation_logits,
                batch["mask"],
                outputs.classification_logits,
                batch["label"],
            )
            losses.append(loss.item())
            dice_scores.append(dice_score(outputs.segmentation_probability, batch["mask"]))
            classification_probabilities.append(outputs.classification_probability.cpu())
            classification_targets.append(batch["label"].cpu())

    probabilities = torch.cat(classification_probabilities, dim=0)
    targets = torch.cat(classification_targets, dim=0)
    classification_metrics = binary_classification_metrics(probabilities, targets)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "dice": sum(dice_scores) / max(len(dice_scores), 1),
        **classification_metrics,
    }


def train_one_epoch(
    model: SRDNet,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: AdamW,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_losses: list[float] = []
    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["bmode"], batch["cdfi"])
        loss, loss_dict = criterion(
            outputs.segmentation_logits,
            batch["mask"],
            outputs.classification_logits,
            batch["label"],
        )
        loss.backward()
        optimizer.step()
        running_losses.append(loss.item())
        progress.set_postfix(loss=loss_dict["loss_total"])
    return {"loss": sum(running_losses) / max(len(running_losses), 1)}


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(config["experiment"]["seed"])

    device = torch.device(config["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(config["experiment"]["output_dir"])
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")

    train_loader = build_dataloader(config, split="train")
    val_loader = build_dataloader(config, split="val")

    model = SRDNet(
        backbone_name=config["model"]["backbone_name"],
        pretrained_backbone=config["model"]["pretrained_backbone"],
        fusion_channels=config["model"]["fusion_channels"],
        decoder_channels=config["model"]["decoder_channels"],
        classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)

    criterion = MultiTaskLoss(
        segmentation_weight=config["loss"]["segmentation_weight"],
        classification_weight=config["loss"]["classification_weight"],
    )
    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"], eta_min=config["training"]["min_learning_rate"])

    history: list[dict[str, float]] = []
    best_dice = -1.0

    for epoch in range(config["training"]["epochs"]):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_accuracy": val_metrics["accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_precision": val_metrics["precision"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2))

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "metrics": epoch_record,
                },
                checkpoints_dir / "best_model.pt",
            )

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()
