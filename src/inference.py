from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from datasets import PairedUltrasoundDataset
from models import SRDNet
from utils import ensure_dir, load_yaml, save_prediction_overlay, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SRD-Net inference on paired B-mode and CDFI ultrasound images.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest for inference. Defaults to config data.test.manifest.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(config["experiment"]["seed"])
    device = torch.device(config["experiment"]["device"] if torch.cuda.is_available() else "cpu")

    manifest = args.manifest or config["data"]["test"]["manifest"]
    output_dir = ensure_dir(args.output_dir or Path(config["experiment"]["output_dir"]) / "inference")
    dataset = PairedUltrasoundDataset(manifest_path=manifest, image_size=config["data"]["image_size"])

    model = SRDNet(
        backbone_name=config["model"]["backbone_name"],
        pretrained_backbone=False,
        fusion_channels=config["model"]["fusion_channels"],
        decoder_channels=config["model"]["decoder_channels"],
        classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions_csv = output_dir / "predictions.csv"
    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "recurrence_probability", "predicted_label", "mask_path"])

        with torch.no_grad():
            for sample in dataset:
                bmode = sample["bmode"].unsqueeze(0).to(device)
                cdfi = sample["cdfi"].unsqueeze(0).to(device)
                outputs = model(bmode, cdfi)
                recurrence_probability = outputs.classification_probability.squeeze().item()
                mask_probability = outputs.segmentation_probability.squeeze().cpu().numpy()
                mask_output_path = output_dir / f"{sample['sample_id']}_mask.png"
                save_prediction_overlay(mask_probability.astype(np.float32), mask_output_path)
                writer.writerow(
                    [
                        sample["sample_id"],
                        f"{recurrence_probability:.6f}",
                        int(recurrence_probability >= config["inference"]["classification_threshold"]),
                        mask_output_path,
                    ]
                )

    print(f"Inference complete. Predictions saved to: {predictions_csv}")


if __name__ == "__main__":
    main()
