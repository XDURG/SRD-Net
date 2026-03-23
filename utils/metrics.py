from __future__ import annotations

import torch


def dice_score(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    predictions = (probabilities >= threshold).float().flatten(1)
    targets = targets.flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def binary_classification_metrics(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).float()
    targets = targets.float()
    true_positive = ((predictions == 1) & (targets == 1)).sum().item()
    true_negative = ((predictions == 0) & (targets == 0)).sum().item()
    false_positive = ((predictions == 1) & (targets == 0)).sum().item()
    false_negative = ((predictions == 0) & (targets == 1)).sum().item()

    accuracy = (true_positive + true_negative) / max(true_positive + true_negative + false_positive + false_negative, 1)
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }
