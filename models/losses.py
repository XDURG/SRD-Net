from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(1)
        targets = targets.flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        union = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        true_positive = (probabilities * targets).sum(dim=1)
        false_positive = (probabilities * (1 - targets)).sum(dim=1)
        false_negative = ((1 - probabilities) * targets).sum(dim=1)
        tversky = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + self.smooth
        )
        return torch.pow(1 - tversky, self.gamma).mean()


class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, focal_tversky_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.dice_loss = DiceLoss()
        self.focal_tversky_loss = FocalTverskyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice_loss(logits, targets) + self.focal_tversky_weight * self.focal_tversky_loss(logits, targets)


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_positive: float = 0.0, gamma_negative: float = 4.0, clip: float = 0.05) -> None:
        super().__init__()
        self.gamma_positive = gamma_positive
        self.gamma_negative = gamma_negative
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view_as(logits)
        probabilities = torch.sigmoid(logits)
        positive_probability = probabilities
        negative_probability = (1 - probabilities).clamp(min=self.clip)

        positive_loss = targets * torch.log(positive_probability.clamp_min(1e-6)) * torch.pow(1 - positive_probability, self.gamma_positive)
        negative_loss = (1 - targets) * torch.log(negative_probability.clamp_min(1e-6)) * torch.pow(probabilities, self.gamma_negative)
        return -(positive_loss + negative_loss).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, segmentation_weight: float = 1.0, classification_weight: float = 1.0) -> None:
        super().__init__()
        self.segmentation_loss = SegmentationLoss()
        self.classification_loss = AsymmetricFocalLoss()
        self.segmentation_weight = segmentation_weight
        self.classification_weight = classification_weight

    def forward(
        self,
        segmentation_logits: torch.Tensor,
        segmentation_targets: torch.Tensor,
        classification_logits: torch.Tensor,
        classification_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        seg_loss = self.segmentation_loss(segmentation_logits, segmentation_targets)
        cls_loss = self.classification_loss(classification_logits, classification_targets)
        total_loss = self.segmentation_weight * seg_loss + self.classification_weight * cls_loss
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_segmentation": seg_loss.item(),
            "loss_classification": cls_loss.item(),
        }
