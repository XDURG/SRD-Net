from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SRDNetOutput:
    segmentation_logits: torch.Tensor
    segmentation_probability: torch.Tensor
    classification_logits: torch.Tensor
    classification_probability: torch.Tensor
    lesion_feature_map: torch.Tensor


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SwinEncoder(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.feature_info = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.backbone(x))


class MultiScaleFusionBlock(nn.Module):
    def __init__(self, b_channels: int, c_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(b_channels + c_channels, out_channels, kernel_size=1),
            ConvNormAct(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, b_feature: torch.Tensor, c_feature: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([b_feature, c_feature], dim=1)
        return self.fuse(fused)


class BoundaryAwareAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.attention_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.refine = ConvNormAct(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_features = self.edge_extractor(x)
        attention = self.attention_gate(edge_features)
        refined = x + x * attention + edge_features
        return self.refine(refined)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvNormAct(in_channels + skip_channels, out_channels)
        self.boundary_attention = BoundaryAwareAttention(out_channels)
        self.conv2 = ConvNormAct(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.boundary_attention(x)
        x = self.conv2(x)
        return x


class MaskGuidedPooling(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, features: torch.Tensor, mask_probability: torch.Tensor) -> torch.Tensor:
        resized_mask = F.interpolate(mask_probability, size=features.shape[-2:], mode="bilinear", align_corners=False)
        lesion_features = features * resized_mask
        numerator = lesion_features.flatten(2).sum(dim=-1)
        denominator = resized_mask.flatten(2).sum(dim=-1).clamp_min(self.eps)
        pooled = numerator / denominator
        return pooled


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.pool = MaskGuidedPooling()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, mask_probability: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(features, mask_probability)
        logits = self.mlp(pooled)
        return logits, pooled


class SRDNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        pretrained_backbone: bool = False,
        fusion_channels: Sequence[int] = (96, 192, 384, 768),
        decoder_channels: Sequence[int] = (384, 192, 96),
        classifier_hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.bmode_encoder = SwinEncoder(backbone_name=backbone_name, pretrained=pretrained_backbone)
        self.cdfi_encoder = SwinEncoder(backbone_name=backbone_name, pretrained=pretrained_backbone)

        encoder_channels = self.bmode_encoder.feature_info
        fusion_channels = list(fusion_channels)
        decoder_channels = list(decoder_channels)

        self.fusion_blocks = nn.ModuleList(
            [
                MultiScaleFusionBlock(b_ch, c_ch, out_ch)
                for b_ch, c_ch, out_ch in zip(encoder_channels, encoder_channels, fusion_channels)
            ]
        )

        self.bottleneck = ConvNormAct(fusion_channels[-1], fusion_channels[-1])
        self.decoder3 = DecoderBlock(fusion_channels[-1], fusion_channels[-2], decoder_channels[0])
        self.decoder2 = DecoderBlock(decoder_channels[0], fusion_channels[-3], decoder_channels[1])
        self.decoder1 = DecoderBlock(decoder_channels[1], fusion_channels[-4], decoder_channels[2])
        self.stem_refine = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(decoder_channels[-1], decoder_channels[-1]),
            BoundaryAwareAttention(decoder_channels[-1]),
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)
        self.classification_head = ClassificationHead(
            in_channels=decoder_channels[-1],
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(self, bmode: torch.Tensor, cdfi: torch.Tensor) -> SRDNetOutput:
        b_features = self.bmode_encoder(bmode)
        c_features = self.cdfi_encoder(cdfi)
        fused_features = [block(b_feature, c_feature) for block, b_feature, c_feature in zip(self.fusion_blocks, b_features, c_features)]

        x = self.bottleneck(fused_features[-1])
        x = self.decoder3(x, fused_features[-2])
        x = self.decoder2(x, fused_features[-3])
        x = self.decoder1(x, fused_features[-4])
        lesion_feature_map = self.stem_refine(x)

        segmentation_logits = self.segmentation_head(lesion_feature_map)
        segmentation_probability = torch.sigmoid(segmentation_logits)
        classification_logits, _ = self.classification_head(lesion_feature_map, segmentation_probability)
        classification_probability = torch.sigmoid(classification_logits)

        return SRDNetOutput(
            segmentation_logits=segmentation_logits,
            segmentation_probability=segmentation_probability,
            classification_logits=classification_logits,
            classification_probability=classification_probability,
            lesion_feature_map=lesion_feature_map,
        )
