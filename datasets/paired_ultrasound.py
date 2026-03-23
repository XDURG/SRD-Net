from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class PairedUltrasoundSample:
    sample_id: str
    bmode_path: Path
    cdfi_path: Path
    mask_path: Path
    label: int


class PairedUltrasoundDataset(Dataset):
    """Dataset for paired B-mode and CDFI ultrasound images acquired on the same imaging plane."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        self.image_size = image_size
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = self._load_manifest(self.manifest_path)

    @staticmethod
    def _load_manifest(manifest_path: Path) -> list[PairedUltrasoundSample]:
        samples: list[PairedUltrasoundSample] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                samples.append(
                    PairedUltrasoundSample(
                        sample_id=row["sample_id"],
                        bmode_path=Path(row["bmode_path"]),
                        cdfi_path=Path(row["cdfi_path"]),
                        mask_path=Path(row["mask_path"]),
                        label=int(row["label"]),
                    )
                )
        if not samples:
            raise ValueError(f"No samples found in manifest: {manifest_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path).convert("RGB").resize((self.image_size, self.image_size))

    def _load_mask(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Mask file not found: {path}")
        return Image.open(path).convert("L").resize((self.image_size, self.image_size))

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = array[..., None]
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        bmode = self._load_rgb(sample.bmode_path)
        cdfi = self._load_rgb(sample.cdfi_path)
        mask = self._load_mask(sample.mask_path)

        if self.transform is not None:
            bmode = self.transform(bmode)
            cdfi = self.transform(cdfi)
        else:
            bmode = self._to_tensor(bmode)
            cdfi = self._to_tensor(cdfi)

        if self.mask_transform is not None:
            mask_tensor = self.mask_transform(mask)
        else:
            mask_tensor = self._to_tensor(mask)

        mask_tensor = (mask_tensor > 0.5).float()

        return {
            "sample_id": sample.sample_id,
            "bmode": bmode,
            "cdfi": cdfi,
            "mask": mask_tensor,
            "label": torch.tensor([sample.label], dtype=torch.float32),
        }
