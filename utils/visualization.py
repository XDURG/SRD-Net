from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_prediction_overlay(mask_probability: np.ndarray, output_path: str | Path) -> None:
    probability_map = (mask_probability * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(probability_map).save(output_path)
