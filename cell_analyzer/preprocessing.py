"""Optional per-channel Gaussian smoothing without intensity transformation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .models import ProcessedChannel


def preprocess_channel(image: np.ndarray, config: dict[str, Any]) -> ProcessedChannel:
    """Convert a channel to float and optionally apply Gaussian smoothing."""

    processed, _ = preprocess_channel_steps(image, config)
    return processed


def preprocess_channel_steps(
    image: np.ndarray, config: dict[str, Any]
) -> tuple[ProcessedChannel, dict[str, np.ndarray]]:
    """Return raw-intensity analysis data and the Gaussian preview stage."""

    raw = np.asarray(image)
    working = raw.astype(np.float32, copy=False)
    working = np.nan_to_num(working, nan=0.0, posinf=0.0, neginf=0.0)

    gaussian_sigma = float(config.get("gaussian_sigma_px", 0.0))
    if gaussian_sigma > 0:
        analysis_image = ndi.gaussian_filter(working, sigma=gaussian_sigma)
    else:
        analysis_image = working.copy()
    analysis_image = analysis_image.astype(np.float32, copy=False)

    processed = ProcessedChannel(
        raw=raw,
        analysis_image=analysis_image,
    )
    steps = {
        "raw": raw,
        "gaussian_smoothed": analysis_image,
    }
    return processed, steps
