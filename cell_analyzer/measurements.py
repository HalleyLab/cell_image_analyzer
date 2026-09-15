"""Per-ROI morphology and per-channel fluorescence measurements."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure

from .config import slugify
from .models import CziInfo, ProcessedChannel
from .segmentation import threshold_image


def _measurement_mask(
    analysis_image: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, float | None]:
    threshold_config = config.get("measurement_threshold", {}) or {}
    return threshold_image(
        analysis_image,
        method=threshold_config.get("method", "otsu"),
        percentile=float(threshold_config.get("percentile", 95.0)),
        threshold_scale=float(threshold_config.get("scale", 1.0)),
        manual_threshold=float(threshold_config.get("value", 0.0)),
    )


def measure_rois(
    labels: np.ndarray,
    channels: dict[int, ProcessedChannel],
    channel_configs: dict[str, dict[str, Any]],
    info: CziInfo,
    zoom: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | None]]:
    """Measure cell shape and every channel within the same ROI labels."""

    regions = measure.regionprops(labels)
    records: list[dict[str, Any]] = []
    area_scale_to_source = 1.0 / (zoom * zoom)
    effective_pixel_size_x = (
        info.pixel_size_um_x / zoom if info.pixel_size_um_x is not None else None
    )
    effective_pixel_size_y = (
        info.pixel_size_um_y / zoom if info.pixel_size_um_y is not None else None
    )
    pixel_area_um2 = (
        effective_pixel_size_x * effective_pixel_size_y
        if effective_pixel_size_x is not None and effective_pixel_size_y is not None
        else None
    )

    for region in regions:
        perimeter = float(region.perimeter)
        area = float(region.area)
        records.append(
            {
                "roi_id": int(region.label),
                "roi_area_analysis_px": area,
                "roi_area_source_px": area * area_scale_to_source,
                "roi_area_um2": area * pixel_area_um2 if pixel_area_um2 else np.nan,
                "perimeter_analysis_px": perimeter,
                "circularity": (
                    4.0 * math.pi * area / (perimeter * perimeter)
                    if perimeter > 0
                    else 0.0
                ),
            }
        )
    table = pd.DataFrame.from_records(records)
    thresholds: dict[str, float | None] = {}
    if table.empty:
        table = pd.DataFrame(
            columns=[
                "roi_id",
                "roi_area_analysis_px",
                "roi_area_source_px",
                "roi_area_um2",
                "perimeter_analysis_px",
                "circularity",
            ]
        )

    roi_count = int(labels.max())
    label_ids = np.arange(1, roi_count + 1)
    for channel_index, processed in sorted(channels.items()):
        channel_config = channel_configs[str(channel_index)]
        alias = str(channel_config.get("alias") or f"Channel_{channel_index}")
        column_id = slugify(alias, fallback=f"channel_{channel_index}")
        positive_mask, threshold = _measurement_mask(
            processed.analysis_image, channel_config
        )
        thresholds[column_id] = threshold
        thresholded_raw = np.where(positive_mask, processed.raw, 0)

        if roi_count:
            raw_means = ndi.mean(thresholded_raw, labels=labels, index=label_ids)
            integrated_intensities = ndi.sum(
                thresholded_raw, labels=labels, index=label_ids
            )
            positive_counts = np.bincount(
                labels[positive_mask].ravel(), minlength=roi_count + 1
            )[1 : roi_count + 1].astype(float)
            roi_counts = np.bincount(labels.ravel(), minlength=roi_count + 1)[
                1 : roi_count + 1
            ].astype(float)
        else:
            raw_means = np.array([], dtype=float)
            integrated_intensities = np.array([], dtype=float)
            positive_counts = np.array([], dtype=float)
            roi_counts = np.array([], dtype=float)

        table[f"{column_id}_mean_intensity"] = raw_means
        table[f"{column_id}_integrated_intensity"] = integrated_intensities
        table[f"{column_id}_positive_area_um2"] = (
            positive_counts * pixel_area_um2 if pixel_area_um2 else np.nan
        )
        table[f"{column_id}_positive_fraction"] = np.divide(
            positive_counts,
            roi_counts,
            out=np.zeros_like(positive_counts),
            where=roi_counts > 0,
        )

    summary_records: list[dict[str, Any]] = [
        {"metric": "roi_count", "value": roi_count},
        {"metric": "analysis_zoom", "value": zoom},
        {"metric": "effective_pixel_size_um_x", "value": effective_pixel_size_x},
        {"metric": "effective_pixel_size_um_y", "value": effective_pixel_size_y},
    ]
    if not table.empty:
        summary_records.extend(
            [
                {"metric": "mean_roi_area_source_px", "value": table["roi_area_source_px"].mean()},
                {"metric": "median_roi_area_source_px", "value": table["roi_area_source_px"].median()},
                {"metric": "mean_roi_area_um2", "value": table["roi_area_um2"].mean()},
            ]
        )
    summary = pd.DataFrame(summary_records)
    return table, summary, thresholds
