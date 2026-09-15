"""Cell boundary detection, touching-cell separation, and shape filtering."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation


def threshold_image(
    image: np.ndarray,
    *,
    method: str,
    percentile: float = 90.0,
    adaptive_block_size_px: int = 51,
    adaptive_offset: float = 0.0,
    threshold_scale: float = 1.0,
    manual_threshold: float = 0.0,
    invert: bool = False,
) -> tuple[np.ndarray, float | None]:
    """Threshold an intensity image and return the mask and scalar threshold if any."""

    method = str(method).lower()
    finite_values = image[np.isfinite(image)]
    informative = finite_values[finite_values > 0]
    values = informative if informative.size >= 32 else finite_values

    if method == "none":
        mask = np.ones(image.shape, dtype=bool)
        threshold: float | None = None
    elif method == "manual":
        threshold = float(manual_threshold)
        mask = image >= threshold
    elif values.size == 0 or np.all(values == values[0]):
        threshold = float(values[0]) if values.size else 0.0
        if method in {"otsu", "yen", "triangle", "percentile"}:
            threshold *= float(threshold_scale)
        mask = image >= threshold if threshold > 0 else image > threshold
    elif method == "otsu":
        threshold = float(filters.threshold_otsu(values)) * float(threshold_scale)
        mask = image > threshold
    elif method == "yen":
        threshold = float(filters.threshold_yen(values)) * float(threshold_scale)
        mask = image > threshold
    elif method == "triangle":
        threshold = float(filters.threshold_triangle(values)) * float(threshold_scale)
        mask = image > threshold
    elif method == "percentile":
        threshold = float(np.percentile(values, float(percentile))) * float(
            threshold_scale
        )
        mask = image > threshold
    elif method == "adaptive":
        block_size = max(3, int(adaptive_block_size_px))
        if block_size % 2 == 0:
            block_size += 1
        local = filters.threshold_local(
            image,
            block_size=block_size,
            offset=float(adaptive_offset),
        )
        threshold = None
        mask = image > local
    else:
        raise ValueError(
            "Unknown threshold method. Use none, manual, otsu, yen, triangle, percentile, "
            "or adaptive."
        )
    if invert:
        mask = ~mask
    return np.asarray(mask, dtype=bool), threshold


def _morphological_cleanup(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    opening_radius = max(0, int(config.get("opening_radius_px", 0)))
    closing_radius = max(0, int(config.get("closing_radius_px", 0)))
    if opening_radius:
        mask = morphology.opening(mask, morphology.disk(opening_radius))
    if closing_radius:
        mask = morphology.closing(mask, morphology.disk(closing_radius))

    min_hole_area = max(0, int(config.get("min_hole_area_px", 0)))
    if bool(config.get("fill_all_holes", True)):
        mask = ndi.binary_fill_holes(mask)
    elif min_hole_area:
        inverse_labels, _ = ndi.label(~mask)
        inverse_sizes = np.bincount(inverse_labels.ravel())
        border_ids = np.unique(
            np.concatenate(
                [
                    inverse_labels[0],
                    inverse_labels[-1],
                    inverse_labels[:, 0],
                    inverse_labels[:, -1],
                ]
            )
        )
        fill_ids = np.flatnonzero(inverse_sizes < min_hole_area)
        fill_ids = np.setdiff1d(fill_ids, border_ids, assume_unique=False)
        if fill_ids.size:
            mask = mask | np.isin(inverse_labels, fill_ids)
    min_area = max(1, int(config.get("min_area_px", 1)))
    foreground_labels, _ = ndi.label(mask)
    foreground_sizes = np.bincount(foreground_labels.ravel())
    keep = foreground_sizes >= min_area
    keep[0] = False
    mask = keep[foreground_labels]
    return np.asarray(mask, dtype=bool)


def _watershed_labels(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    components, component_count = ndi.label(mask)
    if component_count == 0:
        return np.zeros(mask.shape, dtype=np.int32)
    if not bool(config.get("split_touching", True)):
        return components.astype(np.int32, copy=False)

    distance = ndi.distance_transform_edt(mask)
    coordinates = feature.peak_local_max(
        distance,
        labels=mask,
        min_distance=max(1, int(config.get("min_peak_distance_px", 8))),
        threshold_abs=max(
            0.0, float(config.get("watershed_min_peak_height_px", 0.0))
        ),
        exclude_border=False,
    )
    peak_prominence = max(
        0.0, float(config.get("watershed_min_peak_prominence_px", 0.0))
    )
    if coordinates.size and peak_prominence > 0:
        prominent_maxima = morphology.h_maxima(distance, peak_prominence)
        coordinates = coordinates[prominent_maxima[tuple(coordinates.T)]]
    marker_mask = np.zeros(mask.shape, dtype=bool)
    if coordinates.size:
        marker_mask[tuple(coordinates.T)] = True
    markers, _ = ndi.label(marker_mask)

    # Guarantee at least one marker in every connected foreground component.
    present = set(np.unique(components[marker_mask]).tolist()) - {0}
    next_marker = int(markers.max()) + 1
    for component_id in range(1, component_count + 1):
        if component_id in present:
            continue
        component_mask = components == component_id
        flat_index = int(np.argmax(np.where(component_mask, distance, -1.0)))
        y, x = np.unravel_index(flat_index, mask.shape)
        markers[y, x] = next_marker
        next_marker += 1

    return segmentation.watershed(
        -distance,
        markers,
        mask=mask,
        compactness=float(config.get("watershed_compactness", 0.0)),
    ).astype(np.int32, copy=False)


def _local_contrast_ratio(
    region: measure._regionprops.RegionProperties,
    intensity: np.ndarray,
    ring_radius: int,
) -> float:
    min_row, min_col, max_row, max_col = region.bbox
    margin = max(1, ring_radius)
    y0 = max(0, min_row - margin)
    x0 = max(0, min_col - margin)
    y1 = min(intensity.shape[0], max_row + margin)
    x1 = min(intensity.shape[1], max_col + margin)
    local_labels = region.image
    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    offset_y = min_row - y0
    offset_x = min_col - x0
    local_mask[
        offset_y : offset_y + local_labels.shape[0],
        offset_x : offset_x + local_labels.shape[1],
    ] = local_labels
    ring = morphology.dilation(local_mask, morphology.disk(margin)) & ~local_mask
    local_intensity = intensity[y0:y1, x0:x1]
    inside_values = local_intensity[local_mask]
    ring_values = local_intensity[ring]
    inside_mean = float(np.mean(inside_values)) if inside_values.size else 0.0
    ring_mean = float(np.mean(ring_values)) if ring_values.size else 0.0
    return (inside_mean + 1e-8) / (ring_mean + 1e-8)


def _filter_regions(
    labels: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    min_area = max(1, int(config.get("min_area_px", 1)))
    max_area_raw = config.get("max_area_px")
    max_area = float(max_area_raw) if max_area_raw is not None else math.inf
    min_circularity = float(config.get("min_circularity", 0.0))
    min_contrast = float(config.get("min_local_contrast_ratio", 0.0))
    ring_radius = max(1, int(config.get("local_contrast_ring_px", 4)))
    clear_border = bool(config.get("clear_border", False))
    border_margin = max(0, int(config.get("border_exclusion_margin_px", 0)))

    accepted: list[int] = []
    rejected_area = 0
    rejected_shape = 0
    rejected_contrast = 0
    rejected_border = 0
    for region in measure.regionprops(labels, intensity_image=intensity):
        min_row, min_col, max_row, max_col = region.bbox
        if clear_border and (
            min_row <= border_margin
            or min_col <= border_margin
            or max_row >= labels.shape[0] - border_margin
            or max_col >= labels.shape[1] - border_margin
        ):
            rejected_border += 1
            continue
        if region.area < min_area or region.area > max_area:
            rejected_area += 1
            continue
        perimeter = float(region.perimeter)
        circularity = (
            4.0 * math.pi * float(region.area) / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )
        if circularity < min_circularity:
            rejected_shape += 1
            continue
        contrast = _local_contrast_ratio(region, intensity, ring_radius)
        if contrast < min_contrast:
            rejected_contrast += 1
            continue
        accepted.append(int(region.label))

    filtered = np.zeros(labels.shape, dtype=np.int32)
    for new_id, old_id in enumerate(accepted, start=1):
        filtered[labels == old_id] = new_id
    return filtered, {
        "accepted": len(accepted),
        "rejected_area": rejected_area,
        "rejected_shape": rejected_shape,
        "rejected_contrast": rejected_contrast,
        "rejected_border": rejected_border,
    }


def segment_cells(
    intensity_image: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Segment cells from one selected raw-intensity analysis channel."""

    labels, diagnostics, steps = segment_cells_steps(intensity_image, config)
    return labels, steps["cleaned_mask"], diagnostics


def segment_cells_steps(
    intensity_image: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Segment cells and retain threshold, cleanup, and watershed stages."""

    initial_mask, threshold = threshold_image(
        intensity_image,
        method=config.get("threshold_method", "otsu"),
        percentile=float(config.get("threshold_percentile", 90.0)),
        adaptive_block_size_px=int(config.get("adaptive_block_size_px", 51)),
        adaptive_offset=float(config.get("adaptive_offset", 0.0)),
        threshold_scale=float(config.get("threshold_scale", 1.0)),
        invert=bool(config.get("invert", False)),
    )
    cleaned_mask = _morphological_cleanup(initial_mask, config)
    connected_component_count = int(ndi.label(cleaned_mask)[1])
    candidate_labels = _watershed_labels(cleaned_mask, config)
    labels, filter_counts = _filter_regions(candidate_labels, intensity_image, config)
    diagnostics: dict[str, Any] = {
        "threshold": threshold,
        "threshold_method": config.get("threshold_method", "otsu"),
        "initial_foreground_pixels": int(initial_mask.sum()),
        "cleaned_foreground_pixels": int(cleaned_mask.sum()),
        "connected_component_count": connected_component_count,
        "candidate_count": int(candidate_labels.max()),
        **filter_counts,
    }
    steps = {
        "initial_threshold_mask": initial_mask,
        "cleaned_mask": cleaned_mask,
        "candidate_labels": candidate_labels,
        "accepted_labels": labels,
    }
    return labels, diagnostics, steps
