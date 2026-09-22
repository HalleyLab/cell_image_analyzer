"""Cellular Aβ and neighborhood Iba1/CD68 fluorescence measurements."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import feature, measure, morphology, segmentation, transform

from cell_analyzer.image_io import configure_cache_directory, inspect_image, read_image_channels
from cell_analyzer.preprocessing import preprocess_channel
from cell_analyzer.segmentation import threshold_image

from .config import load_config, normalize_config, save_config
from .advanced import ROLES, analyze_relationships
from .advanced_features import ADVANCED_TABLES, analyze_features
from .plots import _display_scale, _tint, _overlay, _selected_qc_panels, save_qc as _save_qc


ProgressCallback = Callable[[str], None]


def _export_table(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    reverse: bool = False,
) -> pd.DataFrame:
    """Translate legacy engine tokens to user-facing channel and object names."""

    role_names = {
        role: re.sub(
            r"[^a-z0-9]+",
            "_",
            str(config["channels"][role]["alias"]).lower(),
        ).strip("_")
        for role in config["channels"]
    }
    replacements = {
        **role_names,
        "plaque": "reference_object",
        "microglia": "cell",
        "neuron": "excluded_object",
    }
    if reverse:
        replacements = {value: key for key, value in replacements.items()}

    def rename(name: str) -> str:
        result = str(name)
        for old, new in sorted(replacements.items(), key=lambda pair: len(pair[0]), reverse=True):
            result = re.sub(
                rf"(^|_){re.escape(old)}(?=_|$)",
                lambda match: f"{match.group(1)}{new}",
                result,
            )
        return result

    result = frame.copy()
    result.columns = [rename(column) for column in result.columns]
    if "marker" in result.columns:
        values = (
            {config["channels"][role]["alias"]: role for role in role_names}
            if reverse
            else {role: config["channels"][role]["alias"] for role in role_names}
        )
        result["marker"] = result["marker"].replace(values)
    return result


def _select_output_columns(
    frame: pd.DataFrame, config: dict[str, Any], table_name: str
) -> pd.DataFrame:
    """Keep the exact user-selected columns; a missing selection means all columns."""

    selections = config.get("output", {}).get("table_columns", {})
    if table_name not in selections:
        return frame
    requested = [
        column if column in frame.columns else column.replace("primary_object", "reference_object")
        for column in selections[table_name]
    ]
    columns = [column for column in requested if column in frame.columns]
    return frame.loc[:, columns]


@dataclass
class AnalysisProducts:
    """Tables, masks, and thresholds produced from one image."""

    image_summary: pd.DataFrame
    plaque_measurements: pd.DataFrame
    abeta_candidate_qc: pd.DataFrame
    marker_component_qc: pd.DataFrame
    microglia_cells: pd.DataFrame
    tissue_mask: np.ndarray
    plaque_labels: np.ndarray
    neuron_like_mask: np.ndarray
    microglia_absent_plaque_mask: np.ndarray
    marker_labels: dict[str, np.ndarray]
    nucleus_labels: np.ndarray
    microglia_labels: np.ndarray
    positive_masks: dict[str, np.ndarray]
    ring_masks: dict[str, np.ndarray]
    thresholds: dict[str, float | None]
    neighbour_enabled: bool = True
    stages: dict = field(default_factory=dict)
    advanced_maps: dict = field(default_factory=dict)
    advanced_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    object_distances: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_tables: dict = field(default_factory=dict)


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _ring_name(inner: float, outer: float) -> str:
    def clean(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    return f"ring_{clean(inner)}_{clean(outer)}um"


def _ring_specs(edges_um: list[float]) -> list[tuple[str, float]]:
    return [(_ring_name(0, outer), float(outer)) for outer in edges_um[1:]]


def _load_mask(path: str | Path, target_shape: tuple[int, int], invert: bool) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Tissue ROI mask not found: {source}")
    if source.suffix.casefold() in {".tif", ".tiff"}:
        array = np.asarray(tifffile.imread(source))
    else:
        with Image.open(source) as image:
            array = np.asarray(image)
    while array.ndim > 2:
        if array.shape[-1] in {3, 4}:
            array = np.any(array[..., :3] > 0, axis=-1)
        else:
            array = array[0]
    if array.shape != target_shape:
        array = transform.resize(
            array.astype(float),
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
    mask = np.asarray(array > 0, dtype=bool)
    if invert:
        mask = ~mask
    if not np.any(mask):
        raise ValueError("The tissue ROI mask contains no positive pixels.")
    return mask


def _tissue_mask(config: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    roi = config["tissue_roi"]
    if roi["mode"] == "full_image":
        return np.ones(shape, dtype=bool)
    if roi["mode"] == "mask_directory":
        raise ValueError("mask_directory must be resolved to one mask by batch analysis.")
    return _load_mask(roi["mask_path"], shape, bool(roi.get("invert_mask", False)))


def _threshold_channels(
    images: dict[str, np.ndarray],
    config: dict[str, Any],
    tissue_mask: np.ndarray,
    stages: dict | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float | None],
]:
    raw_images: dict[str, np.ndarray] = {}
    analysis_images: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    thresholds: dict[str, float | None] = {}
    for position, role in enumerate(images, 1):
        processed = preprocess_channel(images[role], config["channels"][role])
        threshold_config = config["channels"][role]["threshold"]
        mask, threshold = threshold_image(
            processed.analysis_image,
            method=threshold_config["method"],
            percentile=threshold_config["percentile"],
            threshold_scale=threshold_config["scale"],
            manual_threshold=threshold_config["value"],
        )
        raw_images[role] = np.asarray(processed.raw)
        analysis_images[role] = np.asarray(processed.analysis_image)
        masks[role] = np.asarray(mask & tissue_mask, dtype=bool)
        thresholds[role] = threshold
        if stages is not None:
            prefix = f"stage_channel_{position}"
            stages[prefix + "_gaussian"] = ("image", processed.analysis_image, role)
            stages[prefix + "_threshold"] = ("mask", masks[role], role)
    return raw_images, analysis_images, masks, thresholds


def _filter_marker_components(
    mask: np.ndarray,
    object_filter: dict[str, Any],
    pixel_area_um2: float,
    role: str,
    channel_position: int | None = None,
    intensity_image: np.ndarray | None = None,
    stages: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Clean and filter connected channel objects using transparent rules."""

    cleaned = np.asarray(mask, dtype=bool)
    enabled = bool(object_filter.get("enabled", True))
    if enabled:
        opening = max(0, int(object_filter.get("opening_radius_px", 0)))
        closing = max(0, int(object_filter.get("closing_radius_px", 0)))
        if opening:
            cleaned = morphology.opening(cleaned, morphology.disk(opening))
        if closing:
            cleaned = morphology.closing(cleaned, morphology.disk(closing))
        if bool(object_filter.get("fill_holes", False)):
            cleaned = ndi.binary_fill_holes(cleaned)
        else:
            maximum_hole_um2 = max(
                0.0, float(object_filter.get("max_hole_area_um2", 0.0))
            )
            if maximum_hole_um2 > 0:
                maximum_hole_px = max(
                    1, int(math.ceil(maximum_hole_um2 / pixel_area_um2))
                )
                cleaned = morphology.remove_small_holes(
                    cleaned, area_threshold=maximum_hole_px
                )

    candidates = measure.label(cleaned, connectivity=2)
    if stages is not None:
        position = channel_position or list(ROLES).index(role) + 1
        prefix = f"stage_channel_{position}"
        stages[prefix + "_morphology"] = ("mask", cleaned, role)
        stages[prefix + "_candidates"] = ("labels", candidates, role)
    minimum_area_px = max(
        1,
        int(
            math.ceil(
                max(0.0, float(object_filter.get("min_area_um2", 0.0)))
                / pixel_area_um2
            )
        ),
    )
    maximum_area_um2 = object_filter.get("max_area_um2")
    maximum_area_px = (
        int(math.floor(float(maximum_area_um2) / pixel_area_um2))
        if maximum_area_um2 not in {None, ""}
        else None
    )
    accepted_ids: list[int] = []
    records: list[dict[str, Any]] = []
    intensity = None if intensity_image is None else np.asarray(intensity_image, dtype=float)
    for region in measure.regionprops(candidates):
        perimeter_px = float(region.perimeter)
        solidity = float(region.solidity)
        eccentricity = float(region.eccentricity)
        circularity = (
            4.0 * math.pi * float(region.area) / (perimeter_px * perimeter_px)
            if perimeter_px > 0
            else 0.0
        )
        circularity = min(1.0, max(0.0, circularity))
        reasons: list[str] = []
        if enabled:
            if region.area < minimum_area_px:
                reasons.append("below_min_area")
            if maximum_area_px is not None and region.area > maximum_area_px:
                reasons.append("above_max_area")
            if circularity < float(object_filter.get("min_circularity", 0.0)):
                reasons.append("below_min_circularity")
            if solidity < float(object_filter.get("min_solidity", 0.0)):
                reasons.append("below_min_solidity")
            if eccentricity > float(
                object_filter.get("max_eccentricity", 1.0)
            ):
                reasons.append("above_max_eccentricity")
        accepted = not reasons
        if accepted:
            accepted_ids.append(int(region.label))
        values = (
            intensity[region.coords[:, 0], region.coords[:, 1]]
            if intensity is not None
            else np.asarray([], dtype=float)
        )
        area_um2 = float(region.area) * pixel_area_um2
        records.append(
            {
                "marker": role,
                "component_id": int(region.label),
                "accepted": accepted,
                "exclusion_reason": ";".join(reasons),
                "centroid_y_px": float(region.centroid[0]),
                "centroid_x_px": float(region.centroid[1]),
                "component_area_um2": area_um2,
                "component_equivalent_diameter_um": 2.0
                * math.sqrt(area_um2 / math.pi),
                "component_perimeter_px": perimeter_px,
                "component_circularity": circularity,
                "component_solidity": solidity,
                "component_eccentricity": eccentricity,
                "component_mean_intensity": (
                    float(values.mean()) if values.size else float("nan")
                ),
                "component_max_intensity": (
                    float(values.max()) if values.size else float("nan")
                ),
            }
        )
    label_map = np.zeros(int(candidates.max()) + 1, dtype=np.int32)
    label_map[accepted_ids] = np.arange(1, len(accepted_ids) + 1, dtype=np.int32)
    labels = label_map[candidates]
    table = pd.DataFrame.from_records(records)
    if table.empty:
        table = pd.DataFrame(
            columns=[
                "marker",
                "component_id",
                "accepted",
                "exclusion_reason",
                "centroid_y_px",
                "centroid_x_px",
                "component_area_um2",
                "component_equivalent_diameter_um",
                "component_perimeter_px",
                "component_circularity",
                "component_solidity",
                "component_eccentricity",
                "component_mean_intensity",
                "component_max_intensity",
            ]
        )
    if stages is not None:
        stages[prefix + "_accepted"] = ("mask", labels > 0, role)
        stages[prefix + "_rejected"] = ("mask", cleaned & (labels == 0), role)
    return labels > 0, labels, table


def _plaque_candidate_features(
    region: Any,
    intensity_image: np.ndarray | None,
    pixel_area_um2: float,
) -> dict[str, Any]:
    area_um2 = float(region.area) * pixel_area_um2
    perimeter_px = float(region.perimeter)
    circularity = (
        4.0 * math.pi * float(region.area) / (perimeter_px * perimeter_px)
        if perimeter_px > 0
        else 0.0
    )
    circularity = min(1.0, max(0.0, circularity))
    component = np.asarray(region.image, dtype=bool)
    filled = ndi.binary_fill_holes(component)
    filled_area = int(filled.sum())
    hole_fraction = (
        float(filled_area - int(component.sum())) / filled_area if filled_area else 0.0
    )
    filled_labels = measure.label(filled, connectivity=2)
    filled_regions = measure.regionprops(filled_labels)
    if filled_regions:
        filled_region = filled_regions[0]
        outer_perimeter = float(filled_region.perimeter)
        outer_circularity = (
            4.0 * math.pi * float(filled_region.area)
            / (outer_perimeter * outer_perimeter)
            if outer_perimeter > 0
            else 0.0
        )
        outer_solidity = float(filled_region.solidity)
    else:
        outer_circularity = 0.0
        outer_solidity = 0.0
    outer_circularity = min(1.0, max(0.0, outer_circularity))
    center_shell_ratio = float("nan")
    intensity_mean = float("nan")
    intensity_max = float("nan")
    if intensity_image is not None:
        rows = region.coords[:, 0]
        columns = region.coords[:, 1]
        values = np.asarray(intensity_image, dtype=float)[rows, columns]
        if values.size:
            intensity_mean = float(values.mean())
            intensity_max = float(values.max())
        min_row, min_column, max_row, max_column = region.bbox
        crop = np.asarray(
            intensity_image[min_row:max_row, min_column:max_column], dtype=float
        )
        distance_inside = ndi.distance_transform_edt(filled)
        maximum_distance = float(distance_inside.max())
        if maximum_distance > 0:
            center = filled & (distance_inside >= 0.55 * maximum_distance)
            shell = (
                filled
                & (distance_inside >= 0.15 * maximum_distance)
                & (distance_inside < 0.45 * maximum_distance)
            )
            if np.any(center) and np.any(shell):
                shell_mean = float(crop[shell].mean())
                if shell_mean > 0:
                    center_shell_ratio = float(crop[center].mean() / shell_mean)
    return {
        "candidate_id": int(region.label),
        "centroid_y_px": float(region.centroid[0]),
        "centroid_x_px": float(region.centroid[1]),
        "candidate_area_um2": area_um2,
        "candidate_equivalent_diameter_um": 2.0 * math.sqrt(area_um2 / math.pi),
        "candidate_circularity": circularity,
        "candidate_solidity": float(region.solidity),
        "candidate_outer_circularity": outer_circularity,
        "candidate_outer_solidity": outer_solidity,
        "candidate_eccentricity": float(region.eccentricity),
        "candidate_hole_fraction": hole_fraction,
        "candidate_center_shell_intensity_ratio": center_shell_ratio,
        "candidate_mean_abeta_intensity": intensity_mean,
        "candidate_max_abeta_intensity": intensity_max,
    }


def _segment_microglia_nuclei(
    nucleus_mask: np.ndarray,
    confirmation_mask: np.ndarray | None,
    nucleus_image: np.ndarray,
    tissue_mask: np.ndarray,
    config: dict[str, Any],
    *,
    nucleus_role: str,
    confirmation_role: str | None,
    pixel_area_um2: float,
    pixel_size_um_x: float,
    pixel_size_um_y: float,
    plaque_distance_um: np.ndarray,
    nearest_plaque_labels: np.ndarray,
    ring_edges_um: list[float],
    stages: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Identify nuclei and optionally retain only confirmation-positive cells."""

    settings = config["microglia_count"]
    cleaned = np.asarray(nucleus_mask & tissue_mask, dtype=bool)
    opening = int(settings["opening_radius_px"])
    closing = int(settings["closing_radius_px"])
    if opening:
        cleaned = morphology.opening(cleaned, morphology.disk(opening))
    if closing:
        cleaned = morphology.closing(cleaned, morphology.disk(closing))
    if bool(settings["fill_holes"]):
        cleaned = ndi.binary_fill_holes(cleaned)

    components = measure.label(cleaned, connectivity=2)
    if stages is not None:
        stages["stage_nucleus_cleaned"] = ("mask", cleaned, nucleus_role)
    if bool(settings["split_touching"]) and np.any(cleaned):
        nucleus_distance = ndi.distance_transform_edt(cleaned)
        coordinates = feature.peak_local_max(
            nucleus_distance,
            labels=cleaned,
            min_distance=int(settings["min_peak_distance_px"]),
            exclude_border=False,
        )
        markers = np.zeros(cleaned.shape, dtype=np.int32)
        for marker_id, (row, column) in enumerate(coordinates, start=1):
            markers[row, column] = marker_id
        next_marker = int(markers.max()) + 1
        for component_id in range(1, int(components.max()) + 1):
            component_mask = components == component_id
            if np.any(markers[component_mask]):
                continue
            flat_index = int(np.argmax(np.where(component_mask, nucleus_distance, -1.0)))
            row, column = np.unravel_index(flat_index, cleaned.shape)
            markers[row, column] = next_marker
            next_marker += 1
        candidates = segmentation.watershed(-nucleus_distance, markers, mask=cleaned)
        if stages is not None:
            stages["stage_nucleus_distance"] = ("heatmap", nucleus_distance, nucleus_role)
            stages["stage_nucleus_seeds"] = ("labels", markers, nucleus_role)
    else:
        candidates = components

    minimum_area_px = max(
        1, int(math.ceil(float(settings["min_nucleus_area_um2"]) / pixel_area_um2))
    )
    maximum_area = settings.get("max_nucleus_area_um2")
    maximum_area_px = (
        int(math.floor(float(maximum_area) / pixel_area_um2))
        if maximum_area is not None
        else None
    )
    mean_pixel_um = (pixel_size_um_x + pixel_size_um_y) / 2.0
    radius_px = max(
        0, int(math.ceil(float(settings["perinuclear_radius_um"]) / mean_pixel_um))
    )
    footprint = morphology.disk(radius_px) if radius_px else None
    nucleus_map = np.zeros(int(candidates.max()) + 1, dtype=np.int32)
    microglia_map = np.zeros(int(candidates.max()) + 1, dtype=np.int32)
    records: list[dict[str, Any]] = []
    nucleus_id = 0
    cell_id = 0

    for region in measure.regionprops(candidates, intensity_image=nucleus_image):
        area_um2 = float(region.area) * pixel_area_um2
        perimeter = float(region.perimeter)
        solidity = float(region.solidity)
        eccentricity = float(region.eccentricity)
        circularity = (
            4.0 * math.pi * float(region.area) / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )
        circularity = min(1.0, max(0.0, circularity))
        reasons: list[str] = []
        if region.area < minimum_area_px:
            reasons.append("below_min_nucleus_area")
        if maximum_area_px is not None and region.area > maximum_area_px:
            reasons.append("above_max_nucleus_area")
        if circularity < float(settings["min_circularity"]):
            reasons.append("below_min_circularity")
        if solidity < float(settings["min_solidity"]):
            reasons.append("below_min_solidity")
        if eccentricity > float(settings["max_eccentricity"]):
            reasons.append("above_max_eccentricity")
        accepted_nucleus = not reasons

        min_row, min_column, max_row, max_column = region.bbox
        crop = (
            slice(max(0, min_row - radius_px), min(cleaned.shape[0], max_row + radius_px)),
            slice(max(0, min_column - radius_px), min(cleaned.shape[1], max_column + radius_px)),
        )
        local_nucleus = candidates[crop] == int(region.label)
        local_zone = (
            morphology.dilation(local_nucleus, footprint)
            if footprint is not None
            else local_nucleus
        )
        local_zone &= tissue_mask[crop]
        if confirmation_mask is None:
            confirmation_area_um2 = float("nan")
            confirmation_fraction = float("nan")
            accepted_microglia = accepted_nucleus
        else:
            local_confirmation = confirmation_mask[crop] & local_zone
            confirmation_area_um2 = int(local_confirmation.sum()) * pixel_area_um2
            confirmation_fraction = _safe_ratio(
                int(local_confirmation.sum()), int(local_zone.sum())
            )
            accepted_microglia = bool(
                accepted_nucleus
                and confirmation_fraction
                >= float(settings["min_confirmation_positive_fraction"])
            )
        if accepted_nucleus:
            nucleus_id += 1
            nucleus_map[int(region.label)] = nucleus_id
            if not accepted_microglia:
                reasons.append("below_min_confirmation_fraction")
        if accepted_microglia:
            cell_id += 1
            microglia_map[int(region.label)] = cell_id

        row = int(np.clip(round(region.centroid[0]), 0, cleaned.shape[0] - 1))
        column = int(np.clip(round(region.centroid[1]), 0, cleaned.shape[1] - 1))
        nearest_plaque_id = int(nearest_plaque_labels[row, column])
        plaque_distance = float(plaque_distance_um[row, column])
        ring_name = ""
        if accepted_microglia and nearest_plaque_id > 0:
            for name, outer in _ring_specs(ring_edges_um):
                if plaque_distance <= outer:
                    ring_name = name
                    break
        records.append(
            {
                "candidate_nucleus_id": int(region.label),
                "nucleus_id": nucleus_id if accepted_nucleus else None,
                "microglia_cell_id": cell_id if accepted_microglia else None,
                "accepted_nucleus": accepted_nucleus,
                "accepted_microglia": accepted_microglia,
                "nucleus_channel": nucleus_role,
                "confirmation_channel": confirmation_role or "",
                "exclusion_reason": ";".join(reasons),
                "centroid_y_px": float(region.centroid[0]),
                "centroid_x_px": float(region.centroid[1]),
                "centroid_y_um": float(region.centroid[0]) * pixel_size_um_y,
                "centroid_x_um": float(region.centroid[1]) * pixel_size_um_x,
                "nucleus_area_um2": area_um2,
                "nucleus_circularity": circularity,
                "nucleus_solidity": solidity,
                "nucleus_eccentricity": eccentricity,
                "nucleus_mean_intensity": float(region.intensity_mean),
                "perinuclear_area_um2": int(local_zone.sum()) * pixel_area_um2,
                "perinuclear_confirmation_positive_area_um2": confirmation_area_um2,
                "perinuclear_confirmation_positive_fraction": confirmation_fraction,
                "nearest_plaque_id": nearest_plaque_id,
                "distance_to_nearest_plaque_um": plaque_distance,
                "distance_ring": ring_name,
            }
        )

    columns = [
        "candidate_nucleus_id", "nucleus_id", "microglia_cell_id",
        "accepted_nucleus", "accepted_microglia", "nucleus_channel",
        "confirmation_channel", "exclusion_reason", "centroid_y_px",
        "centroid_x_px", "centroid_y_um", "centroid_x_um", "nucleus_area_um2",
        "nucleus_circularity", "nucleus_solidity", "nucleus_eccentricity",
        "nucleus_mean_intensity", "perinuclear_area_um2",
        "perinuclear_confirmation_positive_area_um2",
        "perinuclear_confirmation_positive_fraction", "nearest_plaque_id",
        "distance_to_nearest_plaque_um", "distance_ring",
    ]
    nucleus_labels = nucleus_map[candidates]
    microglia_labels = microglia_map[candidates]
    table = pd.DataFrame.from_records(records)
    if stages is not None:
        stages["stage_nucleus_candidates"] = ("labels", candidates, nucleus_role)
        stages["stage_nucleus_accepted"] = ("labels", nucleus_labels, nucleus_role)
        stages["stage_cell_accepted"] = ("labels", microglia_labels, nucleus_role)
    return nucleus_labels, microglia_labels, table if not table.empty else pd.DataFrame(columns=columns)


def _is_neuron_like_candidate(features: dict[str, Any], plaque_config: dict[str, Any]) -> bool:
    """Classify soma-like Aβ objects without using Iba1 or CD68 signals."""

    mode = str(
        plaque_config.get("neuron_exclusion_mode", "shape_and_dark_center")
    ).lower()
    if mode == "off":
        return False
    diameter = float(features["candidate_equivalent_diameter_um"])
    shape_match = (
        float(plaque_config.get("neuron_min_diameter_um", 8.0))
        <= diameter
        <= float(plaque_config.get("neuron_max_diameter_um", 28.0))
        and float(features["candidate_outer_circularity"])
        >= float(plaque_config.get("neuron_min_circularity", 0.65))
        and float(features["candidate_outer_solidity"])
        >= float(plaque_config.get("neuron_min_solidity", 0.78))
    )
    if mode == "shape":
        return bool(shape_match)
    ratio = float(features["candidate_center_shell_intensity_ratio"])
    dark_center = (
        float(features["candidate_hole_fraction"])
        >= float(plaque_config.get("neuron_min_hole_fraction", 0.08))
        or (
            np.isfinite(ratio)
            and ratio <= float(plaque_config.get("neuron_max_center_shell_ratio", 0.90))
        )
    )
    return bool(shape_match and dark_center)


def _detect_neuron_like_regions(
    detection_image: np.ndarray | None,
    abeta_threshold: float | None,
    plaque_config: dict[str, Any],
    pixel_area_um2: float,
    domain_mask: np.ndarray,
) -> np.ndarray:
    """Detect continuous soma-like outlines at a lower Aβ threshold."""

    detected = np.zeros(domain_mask.shape, dtype=bool)
    if (
        detection_image is None
        or abeta_threshold is None
        or str(plaque_config.get("neuron_exclusion_mode", "off")).lower() == "off"
    ):
        return detected
    threshold_scale = float(
        plaque_config.get("neuron_detection_threshold_scale", 0.75)
    )
    low_mask = (
        np.asarray(detection_image, dtype=float)
        >= float(abeta_threshold) * threshold_scale
    ) & domain_mask
    closing = max(1, int(plaque_config.get("closing_radius_px", 1)))
    low_mask = morphology.closing(low_mask, morphology.disk(closing))
    low_labels = measure.label(low_mask, connectivity=2)
    for region in measure.regionprops(low_labels):
        features = _plaque_candidate_features(
            region,
            np.asarray(detection_image),
            pixel_area_um2,
        )
        if not _is_neuron_like_candidate(features, plaque_config):
            continue
        min_row, min_column, max_row, max_column = region.bbox
        filled = ndi.binary_fill_holes(np.asarray(region.image, dtype=bool))
        detected[min_row:max_row, min_column:max_column] |= filled
    return detected


def _segment_plaque_candidates(
    mask: np.ndarray,
    config: dict[str, Any],
    pixel_area_um2: float,
    intensity_image: np.ndarray | None = None,
    detection_image: np.ndarray | None = None,
    abeta_threshold: float | None = None,
    domain_mask: np.ndarray | None = None,
    stages: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    plaque_config = config["plaque"]
    cleaned = np.asarray(mask, dtype=bool)
    opening = int(plaque_config["opening_radius_px"])
    closing = int(plaque_config["closing_radius_px"])
    if opening:
        cleaned = morphology.opening(cleaned, morphology.disk(opening))
    if closing:
        cleaned = morphology.closing(cleaned, morphology.disk(closing))
    if bool(plaque_config.get("fill_holes", False)):
        cleaned = ndi.binary_fill_holes(cleaned)
    else:
        max_hole_area_um2 = float(plaque_config.get("max_hole_area_um2", 0.0))
        if max_hole_area_um2 > 0:
            hole_area_px = max(1, int(math.ceil(max_hole_area_um2 / pixel_area_um2)))
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=hole_area_px)

    if stages is not None:
        stages["stage_reference_cleaned"] = ("mask", cleaned, config["plaque"].get("reference_channel", "abeta"))
    components = measure.label(cleaned, connectivity=2)
    domain = (
        np.ones(cleaned.shape, dtype=bool)
        if domain_mask is None
        else np.asarray(domain_mask, dtype=bool)
    )
    neuron_like_mask = _detect_neuron_like_regions(
        detection_image,
        abeta_threshold,
        plaque_config,
        pixel_area_um2,
        domain,
    )
    candidate_records: list[dict[str, Any]] = []
    for region in measure.regionprops(components):
        record = _plaque_candidate_features(region, intensity_image, pixel_area_um2)
        rows = region.coords[:, 0]
        columns = region.coords[:, 1]
        low_threshold_overlap = bool(np.any(neuron_like_mask[rows, columns]))
        direct_shape_match = _is_neuron_like_candidate(record, plaque_config)
        neuron_like = low_threshold_overlap or direct_shape_match
        record["neuron_like_excluded"] = neuron_like
        record["exclusion_reason"] = (
            "low_threshold_soma_outline"
            if low_threshold_overlap
            else ("neuron_like_shape" if direct_shape_match else "")
        )
        candidate_records.append(record)
        if neuron_like:
            neuron_like_mask[components == int(region.label)] = True
    cleaned = cleaned & ~neuron_like_mask
    if stages is not None:
        stages["stage_reference_after_exclusion"] = ("mask", cleaned, config["plaque"].get("reference_channel", "abeta"))
    components = measure.label(cleaned, connectivity=2)
    if bool(plaque_config.get("split_touching", False)) and np.any(cleaned):
        distance = ndi.distance_transform_edt(cleaned)
        coordinates = feature.peak_local_max(
            distance,
            labels=cleaned,
            min_distance=max(1, int(plaque_config.get("min_peak_distance_px", 8))),
            threshold_abs=max(
                0.0, float(plaque_config.get("watershed_min_peak_height_px", 0.0))
            ),
            exclude_border=False,
        )
        markers = np.zeros(cleaned.shape, dtype=np.int32)
        for marker_id, (row, column) in enumerate(coordinates, start=1):
            markers[row, column] = marker_id
        next_marker = int(markers.max()) + 1
        for component_id in range(1, int(components.max()) + 1):
            component_mask = components == component_id
            if np.any(markers[component_mask] > 0):
                continue
            flat_index = int(np.argmax(np.where(component_mask, distance, -1.0)))
            row, column = np.unravel_index(flat_index, cleaned.shape)
            markers[row, column] = next_marker
            next_marker += 1
        candidates = segmentation.watershed(
            -distance,
            markers,
            mask=cleaned,
            compactness=max(0.0, float(plaque_config.get("watershed_compactness", 0.0))),
        )
    else:
        candidates = components
    if stages is not None:
        reference_role = config["plaque"].get("reference_channel", "abeta")
        stages["stage_reference_watershed"] = ("labels", candidates, reference_role)
        if bool(plaque_config.get("split_touching", False)) and np.any(cleaned):
            stages["stage_reference_distance"] = ("heatmap", distance, reference_role)
            stages["stage_reference_seeds"] = ("labels", markers, reference_role)
    min_area_px = max(1, int(math.ceil(plaque_config["min_area_um2"] / pixel_area_um2)))
    max_area_um2 = plaque_config.get("max_area_um2")
    max_area_px = (
        int(math.floor(float(max_area_um2) / pixel_area_um2))
        if max_area_um2 is not None
        else None
    )
    keep: list[int] = []
    for region in measure.regionprops(candidates):
        if region.area < min_area_px:
            continue
        if max_area_px is not None and region.area > max_area_px:
            continue
        perimeter = float(region.perimeter)
        circularity = (
            4.0 * math.pi * float(region.area) / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )
        if circularity < float(plaque_config.get("min_circularity", 0.0)):
            continue
        if float(region.solidity) < float(plaque_config.get("min_solidity", 0.0)):
            continue
        if float(region.eccentricity) > float(plaque_config.get("max_eccentricity", 1.0)):
            continue
        keep.append(int(region.label))
    label_map = np.zeros(int(candidates.max()) + 1, dtype=np.int32)
    label_map[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    labels = label_map[candidates]
    candidate_qc = pd.DataFrame.from_records(candidate_records)
    if candidate_qc.empty:
        candidate_qc = pd.DataFrame(
            columns=[
                "candidate_id",
                "centroid_y_px",
                "centroid_x_px",
                "candidate_area_um2",
                "candidate_equivalent_diameter_um",
                "candidate_circularity",
                "candidate_solidity",
                "candidate_outer_circularity",
                "candidate_outer_solidity",
                "candidate_eccentricity",
                "candidate_hole_fraction",
                "candidate_center_shell_intensity_ratio",
                "candidate_mean_abeta_intensity",
                "candidate_max_abeta_intensity",
                "neuron_like_excluded",
                "exclusion_reason",
            ]
        )
    return labels > 0, labels, neuron_like_mask, candidate_qc


def _clean_and_label_plaques(
    mask: np.ndarray,
    config: dict[str, Any],
    pixel_area_um2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper used by tests and external callers."""

    plaque_mask, plaque_labels, _, _ = _segment_plaque_candidates(
        mask,
        config,
        pixel_area_um2,
    )
    return plaque_mask, plaque_labels


def _boundary_zone(
    tissue_mask: np.ndarray,
    pixel_size_y_um: float,
    pixel_size_x_um: float,
    margin_um: float,
) -> np.ndarray:
    # Padding makes the outer image edge an explicit boundary even when the
    # selected tissue ROI is the complete rectangular image.
    padded = np.pad(tissue_mask, 1, mode="constant", constant_values=False)
    distance_inside = ndi.distance_transform_edt(
        padded,
        sampling=(pixel_size_y_um, pixel_size_x_um),
    )[1:-1, 1:-1]
    effective_margin = max(margin_um, min(pixel_size_x_um, pixel_size_y_um) * 1.01)
    return tissue_mask & (distance_inside <= effective_margin)


def _plaque_spatial_maps(
    plaque_mask: np.ndarray,
    plaque_labels: np.ndarray,
    tissue_mask: np.ndarray,
    ring_edges_um: list[float],
    pixel_size_um_y: float,
    pixel_size_um_x: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if np.any(plaque_mask):
        distance, indices = ndi.distance_transform_edt(
            ~plaque_mask,
            sampling=(pixel_size_um_y, pixel_size_um_x),
            return_indices=True,
        )
        nearest_labels = plaque_labels[indices[0], indices[1]]
    else:
        distance = np.full(plaque_mask.shape, np.inf, dtype=float)
        nearest_labels = np.zeros(plaque_mask.shape, dtype=np.int32)
    rings = {
        name: (
            tissue_mask
            & (distance <= float(outer))
            & (nearest_labels > 0)
        )
        for name, outer in _ring_specs(ring_edges_um)
    }
    return distance, nearest_labels, rings


def _assign_microglia_plaque_context(
    table: pd.DataFrame,
    distance_um: np.ndarray,
    nearest_labels: np.ndarray,
    ring_edges_um: list[float],
) -> None:
    if table.empty:
        return
    rows = np.clip(
        np.rint(pd.to_numeric(table["centroid_y_px"])).astype(int),
        0,
        distance_um.shape[0] - 1,
    )
    columns = np.clip(
        np.rint(pd.to_numeric(table["centroid_x_px"])).astype(int),
        0,
        distance_um.shape[1] - 1,
    )
    nearest = nearest_labels[rows, columns].astype(int)
    distances = distance_um[rows, columns].astype(float)
    accepted = table["accepted_microglia"].astype(bool).to_numpy()
    ring_names = np.full(len(table), "", dtype=object)
    for name, outer in _ring_specs(ring_edges_um):
        selected = (
            accepted
            & (nearest > 0)
            & (distances <= float(outer))
        )
        ring_names[(ring_names == "") & selected] = name
    table.loc[:, "nearest_plaque_id"] = nearest
    table.loc[:, "distance_to_nearest_plaque_um"] = distances
    table.loc[:, "distance_ring"] = ring_names


def _nearby_microglia_counts(
    plaque_labels: np.ndarray,
    table: pd.DataFrame,
    radius_um: float,
    pixel_size_um_y: float,
    pixel_size_um_x: float,
) -> dict[int, int]:
    counts = {
        int(region.label): 0 for region in measure.regionprops(plaque_labels)
    }
    if table.empty or not counts:
        return counts
    accepted = table["accepted_microglia"].astype(bool).to_numpy()
    if not accepted.any():
        return counts
    rows = np.clip(
        np.rint(pd.to_numeric(table["centroid_y_px"])).astype(int),
        0,
        plaque_labels.shape[0] - 1,
    )[accepted]
    columns = np.clip(
        np.rint(pd.to_numeric(table["centroid_x_px"])).astype(int),
        0,
        plaque_labels.shape[1] - 1,
    )[accepted]
    cell_points_um = np.column_stack(
        (rows * pixel_size_um_y, columns * pixel_size_um_x)
    )
    for region in measure.regionprops(plaque_labels):
        plaque_points_um = region.coords.astype(float)
        plaque_points_um[:, 0] *= pixel_size_um_y
        plaque_points_um[:, 1] *= pixel_size_um_x
        distances, _ = cKDTree(plaque_points_um).query(cell_points_um)
        counts[int(region.label)] = int((distances <= radius_um).sum())
    return counts


def _mask_metrics(
    region_mask: np.ndarray,
    positive_masks: dict[str, np.ndarray],
    raw_images: dict[str, np.ndarray],
    pixel_area_um2: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    region_pixels = int(region_mask.sum())
    result["area_um2"] = region_pixels * pixel_area_um2
    for role in positive_masks:
        selected = region_mask & positive_masks[role]
        count = int(selected.sum())
        intensity_sum = float(np.asarray(raw_images[role], dtype=float)[selected].sum())
        result[f"{role}_positive_area_um2"] = count * pixel_area_um2
        result[f"{role}_positive_fraction"] = _safe_ratio(count, region_pixels)
        result[f"{role}_thresholded_integrated_intensity"] = intensity_sum
        result[f"{role}_thresholded_mean_intensity"] = _safe_ratio(intensity_sum, region_pixels)
        result[f"{role}_positive_mean_intensity"] = (
            float(np.asarray(raw_images[role], dtype=float)[selected].mean())
            if count
            else float("nan")
        )
    intersections = {}
    if {"cd68", "iba1"} <= positive_masks.keys():
        intersections["cd68_in_iba1"] = positive_masks["cd68"] & positive_masks["iba1"]
    if {"abeta", "cd68"} <= positive_masks.keys():
        intersections["abeta_in_cd68"] = positive_masks["abeta"] & positive_masks["cd68"]
    if {"abeta", "iba1"} <= positive_masks.keys():
        intersections["abeta_in_iba1"] = positive_masks["abeta"] & positive_masks["iba1"]
    for name, intersection in intersections.items():
        count = int((region_mask & intersection).sum())
        result[f"{name}_area_um2"] = count * pixel_area_um2
    if "cd68_in_iba1" in intersections:
        result["cd68_in_iba1_fraction_of_iba1"] = _safe_ratio(
            result["cd68_in_iba1_area_um2"], result["iba1_positive_area_um2"]
        )
    if "abeta_in_cd68" in intersections:
        result["abeta_in_cd68_fraction_of_cd68"] = _safe_ratio(
            result["abeta_in_cd68_area_um2"], result["cd68_positive_area_um2"]
        )
        result["abeta_in_cd68_fraction_of_abeta"] = _safe_ratio(
            result["abeta_in_cd68_area_um2"], result["abeta_positive_area_um2"]
        )
    if "abeta_in_iba1" in intersections:
        result["abeta_in_iba1_fraction_of_iba1"] = _safe_ratio(
            result["abeta_in_iba1_area_um2"], result["iba1_positive_area_um2"]
        )
        result["abeta_in_iba1_fraction_of_abeta"] = _safe_ratio(
            result["abeta_in_iba1_area_um2"], result["abeta_positive_area_um2"]
        )
    return result


def _labeled_mask_metrics(
    labels: np.ndarray,
    label_count: int,
    positive_masks: dict[str, np.ndarray],
    raw_images: dict[str, np.ndarray],
    pixel_area_um2: float,
    *,
    region_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute every object''s mask metrics with one image scan per channel."""

    flat_labels = np.asarray(labels).ravel()
    active = flat_labels > 0
    if region_mask is not None:
        active &= np.asarray(region_mask, dtype=bool).ravel()
    size = int(label_count) + 1
    region_pixels = np.bincount(flat_labels[active], minlength=size).astype(float)

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.full(size, np.nan, dtype=float)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return result

    metrics: dict[str, np.ndarray] = {
        "area_um2": region_pixels * pixel_area_um2
    }
    positive_counts: dict[str, np.ndarray] = {}
    flat_positive: dict[str, np.ndarray] = {}
    for role, mask in positive_masks.items():
        flat_positive[role] = np.asarray(mask, dtype=bool).ravel()
        selected = active & flat_positive[role]
        selected_labels = flat_labels[selected]
        counts = np.bincount(selected_labels, minlength=size).astype(float)
        intensity = np.asarray(raw_images[role]).ravel()
        sums = np.bincount(
            selected_labels,
            weights=intensity[selected],
            minlength=size,
        )
        positive_counts[role] = counts
        metrics[f"{role}_positive_area_um2"] = counts * pixel_area_um2
        metrics[f"{role}_positive_fraction"] = ratio(counts, region_pixels)
        metrics[f"{role}_thresholded_integrated_intensity"] = sums
        metrics[f"{role}_thresholded_mean_intensity"] = ratio(sums, region_pixels)
        metrics[f"{role}_positive_mean_intensity"] = ratio(sums, counts)

    for name, roles in (
        ("cd68_in_iba1", ("cd68", "iba1")),
        ("abeta_in_cd68", ("abeta", "cd68")),
        ("abeta_in_iba1", ("abeta", "iba1")),
    ):
        if not set(roles) <= flat_positive.keys():
            continue
        selected = active & flat_positive[roles[0]] & flat_positive[roles[1]]
        counts = np.bincount(flat_labels[selected], minlength=size).astype(float)
        metrics[f"{name}_area_um2"] = counts * pixel_area_um2

    if "cd68_in_iba1_area_um2" in metrics:
        metrics["cd68_in_iba1_fraction_of_iba1"] = ratio(
            metrics["cd68_in_iba1_area_um2"],
            metrics["iba1_positive_area_um2"],
        )
    for container in ("cd68", "iba1"):
        key = f"abeta_in_{container}_area_um2"
        if key in metrics:
            metrics[f"abeta_in_{container}_fraction_of_{container}"] = ratio(
                metrics[key], metrics[f"{container}_positive_area_um2"]
            )
            metrics[f"abeta_in_{container}_fraction_of_abeta"] = ratio(
                metrics[key], metrics["abeta_positive_area_um2"]
            )
    return metrics


def _metrics_for_label(
    metrics: dict[str, np.ndarray], label: int
) -> dict[str, float]:
    return {name: float(values[label]) for name, values in metrics.items()}


def _prefixed(target: dict[str, Any], prefix: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        target[f"{prefix}_{key}"] = value


def _plaque_ring_metrics_table(
    plaques: pd.DataFrame, ring_names: list[str]
) -> pd.DataFrame:
    """Return the requested peri-plaque marker metrics in tidy long form."""

    metric_names = (
        "iba1_positive_area_um2",
        "iba1_positive_mean_intensity",
        "cd68_positive_fraction",
        "cd68_positive_mean_intensity",
        "cd68_in_iba1_fraction_of_iba1",
    )
    identity = [
        name
        for name in (
            "source_file",
            "mouse_id",
            "genotype",
            "region",
            "section_id",
            "plaque_id",
        )
        if name in plaques.columns
    ]
    rows: list[dict[str, Any]] = []
    for _, plaque in plaques.iterrows():
        for ring_name in ring_names:
            row = {name: plaque[name] for name in identity}
            row["ring"] = ring_name
            for metric in metric_names:
                row[f"ring_{metric}"] = plaque.get(
                    f"{ring_name}_{metric}", float("nan")
                )
            rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            *identity,
            "ring",
            *(f"ring_{metric}" for metric in metric_names),
        ],
    )


def analyze_arrays(
    images: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    pixel_size_um_x: float,
    pixel_size_um_y: float,
    source_file: str = "",
) -> AnalysisProducts:
    """Analyze one or more configured 2-D fluorescence channels."""

    if not images:
        raise ValueError("images must contain at least one configured channel")
    unknown = set(images) - set(config["channels"])
    if unknown:
        raise ValueError(f"images contain unknown configured channels: {sorted(unknown)}")
    shapes = {np.asarray(value).shape for value in images.values()}
    if len(shapes) != 1:
        raise ValueError("All enabled channels must have identical shapes.")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError("Brain-section analysis expects 2-D arrays or projections.")
    pixel_size_um_x = float(pixel_size_um_x)
    pixel_size_um_y = float(pixel_size_um_y)
    if pixel_size_um_x <= 0 or pixel_size_um_y <= 0:
        raise ValueError("Positive X/Y pixel calibration is required.")
    pixel_area_um2 = pixel_size_um_x * pixel_size_um_y

    tissue = _tissue_mask(config, shape)
    stages = {"stage_tissue": ("mask", tissue, None)}
    raw_images, analysis_images, positive_masks, thresholds = _threshold_channels(
        images, config, tissue, stages
    )
    neighbour_enabled = config.get("advanced", {}).get("neighbour_enabled", True)

    # Every enabled channel is filtered independently after thresholding.
    marker_labels: dict[str, np.ndarray] = {}
    marker_tables: list[pd.DataFrame] = []
    for position, role in enumerate(positive_masks, 1):
        filtered_mask, labels, table = _filter_marker_components(
            positive_masks[role],
            config["channels"][role].get("object_filter", {}),
            pixel_area_um2,
            role,
            channel_position=position,
            intensity_image=raw_images[role],
            stages=stages,
        )
        positive_masks[role] = filtered_mask
        marker_labels[role] = labels
        table.insert(0, "source_file", source_file)
        marker_tables.append(table)
    marker_component_qc = pd.concat(marker_tables, ignore_index=True, sort=False)

    reference_role = config["plaque"].get("reference_channel", "abeta") if neighbour_enabled else next(iter(marker_labels))
    reference_detection_image = analysis_images[reference_role]
    plaque_mask, plaque_labels, neuron_like_mask, candidate_qc = _segment_plaque_candidates(
        positive_masks[reference_role] if neighbour_enabled else np.zeros(shape, dtype=bool),
        config,
        pixel_area_um2,
        intensity_image=raw_images[reference_role],
        detection_image=reference_detection_image if neighbour_enabled else None,
        abeta_threshold=thresholds[reference_role],
        domain_mask=tissue,
        stages=stages if neighbour_enabled else None,
    )
    if neighbour_enabled:
        positive_masks[reference_role] = plaque_mask

    edges = config["spatial"]["ring_edges_um"] if neighbour_enabled else []
    distance, nearest_labels, ring_masks = _plaque_spatial_maps(
        plaque_mask,
        plaque_labels,
        tissue,
        edges,
        pixel_size_um_y,
        pixel_size_um_x,
    )

    microglia_enabled = bool(config["microglia_count"].get("enabled", False))
    nucleus_labels = np.zeros(shape, dtype=np.int32)
    microglia_labels = np.zeros(shape, dtype=np.int32)
    microglia_cells = pd.DataFrame(
        columns=[
            "source_file",
            "microglia_cell_id",
            "accepted_nucleus",
            "accepted_microglia",
            "nearest_plaque_id",
            "distance_to_nearest_plaque_um",
            "distance_ring",
        ]
    )
    if microglia_enabled:
        settings = config["microglia_count"]
        nucleus_role = settings.get("nucleus_channel")
        confirmation_role = settings.get("confirmation_channel")
        if nucleus_role not in marker_labels:
            raise ValueError("The selected nucleus channel is not enabled.")
        if confirmation_role is not None and confirmation_role not in marker_labels:
            raise ValueError("The selected confirmation channel is not enabled.")
        nucleus_labels, microglia_labels, microglia_cells = _segment_microglia_nuclei(
            marker_labels[nucleus_role] > 0,
            marker_labels[confirmation_role] > 0 if confirmation_role else None,
            raw_images[nucleus_role],
            tissue,
            config,
            nucleus_role=nucleus_role,
            confirmation_role=confirmation_role,
            pixel_area_um2=pixel_area_um2,
            pixel_size_um_x=pixel_size_um_x,
            pixel_size_um_y=pixel_size_um_y,
            plaque_distance_um=distance,
            nearest_plaque_labels=nearest_labels,
            ring_edges_um=edges,
            stages=stages,
        )
        microglia_cells.insert(0, "source_file", source_file)

    plaque_count_before_microglia_filter = int(plaque_labels.max())
    microglia_absent_plaque_mask = np.zeros(shape, dtype=bool)
    require_nearby_microglia = neighbour_enabled and bool(
        config["plaque"].get("require_nearby_microglia", False)
    )
    nearby_radius_um = float(
        config["plaque"].get("nearby_microglia_radius_um", 30.0)
    )
    minimum_nearby_microglia = int(
        config["plaque"].get("min_nearby_microglia_count", 1)
    )
    nearby_microglia_counts = (
        _nearby_microglia_counts(
            plaque_labels,
            microglia_cells,
            nearby_radius_um,
            pixel_size_um_y,
            pixel_size_um_x,
        )
        if microglia_enabled
        else {}
    )
    if require_nearby_microglia:
        if not microglia_enabled:
            raise ValueError(
                "Nearby-cell filtering requires cell counting."
            )
        keep = [
            plaque_id
            for plaque_id, count in nearby_microglia_counts.items()
            if count >= minimum_nearby_microglia
        ]
        label_map = np.zeros(int(plaque_labels.max()) + 1, dtype=np.int32)
        label_map[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
        filtered_labels = label_map[plaque_labels]
        nearby_microglia_counts = {
            new_id: nearby_microglia_counts[old_id]
            for new_id, old_id in enumerate(keep, start=1)
        }
        microglia_absent_plaque_mask = plaque_mask & (filtered_labels == 0)
        plaque_labels = filtered_labels
        plaque_mask = plaque_labels > 0
        positive_masks[reference_role] = plaque_mask
        distance, nearest_labels, ring_masks = _plaque_spatial_maps(
            plaque_mask,
            plaque_labels,
            tissue,
            edges,
            pixel_size_um_y,
            pixel_size_um_x,
        )
        _assign_microglia_plaque_context(
            microglia_cells,
            distance,
            nearest_labels,
            edges,
        )

    boundary = _boundary_zone(
        tissue,
        pixel_size_um_y,
        pixel_size_um_x,
        float(config["plaque"].get("boundary_margin_um", 0.0)),
    )
    exclude_boundary = bool(
        config["plaque"].get("exclude_boundary_plaques_from_table", True)
    )
    all_regions = measure.regionprops(
        plaque_labels, intensity_image=raw_images[reference_role]
    )
    boundary_labels = set(
        int(value) for value in np.unique(plaque_labels[boundary]) if value > 0
    )

    summary: dict[str, Any] = {
        "source_file": source_file,
        "image_height_px": int(shape[0]),
        "image_width_px": int(shape[1]),
        "pixel_size_um_x": pixel_size_um_x,
        "pixel_size_um_y": pixel_size_um_y,
        "roi_area_um2": int(tissue.sum()) * pixel_area_um2,
        "roi_area_mm2": int(tissue.sum()) * pixel_area_um2 / 1_000_000.0,
        "plaque_count_all": int(plaque_labels.max()),
        "plaque_count_boundary": len(boundary_labels),
        "plaque_count_interior": int(plaque_labels.max()) - len(boundary_labels),
        "plaque_count_before_nearby_microglia_filter": plaque_count_before_microglia_filter,
        "plaque_without_nearby_microglia_excluded_count": (
            plaque_count_before_microglia_filter - int(plaque_labels.max())
        ),
        "nearby_microglia_filter_enabled": require_nearby_microglia,
        "nearby_microglia_radius_um": nearby_radius_um,
        "min_nearby_microglia_count": minimum_nearby_microglia,
        "plaque_reference_channel": config["channels"][reference_role]["alias"],
        "plaque_candidate_count_before_neuron_exclusion": int(len(candidate_qc)),
        "plaque_neuron_like_excluded_count": int(
            candidate_qc["neuron_like_excluded"].sum()
        ),
        "plaque_neuron_like_excluded_area_um2": float(
            candidate_qc.loc[
                candidate_qc["neuron_like_excluded"], "candidate_area_um2"
            ].sum()
        ),
        "microglia_count_available": microglia_enabled,
    }
    for role in marker_labels:
        role_table = marker_component_qc.loc[marker_component_qc["marker"] == role]
        accepted = role_table["accepted"].astype(bool)
        summary[f"{role}_component_candidate_count"] = int(len(role_table))
        summary[f"{role}_component_accepted_count"] = int(accepted.sum())
        summary[f"{role}_component_excluded_count"] = int((~accepted).sum())
    for key, value in config.get("metadata", {}).items():
        summary[key] = value
    roi_metrics = _mask_metrics(tissue, positive_masks, raw_images, pixel_area_um2)
    for key, value in roi_metrics.items():
        if key == "area_um2":
            continue
        summary[key] = value
    summary["plaque_density_all_per_mm2"] = _safe_ratio(
        summary["plaque_count_all"], summary["roi_area_mm2"]
    )
    summary["plaque_density_interior_per_mm2"] = _safe_ratio(
        summary["plaque_count_interior"], summary["roi_area_mm2"]
    )
    accepted_microglia = (
        microglia_cells["accepted_microglia"].astype(bool)
        if microglia_enabled
        else pd.Series(dtype=bool)
    )
    microglia_nearest = (
        pd.to_numeric(microglia_cells["nearest_plaque_id"], errors="coerce")
        if microglia_enabled
        else pd.Series(dtype=float)
    )
    microglia_distance = (
        pd.to_numeric(
            microglia_cells["distance_to_nearest_plaque_um"], errors="coerce"
        )
        if microglia_enabled
        else pd.Series(dtype=float)
    )
    summary["microglia_count_roi"] = (
        int(accepted_microglia.sum()) if microglia_enabled else float("nan")
    )
    summary["microglia_density_roi_per_mm2"] = (
        _safe_ratio(summary["microglia_count_roi"], summary["roi_area_mm2"])
        if microglia_enabled
        else float("nan")
    )
    for role, threshold in thresholds.items():
        summary[f"{role}_threshold_raw"] = threshold
    for name, outer in _ring_specs(edges):
        ring_mask = ring_masks[name]
        _prefixed(summary, name, _mask_metrics(ring_mask, positive_masks, raw_images, pixel_area_um2))
        ring_cell_count = (
            int(
                (
                    accepted_microglia
                    & (microglia_nearest > 0)
                    & (microglia_distance <= outer)
                ).sum()
            )
            if microglia_enabled
            else float("nan")
        )
        summary[f"{name}_microglia_count"] = ring_cell_count
        ring_area_mm2 = int(ring_mask.sum()) * pixel_area_um2 / 1_000_000.0
        summary[f"{name}_microglia_density_per_mm2"] = (
            _safe_ratio(ring_cell_count, ring_area_mm2)
            if microglia_enabled
            else float("nan")
        )

    plaque_count = int(plaque_labels.max())
    plaque_metric_arrays = _labeled_mask_metrics(
        plaque_labels,
        plaque_count,
        positive_masks,
        raw_images,
        pixel_area_um2,
    )
    ring_metric_arrays = {
        name: _labeled_mask_metrics(
            nearest_labels,
            plaque_count,
            positive_masks,
            raw_images,
            pixel_area_um2,
            region_mask=mask,
        )
        for name, mask in ring_masks.items()
    }
    ring_cell_counts: dict[str, np.ndarray] = {}
    if microglia_enabled:
        accepted_values = accepted_microglia.to_numpy(dtype=bool)
        nearest_values = microglia_nearest.to_numpy(dtype=float)
        distance_values = microglia_distance.to_numpy(dtype=float)
        for name, outer in _ring_specs(edges):
            selected = (
                accepted_values
                & np.isfinite(nearest_values)
                & (nearest_values > 0)
                & (distance_values <= outer)
            )
            ring_cell_counts[name] = np.bincount(
                nearest_values[selected].astype(int), minlength=plaque_count + 1
            )

    plaque_records: list[dict[str, Any]] = []
    mean_pixel_um = (pixel_size_um_x + pixel_size_um_y) / 2.0
    for region in all_regions:
        plaque_id = int(region.label)
        touches_boundary = plaque_id in boundary_labels
        if exclude_boundary and touches_boundary:
            continue
        area_um2 = float(region.area) * pixel_area_um2
        perimeter_um = float(region.perimeter) * mean_pixel_um
        record: dict[str, Any] = {
            "source_file": source_file,
            "plaque_id": plaque_id,
            "touches_tissue_boundary": touches_boundary,
            "centroid_y_px": float(region.centroid[0]),
            "centroid_x_px": float(region.centroid[1]),
            "centroid_y_um": float(region.centroid[0]) * pixel_size_um_y,
            "centroid_x_um": float(region.centroid[1]) * pixel_size_um_x,
            "plaque_area_um2": area_um2,
            "plaque_perimeter_um": perimeter_um,
            "plaque_equivalent_diameter_um": 2.0 * math.sqrt(area_um2 / math.pi),
            "plaque_circularity": (
                4.0 * math.pi * area_um2 / (perimeter_um * perimeter_um)
                if perimeter_um > 0
                else float("nan")
            ),
            "plaque_solidity": float(region.solidity),
            "plaque_eccentricity": float(region.eccentricity),
            f"{reference_role}_mean_intensity_in_plaque": float(region.intensity_mean),
            f"{reference_role}_integrated_intensity_in_plaque": float(
                np.asarray(region.image_intensity)[region.image].sum()
            ),
        }
        for key, value in config.get("metadata", {}).items():
            record[key] = value
        _prefixed(
            record,
            "plaque",
            _metrics_for_label(plaque_metric_arrays, plaque_id),
        )
        record["nearby_microglia_count"] = (
            nearby_microglia_counts.get(plaque_id, 0)
            if microglia_enabled
            else float("nan")
        )
        for name, outer in _ring_specs(edges):
            ring_metrics = _metrics_for_label(ring_metric_arrays[name], plaque_id)
            _prefixed(record, name, ring_metrics)
            plaque_cell_count = (
                int(ring_cell_counts[name][plaque_id])
                if microglia_enabled
                else float("nan")
            )
            record[f"{name}_microglia_count"] = plaque_cell_count
            plaque_ring_area_mm2 = ring_metrics["area_um2"] / 1_000_000.0
            record[f"{name}_microglia_density_per_mm2"] = (
                _safe_ratio(plaque_cell_count, plaque_ring_area_mm2)
                if microglia_enabled
                else float("nan")
            )
        plaque_records.append(record)

    plaques = pd.DataFrame.from_records(plaque_records)
    if plaques.empty:
        plaques = pd.DataFrame(
            columns=[
                "source_file",
                "plaque_id",
                "touches_tissue_boundary",
                "centroid_y_px",
                "centroid_x_px",
                "centroid_y_um",
                "centroid_x_um",
                "plaque_area_um2",
                "plaque_perimeter_um",
                "plaque_equivalent_diameter_um",
                "plaque_circularity",
                "plaque_solidity",
                "plaque_eccentricity",
                f"{reference_role}_mean_intensity_in_plaque",
                f"{reference_role}_integrated_intensity_in_plaque",
                "nearby_microglia_count",
                *config.get("metadata", {}).keys(),
            ]
        )
    if not plaques.empty:
        summary["median_interior_plaque_area_um2"] = float(plaques["plaque_area_um2"].median())
        summary["mean_interior_plaque_area_um2"] = float(plaques["plaque_area_um2"].mean())
    else:
        summary["median_interior_plaque_area_um2"] = float("nan")
        summary["mean_interior_plaque_area_um2"] = float("nan")
    advanced_metrics, object_distances, advanced_maps = pd.DataFrame(), pd.DataFrame(), {}
    feature_tables = {}
    if "advanced" in config:
        scope = config["advanced"]["scope"]
        domain = tissue & ((microglia_labels > 0) if scope == "cells" else (plaque_labels > 0) if scope == "reference_objects" else tissue)
        advanced_metrics, object_distances, advanced_maps = analyze_relationships(
            raw_images, marker_labels, domain, config, pixel_size_um_x, pixel_size_um_y, source_file, thresholds)
        feature_tables, feature_maps, feature_summary = analyze_features(
            raw_images, marker_labels, microglia_labels, domain, config, pixel_size_um_x, pixel_size_um_y, source_file)
        advanced_maps.update(feature_maps)
        summaries = [table for table in (advanced_metrics, feature_summary) if not table.empty]
        advanced_metrics = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    if neighbour_enabled:
        stages["stage_reference_final"] = ("labels", plaque_labels, reference_role)
        stages["stage_reference_distance_um"] = ("heatmap", np.where(tissue, distance, np.nan), reference_role)
    summary["neighbour_analysis_enabled"] = neighbour_enabled
    return AnalysisProducts(
        neighbour_enabled=neighbour_enabled,
        stages=stages,
        advanced_maps=advanced_maps,
        advanced_metrics=advanced_metrics,
        object_distances=object_distances,
        feature_tables=feature_tables,
        image_summary=pd.DataFrame([summary]),
        plaque_measurements=plaques,
        abeta_candidate_qc=candidate_qc,
        marker_component_qc=marker_component_qc,
        microglia_cells=microglia_cells,
        tissue_mask=tissue,
        plaque_labels=plaque_labels,
        neuron_like_mask=neuron_like_mask,
        microglia_absent_plaque_mask=microglia_absent_plaque_mask,
        marker_labels=marker_labels,
        nucleus_labels=nucleus_labels,
        microglia_labels=microglia_labels,
        positive_masks=positive_masks,
        ring_masks=ring_masks,
        thresholds=thresholds,
    )


def _channel_colors(
    info: Any,
    config: dict[str, Any],
    roles: set[str],
) -> dict[str, tuple[float, float, float]]:
    defaults = {
        "abeta": (0.0, 1.0, 0.2),
        "iba1": (1.0, 0.0, 0.0),
        "cd68": (1.0, 0.35, 0.0),
        "dapi": (0.0, 0.2, 1.0),
    }
    result: dict[str, tuple[float, float, float]] = {}
    for role in roles:
        index = int(config["channels"][role]["index"])
        channel = next(
            (item for item in info.channels if int(item.index) == index), None
        )
        text = str(config["channels"][role].get("color") or getattr(channel, "color", "") or "").lstrip("#")
        if len(text) == 8:
            text = text[-6:]
        try:
            result[role] = tuple(
                int(text[offset : offset + 2], 16) / 255.0
                for offset in (0, 2, 4)
            )
        except (TypeError, ValueError):
            result[role] = defaults.get(role, (1.0, 1.0, 1.0))
    return result


def _write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    info: Any,
    products: AnalysisProducts,
    raw_images: dict[str, np.ndarray],
) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_csv = output_dir / "image_summary.csv"
    plaque_csv = output_dir / "neighbour_object_measurements.csv"
    candidate_csv = output_dir / "neighbour_candidate_qc.csv"
    marker_csv = output_dir / "channel_object_qc.csv"
    microglia_csv = output_dir / "cells.csv"
    ring_csv = output_dir / "neighbour_ring_metrics.csv"
    excel_path = output_dir / "cell_analysis_measurements.xlsx"
    ring_table = _plaque_ring_metrics_table(
        products.plaque_measurements, list(products.ring_masks)
    )
    image_table = _export_table(products.image_summary, config)
    object_table = _export_table(products.plaque_measurements, config)
    candidate_table = _export_table(products.abeta_candidate_qc, config)
    channel_table = _export_table(products.marker_component_qc, config)
    cell_table = _export_table(products.microglia_cells, config)
    public_ring_table = _export_table(ring_table, config)
    threshold_table = pd.DataFrame(
        [
            {
                "channel": config["channels"][role]["alias"],
                "threshold_raw": value,
            }
            for role, value in products.thresholds.items()
        ]
    )
    tables = {
        "image_summary": image_table,
        "primary_objects": object_table,
        "candidate_qc": candidate_table,
        "channel_objects": channel_table,
        "cells": cell_table,
        "ring_metrics": public_ring_table,
        "thresholds": threshold_table,
    }
    if not products.advanced_metrics.empty:
        tables["advanced_metrics"] = products.advanced_metrics
        if config["advanced"]["object_distances_enabled"]:
            tables["object_distances"] = products.object_distances
    tables.update(products.feature_tables)
    selected_tables = {
        name: _select_output_columns(table, config, name)
        for name, table in tables.items()
    }
    paths: dict[str, str] = {}
    table_outputs = (
        ("save_image_summary_csv", "image_summary_csv", image_csv, selected_tables["image_summary"]),
        ("save_primary_objects_csv", "plaque_measurements_csv", plaque_csv, selected_tables["primary_objects"]),
        ("save_candidate_qc_csv", "abeta_candidate_qc_csv", candidate_csv, selected_tables["candidate_qc"]),
        ("save_channel_objects_csv", "marker_component_qc_csv", marker_csv, selected_tables["channel_objects"]),
        ("save_cells_csv", "microglia_cells_csv", microglia_csv, selected_tables["cells"]),
        ("save_ring_metrics_csv", "plaque_ring_metrics_csv", ring_csv, selected_tables["ring_metrics"]),
    )
    for name, (_, flag) in ADVANCED_TABLES.items():
        if name in selected_tables:
            table_outputs += ((flag, f"{name}_csv", output_dir / f"{name}.csv", selected_tables[name]),)
    for flag, key, table_path, table in table_outputs:
        if not products.neighbour_enabled and flag in {"save_primary_objects_csv", "save_candidate_qc_csv", "save_ring_metrics_csv"}:
            continue
        if bool(config["output"].get(flag, True)):
            table.to_csv(table_path, index=False)
            paths[key] = str(table_path)
    if bool(config["output"].get("save_excel", True)):
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            selected_tables["image_summary"].to_excel(writer, sheet_name="Image Summary", index=False)
            if products.neighbour_enabled:
                selected_tables["primary_objects"].to_excel(writer, sheet_name="Neighbour Objects", index=False)
            if products.neighbour_enabled:
                selected_tables["candidate_qc"].to_excel(writer, sheet_name="Neighbour QC", index=False)
            selected_tables["channel_objects"].to_excel(writer, sheet_name="Channel Objects", index=False)
            selected_tables["cells"].to_excel(writer, sheet_name="Cells", index=False)
            if products.neighbour_enabled:
                selected_tables["ring_metrics"].to_excel(writer, sheet_name="Object Ring Metrics", index=False)
            selected_tables["thresholds"].to_excel(writer, sheet_name="Thresholds", index=False)
            for name, (title, _) in ADVANCED_TABLES.items():
                if name in selected_tables:
                    selected_tables[name].to_excel(writer, sheet_name=title, index=False)
        paths["excel"] = str(excel_path)

    config_path = output_dir / "config_used.yaml"
    save_config(config, config_path)
    metadata_path = output_dir / "image_metadata.json"
    metadata_path.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
    paths.update(
        {
            "config_used": str(config_path),
            "metadata": str(metadata_path),
        }
    )
    output = config["output"]
    if bool(output.get("save_tissue_mask", True)):
        tissue_path = output_dir / "tissue_roi_mask.tiff"
        tifffile.imwrite(
            tissue_path, products.tissue_mask.astype(np.uint8), compression="zlib"
        )
        paths["tissue_mask"] = str(tissue_path)
    if products.neighbour_enabled and bool(output.get("save_primary_object_labels", True)):
        labels_path = output_dir / "neighbour_reference_labels.tiff"
        tifffile.imwrite(
            labels_path, products.plaque_labels.astype(np.uint32), compression="zlib"
        )
        paths["plaque_labels"] = str(labels_path)
    if products.neighbour_enabled and bool(output.get("save_excluded_object_masks", True)):
        neuron_path = output_dir / "neighbour_excluded_objects.tiff"
        no_microglia_path = output_dir / "neighbour_no_nearby_cell_excluded.tiff"
        tifffile.imwrite(
            neuron_path,
            products.neuron_like_mask.astype(np.uint8),
            compression="zlib",
        )
        tifffile.imwrite(
            no_microglia_path,
            products.microglia_absent_plaque_mask.astype(np.uint8),
            compression="zlib",
        )
        paths.update(
            {
                "abeta_neuron_like_excluded": str(neuron_path),
                "abeta_no_nearby_microglia_excluded": str(no_microglia_path),
            }
        )
    if bool(output.get("save_channel_object_labels", True)):
        marker_labels_path = output_dir / "marker_component_labels.tiff"
        tifffile.imwrite(
            marker_labels_path,
            np.stack(
                [products.marker_labels[role].astype(np.uint32) for role in products.marker_labels]
            ),
            metadata={
                "axes": "CYX",
                "channel_names": [
                    config["channels"][role]["alias"] for role in products.marker_labels
                ],
            },
            compression="zlib",
        )
        paths["marker_component_labels"] = str(marker_labels_path)
    nucleus_role = config["microglia_count"].get("nucleus_channel")
    if bool(output.get("save_cell_labels", True)) and nucleus_role in products.positive_masks:
        nucleus_labels_path = output_dir / "nucleus_labels.tiff"
        cell_labels_path = output_dir / "cell_labels.tiff"
        tifffile.imwrite(
            nucleus_labels_path,
            products.nucleus_labels.astype(np.uint32),
            compression="zlib",
        )
        tifffile.imwrite(
            cell_labels_path,
            products.microglia_labels.astype(np.uint32),
            compression="zlib",
        )
        paths.update(
            {
                "nucleus_labels": str(nucleus_labels_path),
                "cell_labels": str(cell_labels_path),
            }
        )
    if bool(output.get("save_positive_masks", True)):
        channels_path = output_dir / "positive_masks.tiff"
        mask_roles = list(products.positive_masks)
        tifffile.imwrite(
            channels_path,
            np.stack(
                [products.positive_masks[role].astype(np.uint8) for role in mask_roles]
            ),
            metadata={
                "axes": "CYX",
                "channel_names": [
                    config["channels"][role]["alias"] for role in mask_roles
                ],
            },
            compression="zlib",
        )
        paths["positive_masks"] = str(channels_path)

    save_qc = bool(output.get("save_qc", True))
    processing_flags = {
        "save_raw_channels": bool(output.get("save_raw_channel_images", True)),
        "save_composite": bool(output.get("save_composite_image", True)),
        "save_segmentation": bool(output.get("save_segmentation_images", True)),
        "save_mask_images": bool(output.get("save_mask_images", True)),
        "save_stages": bool(output.get("save_stage_images", True)),
        "save_advanced": bool(output.get("save_advanced_images", True)),
    }
    save_processing = any(processing_flags.values())
    if save_qc or save_processing:
        channel_colors = _channel_colors(info, config, set(raw_images))
        channel_names = {
            role: str(config["channels"][role]["alias"]) for role in raw_images
        }
        paths.update(
            _save_qc(
                output_dir / "cell_analysis_qc.png" if save_qc else None,
                raw_images,
                products,
                int(output.get("preview_max_dimension_px", 2200)),
                channel_colors,
                channel_names,
                config["microglia_count"],
                config["plaque"].get("reference_channel", "abeta") if products.neighbour_enabled else next(iter(raw_images)),
                list(output.get("qc_panels", [])),
                output_dir / "processing_images" if save_processing else None,
                **processing_flags,
                ring_boundary_width_px=int(output.get("ring_boundary_width_px", 1)),
                drawing=output.get("drawing"),
                processing_steps=output.get("processing_steps"),
            )
        )
    return paths, tables

def run_analysis(
    config_or_path: dict[str, Any] | str | Path,
    *,
    progress: ProgressCallback | None = print,
    include_tables: bool = False,
) -> dict[str, Any]:
    """Read one CZI/PNG, analyze it, and export tables and QC files."""

    raw_config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    cache_directory = raw_config.get("application", {}).get("cache_directory")
    if cache_directory:
        configure_cache_directory(cache_directory)
    raw_input = raw_config.get("input", {})
    image_path = raw_input.get("image_path")
    if not image_path:
        raise ValueError("input.image_path is required.")
    _notify(progress, "Inspecting image metadata...")
    info = inspect_image(
        image_path,
        pixel_size_um_x=raw_input.get("pixel_size_um_x"),
        pixel_size_um_y=raw_input.get("pixel_size_um_y"),
        image_width_um=raw_input.get("image_width_um"),
        image_height_um=raw_input.get("image_height_um"),
    )
    config = normalize_config(raw_config, info)
    input_config = config["input"]
    roles = [
        role for role, item in config["channels"].items()
        if bool(item.get("enabled", True))
    ]
    role_indices = {role: int(config["channels"][role]["index"]) for role in roles}
    _notify(progress, "Reading configured fluorescence channels...")
    by_index = read_image_channels(
        image_path,
        info,
        scene=input_config["scene"],
        time_index=input_config["time_index"],
        z_projection=input_config["z_projection"],
        z_index=input_config["z_index"],
        zoom=input_config["zoom"],
        channel_indices=sorted(set(role_indices.values())),
    )
    images = {role: by_index[index] for role, index in role_indices.items()}
    pixel_x = info.pixel_size_um_x
    pixel_y = info.pixel_size_um_y
    if pixel_x is None or pixel_y is None:
        raise ValueError("Physical X/Y pixel calibration is required for cell analysis.")
    effective_x = float(pixel_x) / input_config["zoom"]
    effective_y = float(pixel_y) / input_config["zoom"]
    _notify(progress, "Segmenting channels and running selected advanced analyses...")
    products = analyze_arrays(
        images,
        config,
        pixel_size_um_x=effective_x,
        pixel_size_um_y=effective_y,
        source_file=str(Path(image_path).expanduser().resolve()),
    )
    output_dir = Path(input_config["output_dir"])
    paths, tables = _write_outputs(output_dir, config, info, products, images)
    result = {
        "source_file": str(Path(image_path).expanduser().resolve()),
        "output_dir": str(output_dir),
        "plaque_count_all": int(products.image_summary.iloc[0]["plaque_count_all"]),
        "plaque_count_exported": int(len(products.plaque_measurements)),
        "thresholds": products.thresholds,
        "files": paths,
        "warnings": info.warnings,
    }
    summary_path = output_dir / "analysis_summary.json"
    result["files"]["analysis_summary"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if include_tables:
        result["_tables"] = tables
        result["_render_context"] = {"products": products, "images": images, "info": info, "config": config}
    _notify(progress, f"Complete: {result['plaque_count_all']} reference objects detected.")
    return result
