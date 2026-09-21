"""Configuration helpers for cell fluorescence analysis."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

import yaml

from cell_analyzer.image_io import inspect_image
from cell_analyzer.models import CziInfo


from .advanced import (
    ADVANCED_DEFAULTS,
    CHANNEL_COLORS,
    DRAWING_DEFAULTS,
    STAGE_LABELS,
    normalize_advanced,
    roles_for_count,
)
from .advanced_features import ADVANCED_TABLES, FEATURE_IMAGES

THRESHOLD_METHODS = {"manual", "otsu", "yen", "triangle", "percentile"}
QC_PANEL_DEFAULTS = (
    "05_composite",
    "06_primary_object_segmentation",
    "07_channel_1_objects",
    "07_channel_2_objects",
    "07_channel_3_objects",
    "07_channel_4_objects",
    "09_cell_counting",
    "10_primary_object_distance_rings",
    "11_tissue_roi",
)


QC_PANEL_CHOICES = (*QC_PANEL_DEFAULTS, *STAGE_LABELS, *(f"raw_channel_{i}" for i in range(1, 5)), *FEATURE_IMAGES)
PROCESSING_CHOICES = {
    **FEATURE_IMAGES,
    **{name: name.replace("_", " ").title() for name in QC_PANEL_DEFAULTS},
    **STAGE_LABELS,
    **{f"raw_channel_{i}": f"Channel {i}: raw image" for i in range(1, 5)},
    **{f"mask_channel_{i}_object_filter_mask": f"Channel {i}: filtered mask" for i in range(1, 5)},
    "mask_neighbour_reference_object_mask": "Neighbour reference mask",
    "mask_neighbour_excluded_object_mask": "Excluded object mask",
    "mask_neighbour_no_nearby_cell_excluded": "No-nearby-cell mask",
    "mask_nucleus_mask": "Nucleus mask", "mask_cell_mask": "Confirmed cell mask",
    "individual_rings": "Individual distance-range masks and overlays",
    "advanced_overlap": "Advanced: overlap maps", "advanced_scatter": "Advanced: intensity density plots",
    "advanced_distance": "Advanced: distance heatmaps", "advanced_histogram": "Advanced: distance histograms",
}

DEFAULT_THRESHOLD: dict[str, Any] = {
    "method": "otsu",
    "value": 0.0,
    "scale": 1.0,
    "percentile": 95.0,
}


DEFAULT_OBJECT_FILTER: dict[str, Any] = {
    "enabled": True,
    "opening_radius_px": 0,
    "closing_radius_px": 0,
    "fill_holes": False,
    "max_hole_area_um2": 0.0,
    "min_area_um2": 0.0,
    "max_area_um2": None,
    "min_circularity": 0.0,
    "min_solidity": 0.0,
    "max_eccentricity": 1.0,
}


OUTPUT_SELECTION_DEFAULTS: dict[str, bool] = {
    "save_excel": True,
    "save_image_summary_csv": True,
    "save_primary_objects_csv": True,
    "save_candidate_qc_csv": True,
    "save_channel_objects_csv": True,
    "save_cells_csv": True,
    "save_ring_metrics_csv": True,
    "save_animal_summary_csv": True,
    "save_tissue_mask": True,
    "save_primary_object_labels": True,
    "save_excluded_object_masks": True,
    "save_channel_object_labels": True,
    "save_cell_labels": True,
    "save_positive_masks": True,
    "save_qc": True,
    "save_raw_channel_images": True,
    "save_composite_image": True,
    "save_segmentation_images": True,
    "save_mask_images": True,
    "save_stage_images": True,
    "save_advanced_images": True,
    **{flag: True for _, flag in ADVANCED_TABLES.values()},
}
MASK_OUTPUT_KEYS = (
    "save_tissue_mask", "save_primary_object_labels", "save_excluded_object_masks",
    "save_channel_object_labels", "save_cell_labels", "save_positive_masks",
)
PROCESSING_IMAGE_OUTPUT_KEYS = (
    "save_raw_channel_images", "save_composite_image",
    "save_segmentation_images", "save_mask_images", "save_stage_images", "save_advanced_images",
)


DEFAULT_CONFIG: dict[str, Any] = {
    "advanced": copy.deepcopy(ADVANCED_DEFAULTS),
    "application": {"cache_directory": None},
    "input": {
        "image_path": "",
        "output_dir": "",
        "scene": 0,
        "time_index": 0,
        "z_projection": "max",
        "z_index": 0,
        "zoom": 1.0,
        "pixel_size_um_x": None,
        "pixel_size_um_y": None,
        "image_width_um": None,
        "image_height_um": None,
    },
    "channels": {
        "abeta": {
            "index": 0,
            "alias": "Channel 1",
            "wavelength_nm": None,
            "color": "#00FF33",
            "gaussian_sigma_px": 1.0,
            "threshold": copy.deepcopy(DEFAULT_THRESHOLD),
            "object_filter": copy.deepcopy(DEFAULT_OBJECT_FILTER),
        },
        "iba1": {
            "index": 1,
            "alias": "Channel 2",
            "wavelength_nm": None,
            "color": "#FF00FF",
            "gaussian_sigma_px": 1.0,
            "threshold": copy.deepcopy(DEFAULT_THRESHOLD),
            "object_filter": copy.deepcopy(DEFAULT_OBJECT_FILTER),
        },
        "cd68": {
            "index": 2,
            "alias": "Channel 3",
            "wavelength_nm": None,
            "color": "#FF5900",
            "gaussian_sigma_px": 1.0,
            "threshold": copy.deepcopy(DEFAULT_THRESHOLD),
            "object_filter": copy.deepcopy(DEFAULT_OBJECT_FILTER),
        },
        "dapi": {
            "enabled": False,
            "index": 3,
            "alias": "Channel 4",
            "wavelength_nm": None,
            "color": "#0033FF",
            "gaussian_sigma_px": 1.0,
            "threshold": copy.deepcopy(DEFAULT_THRESHOLD),
            "object_filter": copy.deepcopy(DEFAULT_OBJECT_FILTER),
        },
    },
    "tissue_roi": {
        "mode": "full_image",
        "mask_path": None,
        "mask_directory": None,
        "mask_suffix": "_mask.png",
        "invert_mask": False,
    },
    "plaque": {
        "reference_channel": "abeta",
        "opening_radius_px": 0,
        "closing_radius_px": 1,
        "fill_holes": False,
        "max_hole_area_um2": 0.0,
        "min_area_um2": 10.0,
        "max_area_um2": None,
        "min_circularity": 0.0,
        "min_solidity": 0.0,
        "max_eccentricity": 1.0,
        "neuron_exclusion_mode": "shape_and_dark_center",
        "neuron_detection_threshold_scale": 0.75,
        "neuron_min_diameter_um": 8.0,
        "neuron_max_diameter_um": 28.0,
        "neuron_min_circularity": 0.35,
        "neuron_min_solidity": 0.60,
        "neuron_min_hole_fraction": 0.03,
        "neuron_max_center_shell_ratio": 0.95,
        "require_nearby_microglia": False,
        "nearby_microglia_radius_um": 30.0,
        "min_nearby_microglia_count": 1,
        "split_touching": False,
        "min_peak_distance_px": 8,
        "watershed_min_peak_height_px": 0.0,
        "watershed_compactness": 0.0,
        "exclude_boundary_plaques_from_table": True,
        "boundary_margin_um": 0.0,
    },
    "spatial": {
        "ring_edges_um": [],
    },
    "microglia_count": {
        "enabled": False,
        "nucleus_channel": None,
        "confirmation_channel": None,
        "opening_radius_px": 0,
        "closing_radius_px": 1,
        "fill_holes": True,
        "min_nucleus_area_um2": 10.0,
        "max_nucleus_area_um2": 150.0,
        "min_circularity": 0.20,
        "min_solidity": 0.70,
        "max_eccentricity": 0.98,
        "split_touching": True,
        "min_peak_distance_px": 3,
        "perinuclear_radius_um": 3.0,
        "min_confirmation_positive_fraction": 0.15,
    },
    "metadata": {
        "mouse_id": None,
        "genotype": None,
        "region": None,
        "section_id": None,
        "sex": None,
    },
    "batch": {
        "metadata_csv": None,
        "continue_on_error": True,
    },
    "output": {
        **OUTPUT_SELECTION_DEFAULTS,
        "table_columns": {},
        "drawing": copy.deepcopy(DRAWING_DEFAULTS),
        "processing_steps": None,
        "qc_panels": list(QC_PANEL_DEFAULTS),
        # Legacy aggregate keys remain readable by older sessions and notebooks.
        "save_masks": True,
        "save_processing_images": True,
        "ring_boundary_width_px": 1,
        "preview_max_dimension_px": 2200,
    },
}


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def default_channel(role: str, position: int) -> dict[str, Any]:
    """Create one channel config; legacy role keys keep their saved defaults."""

    legacy = DEFAULT_CONFIG["channels"].get(role, DEFAULT_CONFIG["channels"]["abeta"])
    result = copy.deepcopy(legacy)
    result.update(
        enabled=True,
        index=position,
        alias=f"Channel {position + 1}",
        color=CHANNEL_COLORS[position % len(CHANNEL_COLORS)],
    )
    return result


def dynamic_stage_labels(count: int) -> dict[str, str]:
    return {
        f"stage_channel_{index}_{stage}": f"Channel {index}: {title}"
        for index in range(1, count + 1)
        for stage, title in {
            "gaussian": "Gaussian image", "threshold": "Threshold mask",
            "morphology": "Morphology mask", "candidates": "Candidate labels",
            "accepted": "Accepted mask", "rejected": "Rejected mask",
        }.items()
    }


def processing_choices(count: int) -> dict[str, str]:
    def available(name: str) -> bool:
        match = re.search(r"(?:channel_|_channel_)([0-9]+)", name)
        return match is None or int(match.group(1)) <= count

    choices = {key: value for key, value in PROCESSING_CHOICES.items() if available(key)}
    choices.update(dynamic_stage_labels(count))
    choices.update({f"raw_channel_{i}": f"Channel {i}: raw image" for i in range(1, count + 1)})
    choices.update({f"mask_channel_{i}_object_filter_mask": f"Channel {i}: filtered mask" for i in range(1, count + 1)})
    return choices


def qc_panel_choices(count: int) -> tuple[str, ...]:
    def available(name: str) -> bool:
        match = re.search(r"(?:channel_|_channel_)([0-9]+)", name)
        return match is None or int(match.group(1)) <= count

    return tuple(dict.fromkeys((
        *(name for name in QC_PANEL_CHOICES if available(name)),
        *(f"07_channel_{i}_objects" for i in range(1, count + 1)),
        *(f"raw_channel_{i}" for i in range(1, count + 1)),
        *dynamic_stage_labels(count),
    )))


def _selection_channel(name: str) -> int | None:
    match = re.search(r"(?:channel_|_channel_)([0-9]+)", name)
    return int(match.group(1)) if match else None


def _wavelength_family(value: int) -> set[int]:
    families = (
        {405},
        {488},
        {546, 555, 568},
        {594},
        {633, 640, 647},
    )
    return next((family for family in families if value in family), {value})


def _role_wavelengths_from_filename(path: str | Path) -> dict[str, int]:
    name = Path(path).stem.casefold().replace("β", "beta")
    role_patterns = {
        "abeta": r"(?:abeta|a[-_ ]?beta|\bab)\s*[-_+ ]*([0-9]{3})",
        "iba1": r"iba[-_ ]?1\s*[-_+ ]*([0-9]{3})",
        "cd68": r"cd[-_ ]?68\s*[-_+ ]*([0-9]{3})",
    }
    result: dict[str, int] = {}
    for role, pattern in role_patterns.items():
        match = re.search(pattern, name)
        if match:
            result[role] = int(match.group(1))
    return result


def _channel_wavelength(name: str) -> int | None:
    match = re.search(r"(?:af|alexa|ch|channel)?\s*([0-9]{3})", name.casefold())
    return int(match.group(1)) if match else None


def _guess_channel_indices(info: CziInfo) -> dict[str, int]:
    names = {channel.index: (channel.name or "").casefold() for channel in info.channels}
    rules = {
        "abeta": ("abeta", "amyloid", "6e10", "3d6", "x34", "thio", "aβ"),
        "iba1": ("iba1", "iba-1", "aif1"),
        "cd68": ("cd68",),
        "dapi": ("dapi", "hoechst"),
    }
    guessed: dict[str, int] = {}
    used: set[int] = set()
    requested_wavelengths = _role_wavelengths_from_filename(info.path)
    channel_wavelengths = {
        index: _channel_wavelength(name) for index, name in names.items()
    }
    for role, wavelength in requested_wavelengths.items():
        family = _wavelength_family(wavelength)
        found = next(
            (
                index
                for index, channel_wavelength in channel_wavelengths.items()
                if channel_wavelength in family and index not in used
            ),
            None,
        )
        if found is not None:
            guessed[role] = found
            used.add(found)
    for role, terms in rules.items():
        if role in guessed:
            continue
        found = next(
            (index for index, name in names.items() if any(term in name for term in terms)),
            None,
        )
        if found is not None and found not in used:
            guessed[role] = found
            used.add(found)
    remaining = [channel.index for channel in info.channels if channel.index not in used]
    for role in ("abeta", "iba1", "cd68"):
        if role not in guessed and remaining:
            guessed[role] = remaining.pop(0)
    return guessed


def create_default_config(
    image_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create an editable configuration using image metadata."""

    info = inspect_image(image_path)
    config = copy.deepcopy(DEFAULT_CONFIG)
    source = Path(info.path)
    config["input"]["image_path"] = str(source)
    config["input"]["output_dir"] = str(
        Path(output_dir) if output_dir is not None else source.with_name(f"{source.stem}_cell_analysis")
    )
    config["input"]["pixel_size_um_x"] = info.pixel_size_um_x
    config["input"]["pixel_size_um_y"] = info.pixel_size_um_y
    config["channels"] = {
        role: default_channel(role, position)
        for position, role in enumerate(roles_for_count(len(info.channels)))
    }
    for position, role in enumerate(config["channels"]):
        config["channels"][role]["index"] = info.channels[position].index
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("The configuration root must be a mapping.")
    return loaded


def save_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def normalize_config(config: dict[str, Any], info: CziInfo) -> dict[str, Any]:
    """Merge defaults and validate fields needed by the analysis engine."""

    merged = _deep_merge(DEFAULT_CONFIG, config)
    if "advanced" not in config and "plaque" in config:
        # Preserve the always-on neighbour workflow of previous saved sessions.
        merged["advanced"]["neighbour_enabled"] = True
    if "drawing" not in config.get("output", {}):
        merged["output"]["drawing"]["boundaries"]["ring"]["width_px"] = config.get("output", {}).get("ring_boundary_width_px", 1)
    cache_directory = merged["application"].get("cache_directory")
    merged["application"]["cache_directory"] = (
        str(Path(cache_directory).expanduser().resolve()) if cache_directory else None
    )
    input_config = merged["input"]
    input_config["image_path"] = str(Path(input_config["image_path"]).expanduser().resolve())
    if not input_config.get("output_dir"):
        source = Path(input_config["image_path"])
        input_config["output_dir"] = str(source.with_name(f"{source.stem}_cell_analysis"))
    input_config["output_dir"] = str(Path(input_config["output_dir"]).expanduser().resolve())
    input_config["scene"] = int(input_config.get("scene", 0))
    input_config["time_index"] = int(input_config.get("time_index", 0))
    input_config["z_index"] = int(input_config.get("z_index", 0))
    input_config["zoom"] = float(input_config.get("zoom", 1.0))
    if not 0.01 <= input_config["zoom"] <= 1.0:
        raise ValueError("input.zoom must be between 0.01 and 1.0.")
    if input_config.get("z_projection") not in {"single", "max", "mean"}:
        raise ValueError("input.z_projection must be single, max, or mean.")

    if not info.channels:
        raise ValueError("The image has no readable channels.")
    channel_roles = roles_for_count(len(info.channels))
    supplied_channels = config.get("channels", {})
    merged["channels"] = {
        role: _deep_merge(
            default_channel(role, position),
            supplied_channels.get(role, {}),
        )
        for position, role in enumerate(channel_roles)
    }
    for position, role in enumerate(channel_roles):
        if role not in supplied_channels:
            merged["channels"][role]["index"] = info.channels[position].index
    available = {channel.index for channel in info.channels}
    roles = tuple(
        role for role, item in merged["channels"].items()
        if bool(item.get("enabled", True))
    )
    if not roles:
        raise ValueError("Enable at least one image channel.")
    indices: list[int] = []
    export_names: list[str] = []
    for position, role in enumerate(merged["channels"]):
        item = merged["channels"][role]
        item["enabled"] = bool(item.get("enabled", True))
        item["alias"] = str(item.get("alias") or f"Channel {position + 1}").strip()
        wavelength = item.get("wavelength_nm")
        item["wavelength_nm"] = None if wavelength in {None, ""} else float(wavelength)
        if item["wavelength_nm"] is not None and item["wavelength_nm"] <= 0:
            raise ValueError(f"{item['alias']} wavelength must be greater than zero.")
        item["color"] = str(item.get("color") or "").upper()
        if not re.fullmatch(r"#[0-9A-F]{6}", item["color"]):
            raise ValueError(f"{item['alias']} color must be #RRGGBB.")
        if item["enabled"]:
            export_names.append(re.sub(r"[^a-z0-9]+", "_", item["alias"].lower()).strip("_"))
        item["index"] = int(item["index"])
        if item["enabled"] and item["index"] not in available:
            raise ValueError(
                f"{item['alias']} image channel {item['index']} is unavailable; available channels: {sorted(available)}"
            )
        if item["enabled"]:
            indices.append(item["index"])
        item["gaussian_sigma_px"] = max(0.0, float(item.get("gaussian_sigma_px", 0.0)))
        threshold = item["threshold"]
        threshold["method"] = str(threshold.get("method", "otsu")).lower()
        if threshold["method"] not in THRESHOLD_METHODS:
            raise ValueError(f"Unsupported {item['alias']} threshold method: {threshold['method']}")
        threshold["value"] = float(threshold.get("value", 0.0))
        threshold["scale"] = float(threshold.get("scale", 1.0))
        threshold["percentile"] = float(threshold.get("percentile", 95.0))
        if not math.isfinite(threshold["value"]):
            raise ValueError(f"{item['alias']} manual threshold must be finite.")
        if threshold["scale"] <= 0:
            raise ValueError(f"{item['alias']} threshold scale must be greater than zero.")
        if not 0 <= threshold["percentile"] <= 100:
            raise ValueError(f"{item['alias']} threshold percentile must be between 0 and 100.")
        if item["enabled"]:
            object_filter = item["object_filter"]
            object_filter["enabled"] = bool(object_filter.get("enabled", True))
            object_filter["opening_radius_px"] = max(
                0, int(object_filter.get("opening_radius_px", 0))
            )
            object_filter["closing_radius_px"] = max(
                0, int(object_filter.get("closing_radius_px", 0))
            )
            object_filter["fill_holes"] = bool(object_filter.get("fill_holes", False))
            object_filter["max_hole_area_um2"] = max(
                0.0, float(object_filter.get("max_hole_area_um2", 0.0))
            )
            object_filter["min_area_um2"] = max(
                0.0, float(object_filter.get("min_area_um2", 0.0))
            )
            maximum_area = object_filter.get("max_area_um2")
            object_filter["max_area_um2"] = (
                None if maximum_area in {None, ""} else float(maximum_area)
            )
            if (
                object_filter["max_area_um2"] is not None
                and object_filter["max_area_um2"] < object_filter["min_area_um2"]
            ):
                raise ValueError(
                    f"{role} object_filter.max_area_um2 cannot be below min_area_um2."
                )
            for key, default in (
                ("min_circularity", 0.0),
                ("min_solidity", 0.0),
                ("max_eccentricity", 1.0),
            ):
                object_filter[key] = float(object_filter.get(key, default))
                if not 0 <= object_filter[key] <= 1:
                    raise ValueError(
                        f"{role} object_filter.{key} must be between 0 and 1."
                    )
    if len(set(indices)) != len(indices):
        raise ValueError("Enabled channels must use different image-channel indices.")
    if len(set(export_names)) != len(export_names) or any(not name for name in export_names):
        raise ValueError("Enabled channel names must be unique and contain a letter or number.")
    if set(export_names) & {"primary_object", "cell", "excluded_object"}:
        raise ValueError("Channel names cannot be Primary Object, Cell, or Excluded Object.")

    roi = merged["tissue_roi"]
    roi["mode"] = str(roi.get("mode", "full_image")).lower()
    if roi["mode"] not in {"full_image", "mask", "mask_directory"}:
        raise ValueError("tissue_roi.mode must be full_image, mask, or mask_directory.")
    if roi["mode"] == "mask" and not roi.get("mask_path"):
        raise ValueError("tissue_roi.mask_path is required when mode is mask.")
    roi["invert_mask"] = bool(roi.get("invert_mask", False))

    plaque = merged["plaque"]
    plaque["reference_channel"] = str(plaque.get("reference_channel", "abeta"))
    if plaque["reference_channel"] not in (roles if merged["advanced"]["neighbour_enabled"] else channel_roles):
        raise ValueError("Neighbour analysis must select an enabled reference channel.")
    plaque["opening_radius_px"] = max(0, int(plaque.get("opening_radius_px", 0)))
    plaque["closing_radius_px"] = max(0, int(plaque.get("closing_radius_px", 0)))
    plaque["fill_holes"] = bool(plaque.get("fill_holes", False))
    plaque["max_hole_area_um2"] = max(
        0.0, float(plaque.get("max_hole_area_um2", 0.0))
    )
    plaque["min_area_um2"] = max(0.0, float(plaque.get("min_area_um2", 0.0)))
    max_area = plaque.get("max_area_um2")
    plaque["max_area_um2"] = None if max_area in {None, ""} else float(max_area)
    if plaque["max_area_um2"] is not None and plaque["max_area_um2"] < plaque["min_area_um2"]:
        raise ValueError("plaque.max_area_um2 cannot be below min_area_um2.")
    plaque["min_circularity"] = float(plaque.get("min_circularity", 0.0))
    plaque["min_solidity"] = float(plaque.get("min_solidity", 0.0))
    plaque["max_eccentricity"] = float(plaque.get("max_eccentricity", 1.0))
    if not 0 <= plaque["min_circularity"] <= 1:
        raise ValueError("plaque.min_circularity must be between 0 and 1.")
    if not 0 <= plaque["min_solidity"] <= 1:
        raise ValueError("plaque.min_solidity must be between 0 and 1.")
    if not 0 <= plaque["max_eccentricity"] <= 1:
        raise ValueError("plaque.max_eccentricity must be between 0 and 1.")
    plaque["neuron_exclusion_mode"] = str(
        plaque.get("neuron_exclusion_mode", "shape_and_dark_center")
    ).lower()
    if plaque["neuron_exclusion_mode"] not in {
        "off",
        "shape",
        "shape_and_dark_center",
    }:
        raise ValueError(
            "plaque.neuron_exclusion_mode must be off, shape, or shape_and_dark_center."
        )
    plaque["neuron_min_diameter_um"] = max(
        0.0, float(plaque.get("neuron_min_diameter_um", 8.0))
    )
    plaque["neuron_max_diameter_um"] = max(
        0.0, float(plaque.get("neuron_max_diameter_um", 28.0))
    )
    if plaque["neuron_max_diameter_um"] < plaque["neuron_min_diameter_um"]:
        raise ValueError(
            "plaque.neuron_max_diameter_um cannot be below neuron_min_diameter_um."
        )
    for key, default in (
        ("neuron_min_circularity", 0.35),
        ("neuron_min_solidity", 0.60),
        ("neuron_min_hole_fraction", 0.03),
    ):
        plaque[key] = float(plaque.get(key, default))
        if not 0 <= plaque[key] <= 1:
            raise ValueError(f"plaque.{key} must be between 0 and 1.")
    plaque["neuron_max_center_shell_ratio"] = max(
        0.0, float(plaque.get("neuron_max_center_shell_ratio", 0.95))
    )
    plaque["neuron_detection_threshold_scale"] = float(
        plaque.get("neuron_detection_threshold_scale", 0.75)
    )
    if plaque["neuron_detection_threshold_scale"] <= 0:
        raise ValueError("plaque.neuron_detection_threshold_scale must be greater than 0.")
    plaque["split_touching"] = bool(plaque.get("split_touching", False))
    plaque["min_peak_distance_px"] = max(
        1, int(plaque.get("min_peak_distance_px", 8))
    )
    plaque["watershed_min_peak_height_px"] = max(
        0.0, float(plaque.get("watershed_min_peak_height_px", 0.0))
    )
    plaque["watershed_compactness"] = max(
        0.0, float(plaque.get("watershed_compactness", 0.0))
    )
    plaque["exclude_boundary_plaques_from_table"] = bool(
        plaque.get("exclude_boundary_plaques_from_table", True)
    )
    plaque["boundary_margin_um"] = max(0.0, float(plaque.get("boundary_margin_um", 0.0)))
    plaque["require_nearby_microglia"] = bool(
        plaque.get("require_nearby_microglia", False)
    )
    plaque["nearby_microglia_radius_um"] = max(
        0.0, float(plaque.get("nearby_microglia_radius_um", 30.0))
    )
    plaque["min_nearby_microglia_count"] = max(
        1, int(plaque.get("min_nearby_microglia_count", 1))
    )

    output = merged["output"]
    raw_output = config.get("output", {})
    legacy_masks = bool(raw_output.get("save_masks", True))
    legacy_processing = bool(raw_output.get("save_processing_images", True))
    for key, default in OUTPUT_SELECTION_DEFAULTS.items():
        if key in raw_output:
            value = raw_output[key]
        elif key in MASK_OUTPUT_KEYS and "save_masks" in raw_output:
            value = legacy_masks
        elif key in PROCESSING_IMAGE_OUTPUT_KEYS and "save_processing_images" in raw_output:
            value = legacy_processing
        else:
            value = output.get(key, default)
        output[key] = bool(value)
    output["save_masks"] = any(output[key] for key in MASK_OUTPUT_KEYS)
    output["save_processing_images"] = any(
        output[key] for key in PROCESSING_IMAGE_OUTPUT_KEYS
    )
    qc_panels = output.get("qc_panels", QC_PANEL_DEFAULTS)
    if not isinstance(qc_panels, (list, tuple)):
        raise ValueError("output.qc_panels must be a list.")
    qc_panels = [
        name for name in qc_panels
        if _selection_channel(name) in {None, *range(1, len(channel_roles) + 1)}
    ]
    unknown_qc_panels = set(qc_panels) - set(qc_panel_choices(len(channel_roles)))
    if unknown_qc_panels:
        raise ValueError(f"Unknown output.qc_panels: {sorted(unknown_qc_panels)}")
    output["qc_panels"] = list(dict.fromkeys(qc_panels))
    if output["save_qc"] and not output["qc_panels"]:
        raise ValueError("Select at least one Overview QC panel or disable Overview QC.")
    raw_table_columns = output.get("table_columns", {})
    if not isinstance(raw_table_columns, dict):
        raise ValueError("output.table_columns must be a mapping of table names to column lists.")
    table_columns: dict[str, list[str]] = {}
    for table_name, columns in raw_table_columns.items():
        if not isinstance(columns, (list, tuple)):
            raise ValueError(
                f"output.table_columns.{table_name} must be a list of column names."
            )
        table_columns[str(table_name)] = list(
            dict.fromkeys(str(column) for column in columns)
        )
    output["table_columns"] = table_columns
    output["ring_boundary_width_px"] = max(
        1, int(output.get("ring_boundary_width_px", 1))
    )
    output["preview_max_dimension_px"] = max(
        200, int(output.get("preview_max_dimension_px", 2200))
    )

    edges = [float(value) for value in merged["spatial"].get("ring_edges_um", [])]
    if edges and (
        len(edges) < 2
        or edges[0] != 0
        or any(b <= a for a, b in zip(edges, edges[1:]))
    ):
        raise ValueError("spatial.ring_edges_um must start at 0 and increase, e.g. [0, 30].")
    merged["spatial"]["ring_edges_um"] = edges

    microglia = merged["microglia_count"]
    raw_microglia = config.get("microglia_count", {})
    if (
        raw_microglia.get("nucleus_channel") in {None, ""}
        and "min_iba1_positive_fraction" in raw_microglia
    ):
        # Preserve old sessions and the legacy notebook while new GUI sessions stay unassigned.
        microglia["nucleus_channel"] = "dapi"
        microglia["confirmation_channel"] = "iba1"
        microglia["min_confirmation_positive_fraction"] = raw_microglia[
            "min_iba1_positive_fraction"
        ]
    microglia["enabled"] = bool(microglia.get("enabled", False))
    for key in ("nucleus_channel", "confirmation_channel"):
        value = microglia.get(key)
        microglia[key] = None if value in {None, ""} else str(value)
        if microglia[key] is not None and microglia[key] not in roles:
            raise ValueError(f"microglia_count.{key} must select an enabled channel.")
    if microglia["enabled"] and microglia["nucleus_channel"] is None:
        raise ValueError("Select a nucleus channel for cell counting.")
    for key, default in (
        ("opening_radius_px", 0),
        ("closing_radius_px", 1),
        ("min_peak_distance_px", 3),
    ):
        microglia[key] = max(0, int(microglia.get(key, default)))
    microglia["min_peak_distance_px"] = max(1, microglia["min_peak_distance_px"])
    microglia["fill_holes"] = bool(microglia.get("fill_holes", True))
    microglia["split_touching"] = bool(microglia.get("split_touching", True))
    microglia["min_nucleus_area_um2"] = max(
        0.0, float(microglia.get("min_nucleus_area_um2", 10.0))
    )
    maximum_nucleus_area = microglia.get("max_nucleus_area_um2")
    microglia["max_nucleus_area_um2"] = (
        None if maximum_nucleus_area in {None, ""} else float(maximum_nucleus_area)
    )
    if (
        microglia["max_nucleus_area_um2"] is not None
        and microglia["max_nucleus_area_um2"]
        < microglia["min_nucleus_area_um2"]
    ):
        raise ValueError(
            "microglia_count.max_nucleus_area_um2 cannot be below min_nucleus_area_um2."
        )
    for key, default in (
        ("min_circularity", 0.20),
        ("min_solidity", 0.70),
        ("max_eccentricity", 0.98),
        ("min_confirmation_positive_fraction", 0.15),
    ):
        microglia[key] = float(microglia.get(key, default))
        if not 0 <= microglia[key] <= 1:
            raise ValueError(f"microglia_count.{key} must be between 0 and 1.")
    microglia["perinuclear_radius_um"] = max(
        0.0, float(microglia.get("perinuclear_radius_um", 3.0))
    )
    if merged["advanced"]["neighbour_enabled"] and plaque["require_nearby_microglia"] and not microglia["enabled"]:
        raise ValueError(
            "Nearby-cell gating requires cell counting to be enabled."
        )
    normalize_advanced(merged, roles)
    steps = output.get("processing_steps")
    if steps is not None:
        steps = [
            name for name in steps
            if _selection_channel(name) in {None, *range(1, len(channel_roles) + 1)}
        ]
        if not isinstance(steps, (list, tuple)) or set(steps) - set(processing_choices(len(channel_roles))):
            raise ValueError("Unknown processing image selection.")
        output["processing_steps"] = list(dict.fromkeys(steps))
    drawing = output["drawing"]
    for key in ("low_percentile", "high_percentile", "gamma", "gain", "opacity"):
        drawing[key] = float(drawing[key])
        if not math.isfinite(drawing[key]):
            raise ValueError(f"Drawing {key} must be finite.")
    if not 0 <= drawing["low_percentile"] < drawing["high_percentile"] <= 100:
        raise ValueError("Display percentiles must satisfy 0 <= low < high <= 100.")
    if drawing["gamma"] <= 0 or drawing["gain"] < 0 or not 0 <= drawing["opacity"] <= 1:
        raise ValueError("Gamma must be positive, gain nonnegative, and opacity between 0 and 1.")
    for key, minimum, maximum in (("font_size", 6, 48), ("dpi", 72, 600), ("histogram_bins", 5, 512)):
        drawing[key] = int(drawing[key])
        if not minimum <= drawing[key] <= maximum:
            raise ValueError(f"Drawing {key} must be between {minimum} and {maximum}.")
    drawing["density_log_scale"] = bool(drawing["density_log_scale"])
    drawing["show_object_ids"] = bool(drawing["show_object_ids"])
    if drawing["heatmap_cmap"] not in {"viridis", "magma", "inferno", "plasma", "cividis", "gray", "turbo"}:
        raise ValueError("Unknown display heatmap color map.")
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", str(drawing["background"])):
        raise ValueError("Plot background must be #RRGGBB.")
    for position, role in enumerate(channel_roles):
        drawing["boundaries"].setdefault(
            role,
            {"color": merged["channels"][role]["color"], "width_px": 1},
        )
    for name, style in drawing["boundaries"].items():
        if name not in DRAWING_DEFAULTS["boundaries"] and name not in channel_roles:
            continue
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", str(style["color"])):
            raise ValueError("Boundary colors must be #RRGGBB.")
        style["width_px"] = int(style["width_px"])
        if not 1 <= style["width_px"] <= 50:
            raise ValueError("Boundary width must be between 1 and 50 display pixels.")
    return merged
