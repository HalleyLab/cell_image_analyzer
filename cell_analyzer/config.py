"""Configuration creation, loading, validation, and normalization."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

import yaml

from .models import CziInfo


DEFAULT_CHANNEL_CONFIG: dict[str, Any] = {
    "alias": None,
    "gaussian_sigma_px": 1.0,
    "measurement_threshold": {
        "method": "otsu",
        "percentile": 95.0,
        "scale": 1.0,
        "value": 0.0,
    },
}


DEFAULT_SEGMENTATION_CONFIG: dict[str, Any] = {
    "threshold_method": "otsu",
    "threshold_scale": 0.8,
    "threshold_percentile": 90.0,
    "adaptive_block_size_px": 51,
    "adaptive_offset": 0.0,
    "invert": False,
    "opening_radius_px": 1,
    "closing_radius_px": 2,
    "fill_all_holes": True,
    "min_hole_area_px": 32,
    "min_area_um2": 5.0,
    "max_area_um2": None,
    "min_circularity": 0.05,
    "min_local_contrast_ratio": 1.0,
    "local_contrast_ring_px": 4,
    "clear_border": True,
    "border_exclusion_margin_px": 10,
    "split_touching": False,
    "min_peak_distance_px": 8,
    "watershed_min_peak_height_px": 0.0,
    "watershed_min_peak_prominence_px": 0.0,
    "watershed_compactness": 0.0,
}


def effective_pixel_area_um2(info: CziInfo, zoom: float) -> float:
    """Return the physical area represented by one analysis pixel."""

    scale = float(zoom)
    if scale <= 0:
        raise ValueError("zoom must be greater than 0.")
    if info.pixel_size_um_x is None or info.pixel_size_um_y is None:
        raise ValueError(
            "Physical X/Y pixel size is required for square-micrometer cell-area "
            "filtering. For PNG input, set input.image_width_um and "
            "input.image_height_um."
        )
    pixel_size_x = float(info.pixel_size_um_x)
    pixel_size_y = float(info.pixel_size_um_y)
    if pixel_size_x <= 0 or pixel_size_y <= 0:
        raise ValueError("Physical pixel sizes must be greater than 0.")
    return (pixel_size_x / scale) * (pixel_size_y / scale)


def segmentation_config_for_zoom(
    config: dict[str, Any],
    info: CziInfo,
    zoom: float,
) -> dict[str, Any]:
    """Convert public square-micrometer area limits to runtime pixel limits."""

    result = copy.deepcopy(config)
    pixel_area_um2 = effective_pixel_area_um2(info, zoom)
    min_area_um2 = float(result.get("min_area_um2", 5.0))
    max_area_raw = result.get("max_area_um2")
    min_area_px = max(1, int(math.ceil(min_area_um2 / pixel_area_um2)))
    max_area_px = (
        int(math.floor(float(max_area_raw) / pixel_area_um2))
        if max_area_raw is not None
        else None
    )
    if max_area_px is not None and max_area_px < min_area_px:
        raise ValueError(
            "The square-micrometer cell-area range is narrower than one pixel at "
            f"zoom {float(zoom):g}."
        )
    result["min_area_px"] = min_area_px
    result["max_area_px"] = max_area_px
    result["effective_pixel_area_um2"] = pixel_area_um2
    return result


def slugify(value: str, fallback: str = "channel") -> str:
    """Return a stable ASCII identifier suitable for table column names."""

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return cleaned or fallback


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def create_default_config(
    info: CziInfo,
    output_dir: str | Path | None = None,
    zoom: float = 1.0,
) -> dict[str, Any]:
    """Create a complete editable configuration from normalized image metadata."""

    source = Path(info.path)
    if output_dir is None:
        output_dir = source.with_name(f"{source.stem}_cell_analysis")

    channels: dict[str, Any] = {}
    used_aliases: set[str] = set()
    for channel in info.channels:
        alias = channel.name or f"Channel_{channel.index}"
        candidate = alias
        suffix = 2
        while slugify(candidate) in used_aliases:
            candidate = f"{alias}_{suffix}"
            suffix += 1
        used_aliases.add(slugify(candidate))
        channel_config = copy.deepcopy(DEFAULT_CHANNEL_CONFIG)
        channel_config["alias"] = candidate
        channels[str(channel.index)] = channel_config

    input_config: dict[str, Any] = {
        "image_path": str(source),
        "output_dir": str(output_dir),
        "scene": 0,
        "time_index": 0,
        "z_projection": "max",
        "z_index": 0,
        "zoom": float(zoom),
        "segmentation_channel": info.channels[0].index if info.channels else 0,
        "pixel_size_um_x": info.pixel_size_um_x,
        "pixel_size_um_y": info.pixel_size_um_y,
    }
    if source.suffix.casefold() == ".czi":
        # Keep the original key so existing YAML files and command-line use remain valid.
        input_config["czi_path"] = str(source)
    else:
        width_px = int(info.dimensions.get("X", (0, 0))[1])
        height_px = int(info.dimensions.get("Y", (0, 0))[1])
        input_config["image_width_um"] = (
            float(info.pixel_size_um_x) * width_px
            if info.pixel_size_um_x is not None else None
        )
        input_config["image_height_um"] = (
            float(info.pixel_size_um_y) * height_px
            if info.pixel_size_um_y is not None else None
        )

    return {
        "input": input_config,
        "segmentation": copy.deepcopy(DEFAULT_SEGMENTATION_CONFIG),
        "channels": channels,
        "output": {
            "save_imagej_rois": True,
            "save_geojson": True,
            "save_label_image": True,
            "preview_max_dimension_px": 2500,
            "roi_simplify_tolerance_px": 1.0,
        },
    }


def normalize_config(config: dict[str, Any], info: CziInfo) -> dict[str, Any]:
    """Merge user settings with defaults and validate critical values."""

    if "input" not in config:
        raise ValueError("The configuration must contain an 'input' section.")
    base = create_default_config(
        info,
        output_dir=config["input"].get("output_dir"),
        zoom=float(config["input"].get("zoom", 1.0)),
    )
    merged = _deep_merge(base, config)

    input_config = merged["input"]
    zoom = float(input_config.get("zoom", 1.0))
    if not 0.01 <= zoom <= 1.0:
        raise ValueError("input.zoom must be between 0.01 and 1.0.")
    input_config["zoom"] = zoom
    input_config["scene"] = int(input_config.get("scene", 0))
    input_config["time_index"] = int(input_config.get("time_index", 0))
    input_config["z_index"] = int(input_config.get("z_index", 0))
    input_config["segmentation_channel"] = int(input_config["segmentation_channel"])
    if input_config["z_projection"] not in {"single", "max", "mean"}:
        raise ValueError("input.z_projection must be 'single', 'max', or 'mean'.")

    available = {channel.index for channel in info.channels}
    if input_config["segmentation_channel"] not in available:
        raise ValueError(
            f"Segmentation channel {input_config['segmentation_channel']} is not available. "
            f"Available channels: {sorted(available)}"
        )

    normalized_channels: dict[str, Any] = {}
    used_aliases: set[str] = set()
    for channel in info.channels:
        raw = merged.get("channels", {}).get(str(channel.index), {})
        allowed = {
            key: value
            for key, value in raw.items()
            if key in DEFAULT_CHANNEL_CONFIG
        }
        if isinstance(allowed.get("measurement_threshold"), dict):
            allowed["measurement_threshold"] = {
                key: value
                for key, value in allowed["measurement_threshold"].items()
                if key in DEFAULT_CHANNEL_CONFIG["measurement_threshold"]
            }
        item = _deep_merge(DEFAULT_CHANNEL_CONFIG, allowed)
        item["alias"] = item.get("alias") or channel.name or f"Channel_{channel.index}"
        item["gaussian_sigma_px"] = max(0.0, float(item["gaussian_sigma_px"]))
        measurement_method = str(item["measurement_threshold"].get("method", "otsu"))
        if measurement_method not in {
            "none",
            "otsu",
            "yen",
            "triangle",
            "percentile",
            "manual",
        }:
            raise ValueError(
                f"Unsupported channel {channel.index} measurement threshold method: "
                f"{measurement_method}"
            )
        item["measurement_threshold"]["method"] = measurement_method
        measurement_percentile = float(
            item["measurement_threshold"].get("percentile", 95.0)
        )
        if not 0 <= measurement_percentile <= 100:
            raise ValueError("Channel measurement percentile must be between 0 and 100.")
        item["measurement_threshold"]["percentile"] = measurement_percentile
        measurement_scale = float(item["measurement_threshold"].get("scale", 1.0))
        if measurement_scale <= 0:
            raise ValueError("Channel measurement threshold scale must be greater than 0.")
        item["measurement_threshold"]["scale"] = measurement_scale
        measurement_value = float(item["measurement_threshold"].get("value", 0.0))
        if not math.isfinite(measurement_value):
            raise ValueError(
                "Channel manual measurement threshold must be a finite number."
            )
        item["measurement_threshold"]["value"] = measurement_value
        alias_id = slugify(str(item["alias"]), fallback=f"channel_{channel.index}")
        if alias_id in used_aliases:
            raise ValueError(
                "Channel aliases must produce unique output column names. "
                f"Duplicate alias: {item['alias']}"
            )
        used_aliases.add(alias_id)
        normalized_channels[str(channel.index)] = item
    merged["channels"] = normalized_channels
    raw_segmentation = config.get("segmentation", {})
    filtered_segmentation = {
        key: value
        for key, value in raw_segmentation.items()
        if key in DEFAULT_SEGMENTATION_CONFIG
    }
    merged["segmentation"] = _deep_merge(
        DEFAULT_SEGMENTATION_CONFIG, filtered_segmentation
    )
    segmentation_config = merged["segmentation"]
    if "min_area_um2" not in raw_segmentation and "min_area_px" in raw_segmentation:
        pixel_area_um2 = effective_pixel_area_um2(info, zoom)
        segmentation_config["min_area_um2"] = (
            float(raw_segmentation["min_area_px"]) * pixel_area_um2
        )
    if "max_area_um2" not in raw_segmentation and "max_area_px" in raw_segmentation:
        legacy_max_area = raw_segmentation.get("max_area_px")
        segmentation_config["max_area_um2"] = (
            float(legacy_max_area) * effective_pixel_area_um2(info, zoom)
            if legacy_max_area is not None
            else None
        )
    segmentation_method = str(segmentation_config.get("threshold_method", "otsu"))
    if segmentation_method not in {
        "otsu",
        "yen",
        "triangle",
        "percentile",
        "adaptive",
    }:
        raise ValueError(
            f"Unsupported segmentation threshold method: {segmentation_method}"
        )
    segmentation_config["threshold_method"] = segmentation_method
    threshold_scale = float(segmentation_config.get("threshold_scale", 1.0))
    if threshold_scale <= 0:
        raise ValueError("segmentation.threshold_scale must be greater than 0.")
    segmentation_config["threshold_scale"] = threshold_scale
    border_margin = int(segmentation_config.get("border_exclusion_margin_px", 0))
    if border_margin < 0:
        raise ValueError(
            "segmentation.border_exclusion_margin_px must be at least 0."
        )
    segmentation_config["border_exclusion_margin_px"] = border_margin
    min_area = float(segmentation_config.get("min_area_um2", 5.0))
    max_area = segmentation_config.get("max_area_um2")
    if min_area <= 0:
        raise ValueError("segmentation.min_area_um2 must be greater than 0.")
    if max_area is not None and float(max_area) <= min_area:
        raise ValueError(
            "segmentation.max_area_um2 must be larger than min_area_um2."
        )
    segmentation_config["min_area_um2"] = min_area
    segmentation_config["max_area_um2"] = (
        float(max_area) if max_area is not None else None
    )
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("The configuration root must be a mapping.")
    return data


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)
