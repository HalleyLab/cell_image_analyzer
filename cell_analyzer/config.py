"""Configuration creation, loading, validation, and normalization."""

from __future__ import annotations

import copy
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
    "min_area_px": 50,
    "max_area_px": None,
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
    """Create a complete editable configuration from CZI metadata."""

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

    return {
        "input": {
            "czi_path": str(source),
            "output_dir": str(output_dir),
            "scene": 0,
            "time_index": 0,
            "z_projection": "max",
            "z_index": 0,
            "zoom": float(zoom),
            "segmentation_channel": info.channels[0].index if info.channels else 0,
        },
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
        }:
            raise ValueError(
                f"Unsupported channel {channel.index} measurement threshold method: "
                f"{measurement_method}"
            )
        item["measurement_threshold"]["method"] = measurement_method
        alias_id = slugify(str(item["alias"]), fallback=f"channel_{channel.index}")
        if alias_id in used_aliases:
            raise ValueError(
                "Channel aliases must produce unique output column names. "
                f"Duplicate alias: {item['alias']}"
            )
        used_aliases.add(alias_id)
        normalized_channels[str(channel.index)] = item
    merged["channels"] = normalized_channels
    raw_segmentation = merged.get("segmentation", {})
    filtered_segmentation = {
        key: value
        for key, value in raw_segmentation.items()
        if key in DEFAULT_SEGMENTATION_CONFIG
    }
    merged["segmentation"] = _deep_merge(
        DEFAULT_SEGMENTATION_CONFIG, filtered_segmentation
    )
    segmentation_config = merged["segmentation"]
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
    min_area = int(segmentation_config.get("min_area_px", 1))
    max_area = segmentation_config.get("max_area_px")
    if min_area < 1:
        raise ValueError("segmentation.min_area_px must be at least 1.")
    if max_area is not None and float(max_area) <= min_area:
        raise ValueError("segmentation.max_area_px must be larger than min_area_px.")
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
