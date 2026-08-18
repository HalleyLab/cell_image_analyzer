"""End-to-end CZI cell analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_yaml, normalize_config, save_yaml, slugify
from .czi_io import inspect_czi, read_czi_channels
from .measurements import measure_rois
from .preprocessing import preprocess_channel
from .rois import export_rois, save_overlay
from .segmentation import segment_cells

ProgressCallback = Callable[[str], None]


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _flatten_mapping(mapping: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_mapping(value, path))
        else:
            rows.append({"parameter": path, "value": value})
    return rows


def _save_channel_panel(
    processed: dict[int, Any],
    channel_configs: dict[str, dict[str, Any]],
    path: Path,
    max_dimension: int,
) -> None:
    count = len(processed)
    columns = min(3, max(1, count))
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (channel_index, channel) in zip(axes.ravel(), sorted(processed.items())):
        image = channel.analysis_image
        height, width = image.shape
        stride = max(1, int(np.ceil(max(height, width) / max_dimension)))
        axis.imshow(image[::stride, ::stride], cmap="gray")
        alias = channel_configs[str(channel_index)].get("alias", f"Channel_{channel_index}")
        axis.set_title(f"C{channel_index}: {alias}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _write_excel(
    path: Path,
    measurements: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
    channel_rows: list[dict[str, Any]],
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        measurements.to_excel(writer, sheet_name="ROI Measurements", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(channel_rows).to_excel(writer, sheet_name="Channels", index=False)
        pd.DataFrame(_flatten_mapping(config)).to_excel(
            writer, sheet_name="Parameters", index=False
        )


def run_analysis(
    config_or_path: dict[str, Any] | str | Path,
    *,
    progress: ProgressCallback | None = print,
) -> dict[str, Any]:
    """Run segmentation, ROI export, and per-channel measurements."""

    raw_config = (
        load_yaml(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else dict(config_or_path)
    )
    czi_path = raw_config.get("input", {}).get("czi_path")
    if not czi_path:
        raise ValueError("input.czi_path is required.")

    _notify(progress, "Inspecting CZI metadata...")
    info = inspect_czi(czi_path)
    config = normalize_config(raw_config, info)
    input_config = config["input"]
    output_config = config["output"]
    output_dir = Path(input_config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress, "Reading requested scene and channels...")
    images = read_czi_channels(
        czi_path,
        info,
        scene=int(input_config["scene"]),
        time_index=int(input_config["time_index"]),
        z_projection=str(input_config["z_projection"]),
        z_index=int(input_config["z_index"]),
        zoom=float(input_config["zoom"]),
    )

    _notify(progress, "Applying independent preprocessing to each channel...")
    processed = {
        channel_index: preprocess_channel(image, config["channels"][str(channel_index)])
        for channel_index, image in images.items()
    }
    segmentation_channel = int(input_config["segmentation_channel"])
    _notify(progress, f"Segmenting cells from channel {segmentation_channel}...")
    labels, cleaned_mask, segmentation_diagnostics = segment_cells(
        processed[segmentation_channel].analysis_image,
        config["segmentation"],
    )
    del cleaned_mask

    _notify(progress, "Measuring every ROI across all channels...")
    measurements, summary, measurement_thresholds = measure_rois(
        labels,
        processed,
        config["channels"],
        info,
        float(input_config["zoom"]),
    )

    measurement_csv = output_dir / "roi_measurements.csv"
    measurement_xlsx = output_dir / "roi_measurements.xlsx"
    measurements.to_csv(measurement_csv, index=False)
    channel_rows = [
        {
            "channel_index": channel.index,
            "channel_name": channel.name,
            "alias": config["channels"][str(channel.index)]["alias"],
            "pixel_type": channel.pixel_type,
            "color": channel.color,
            "measurement_threshold_raw": measurement_thresholds.get(
                slugify(
                    str(config["channels"][str(channel.index)]["alias"]),
                    fallback=f"channel_{channel.index}",
                )
            ),
        }
        for channel in info.channels
    ]
    _write_excel(measurement_xlsx, measurements, summary, config, channel_rows)

    _notify(progress, "Exporting labels, vector ROIs, and QC previews...")
    roi_paths = export_rois(
        labels,
        output_dir,
        save_label_image=bool(output_config.get("save_label_image", True)),
        save_imagej_rois=bool(output_config.get("save_imagej_rois", True)),
        save_geojson=bool(output_config.get("save_geojson", True)),
        simplify_tolerance_px=float(output_config.get("roi_simplify_tolerance_px", 1.0)),
    )
    preview_limit = int(output_config.get("preview_max_dimension_px", 2500))
    overlay_path = output_dir / "roi_overlay.png"
    save_overlay(
        processed[segmentation_channel].analysis_image,
        labels,
        overlay_path,
        max_dimension=preview_limit,
    )
    channel_panel_path = output_dir / "channel_previews.png"
    _save_channel_panel(
        processed,
        config["channels"],
        channel_panel_path,
        max_dimension=preview_limit,
    )

    used_config_path = output_dir / "config_used.yaml"
    save_yaml(config, used_config_path)
    metadata_path = output_dir / "czi_metadata.json"
    metadata_path.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
    result = {
        "input_czi": str(Path(czi_path).expanduser().resolve()),
        "output_dir": str(output_dir),
        "roi_count": int(labels.max()),
        "segmentation_channel": segmentation_channel,
        "segmentation_diagnostics": segmentation_diagnostics,
        "measurement_thresholds_raw": measurement_thresholds,
        "files": {
            "measurements_csv": str(measurement_csv),
            "measurements_xlsx": str(measurement_xlsx),
            "overlay": str(overlay_path),
            "channel_previews": str(channel_panel_path),
            "config_used": str(used_config_path),
            "metadata": str(metadata_path),
            **roi_paths,
        },
        "warnings": info.warnings,
    }
    result_path = output_dir / "analysis_summary.json"
    result["files"]["analysis_summary"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _notify(progress, f"Analysis complete: {int(labels.max())} ROIs")
    return result
