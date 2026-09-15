"""Batch configuration, execution, and aggregate result export."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from .config import create_default_config, normalize_config, save_yaml, slugify
from .image_io import SUPPORTED_IMAGE_SUFFIXES, inspect_image
from .pipeline import run_analysis

ProgressCallback = Callable[[str], None]


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _normalize_paths(image_paths: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for value in image_paths:
        path = Path(value).expanduser().resolve()
        key = str(path).casefold()
        if key in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        if path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
            raise ValueError(f"Expected one of {supported}: {path}")
        paths.append(path)
        seen.add(key)
    if not paths:
        raise ValueError("Select at least one supported microscopy image.")
    return paths


def _output_name(path: Path) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", path.stem).strip(" ._")
    return name or "image"


def _output_directories(paths: list[Path], output_root: Path) -> list[Path]:
    counts: dict[str, int] = {}
    directories: list[Path] = []
    for path in paths:
        base = _output_name(path)
        key = base.casefold()
        counts[key] = counts.get(key, 0) + 1
        suffix = "" if counts[key] == 1 else f"_{counts[key]}"
        directories.append(output_root / f"{base}{suffix}")
    return directories


def prepare_batch_config(
    image_path: str | Path,
    template_config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Adapt shared parameters to one image while preserving its metadata."""

    source = Path(image_path).expanduser().resolve()
    template_input = template_config.get("input", {})
    info = inspect_image(
        source,
        pixel_size_um_x=template_input.get("pixel_size_um_x"),
        pixel_size_um_y=template_input.get("pixel_size_um_y"),
        image_width_um=template_input.get("image_width_um"),
        image_height_um=template_input.get("image_height_um"),
    )
    zoom = float(template_input.get("zoom", 1.0))
    config = create_default_config(info, output_dir=output_dir, zoom=zoom)

    for key in (
        "scene",
        "time_index",
        "z_projection",
        "z_index",
        "zoom",
    ):
        if key in template_input:
            config["input"][key] = copy.deepcopy(template_input[key])
    available_channels = {channel.index for channel in info.channels}
    requested_channel = int(template_input.get("segmentation_channel", 0))
    config["input"]["segmentation_channel"] = (
        requested_channel if requested_channel in available_channels else info.channels[0].index
    )
    config["input"]["image_path"] = str(source)
    if source.suffix.casefold() == ".czi":
        config["input"]["czi_path"] = str(source)
    else:
        config["input"].pop("czi_path", None)
    config["input"]["output_dir"] = str(Path(output_dir).expanduser().resolve())

    if "segmentation" in template_config:
        config["segmentation"].update(copy.deepcopy(template_config["segmentation"]))
    if "output" in template_config:
        config["output"].update(copy.deepcopy(template_config["output"]))

    template_channels = template_config.get("channels", {})
    for channel in info.channels:
        key = str(channel.index)
        channel_name = slugify(channel.name or "")
        source_channel = next(
            (
                item
                for item in template_channels.values()
                if isinstance(item, dict)
                and channel_name
                and slugify(str(item.get("alias") or "")) == channel_name
            ),
            template_channels.get(key),
        )
        if not isinstance(source_channel, dict):
            continue
        target_channel = config["channels"][key]
        for parameter, value in source_channel.items():
            if parameter == "measurement_threshold" and isinstance(value, dict):
                target_channel[parameter].update(copy.deepcopy(value))
            else:
                target_channel[parameter] = copy.deepcopy(value)

    return normalize_config(config, info)


def _write_batch_summary(records: list[dict[str, Any]], output_root: Path) -> Path:
    path = output_root / "batch_summary.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def run_batch_analysis(
    czi_paths: Iterable[str | Path],
    template_config: dict[str, Any],
    output_root: str | Path,
    *,
    per_file_configs: dict[str, dict[str, Any]] | None = None,
    progress: ProgressCallback | None = print,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Analyze multiple microscopy images and create combined outputs."""

    paths = _normalize_paths(czi_paths)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_directories = _output_directories(paths, root)
    records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    combined_measurements: list[pd.DataFrame] = []
    specific_configs = {
        str(Path(key).expanduser().resolve()).casefold(): value
        for key, value in (per_file_configs or {}).items()
    }

    selected_files_txt = root / "selected_image_files.txt"
    selected_files_txt.write_text(
        "\n".join(str(path) for path in paths) + "\n", encoding="utf-8"
    )
    batch_parameters_yaml = root / "batch_parameters.yaml"
    batch_parameter_document: dict[str, Any] = {
        "format_version": 2,
        "selected_image_files": [str(path) for path in paths],
        "output_root": str(root),
        "template_config": copy.deepcopy(template_config),
        "per_file_overrides": {
            str(path): copy.deepcopy(specific_configs[str(path).casefold()])
            for path in paths
            if str(path).casefold() in specific_configs
        },
        "effective_file_configs": {},
    }
    if all(path.suffix.casefold() == ".czi" for path in paths):
        # Preserve the legacy key for existing CZI-only workflows.
        batch_parameter_document["selected_czi_files"] = [str(path) for path in paths]
    save_yaml(batch_parameter_document, batch_parameters_yaml)

    for file_index, (path, output_dir) in enumerate(
        zip(paths, output_directories), start=1
    ):
        prefix = f"[{file_index}/{len(paths)}] {path.name}"
        _notify(progress, f"{prefix}: preparing configuration...")
        try:
            file_template = specific_configs.get(str(path).casefold(), template_config)
            config = prepare_batch_config(path, file_template, output_dir)
            batch_parameter_document["effective_file_configs"][str(path)] = copy.deepcopy(config)
            result = run_analysis(
                config,
                progress=lambda message, item=prefix: _notify(
                    progress, f"{item}: {message}"
                ),
            )
            results.append(result)
            record = {
                "batch_file_index": file_index,
                "source_file": str(path),
                "source_name": path.name,
                "status": "complete",
                "roi_count": int(result["roi_count"]),
                "output_dir": str(output_dir),
                "error": "",
            }
            measurement_path = Path(result["files"]["measurements_csv"])
            measurements = pd.read_csv(measurement_path)
            measurements.insert(0, "source_name", path.name)
            measurements.insert(0, "source_file", str(path))
            measurements.insert(0, "batch_file_index", file_index)
            combined_measurements.append(measurements)
            _notify(
                progress,
                f"{prefix}: complete with {int(result['roi_count'])} ROIs.",
            )
        except Exception as error:
            record = {
                "batch_file_index": file_index,
                "source_file": str(path),
                "source_name": path.name,
                "status": "failed",
                "roi_count": 0,
                "output_dir": str(output_dir),
                "error": str(error),
            }
            _notify(progress, f"{prefix}: failed: {error}")
            if not continue_on_error:
                records.append(record)
                _write_batch_summary(records, root)
                raise
        records.append(record)
        _write_batch_summary(records, root)

    save_yaml(batch_parameter_document, batch_parameters_yaml)

    summary = pd.DataFrame(records)
    summary_csv = root / "batch_summary.csv"
    summary_xlsx = root / "batch_summary.xlsx"
    summary.to_excel(summary_xlsx, index=False)

    combined = (
        pd.concat(combined_measurements, ignore_index=True, sort=False)
        if combined_measurements
        else pd.DataFrame(columns=["batch_file_index", "source_file", "source_name"])
    )
    combined_csv = root / "combined_roi_measurements.csv"
    combined_xlsx = root / "combined_roi_measurements.xlsx"
    combined.to_csv(combined_csv, index=False)
    with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="ROI Measurements", index=False)
        summary.to_excel(writer, sheet_name="Batch Summary", index=False)

    completed = int((summary["status"] == "complete").sum())
    failed = int((summary["status"] == "failed").sum())
    batch_result: dict[str, Any] = {
        "output_root": str(root),
        "file_count": len(paths),
        "completed": completed,
        "failed": failed,
        "total_roi_count": int(summary["roi_count"].sum()),
        "records": records,
        "results": results,
        "files": {
            "batch_summary_csv": str(summary_csv),
            "batch_summary_xlsx": str(summary_xlsx),
            "combined_measurements_csv": str(combined_csv),
            "combined_measurements_xlsx": str(combined_xlsx),
            "selected_files_txt": str(selected_files_txt),
            "batch_parameters_yaml": str(batch_parameters_yaml),
        },
    }
    summary_json = root / "batch_analysis_summary.json"
    batch_result["files"]["batch_summary_json"] = str(summary_json)
    summary_json.write_text(
        json.dumps(batch_result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    _notify(
        progress,
        f"Batch complete: {completed} completed, {failed} failed, "
        f"{batch_result['total_roi_count']} total ROIs.",
    )
    return batch_result
