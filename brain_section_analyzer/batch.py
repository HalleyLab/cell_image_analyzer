"""Batch execution and animal-level aggregation."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from cell_analyzer.image_io import SUPPORTED_IMAGE_SUFFIXES

from .analysis import _export_table, run_analysis
from .config import load_config, save_config


ProgressCallback = Callable[[str], None]


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _normalize_paths(paths: Iterable[str | Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        key = str(path).casefold()
        if key in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")
        if path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
            raise ValueError(f"Unsupported image format: {path}; expected {supported}")
        result.append(path)
        seen.add(key)
    if not result:
        raise ValueError("Select at least one supported microscopy image.")
    return result


def _safe_name(path: Path) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", path.stem).strip(" ._") or "image"


def _metadata_table(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {source}")
    table = pd.read_csv(source)
    if "source_file" not in table.columns and "source_name" not in table.columns:
        raise ValueError("Metadata CSV needs a source_file or source_name column.")
    return table


def _metadata_for(path: Path, table: pd.DataFrame) -> dict[str, Any]:
    if table.empty:
        return {}
    matches = pd.Series(False, index=table.index)
    if "source_file" in table.columns:
        normalized = table["source_file"].astype(str).map(
            lambda value: str(Path(value).expanduser().resolve()).casefold()
        )
        matches |= normalized == str(path).casefold()
    if "source_name" in table.columns:
        matches |= table["source_name"].astype(str).str.casefold() == path.name.casefold()
    selected = table.loc[matches]
    if selected.empty:
        return {}
    if len(selected) > 1:
        raise ValueError(f"Metadata contains multiple rows for {path.name}")
    ignored = {"source_file", "source_name"}
    return {
        str(key): (None if pd.isna(value) else value)
        for key, value in selected.iloc[0].items()
        if key not in ignored
    }


def _resolve_mask(config: dict[str, Any], image_path: Path) -> None:
    roi = config["tissue_roi"]
    if str(roi.get("mode", "full_image")).lower() != "mask_directory":
        return
    directory = Path(roi.get("mask_directory") or "").expanduser().resolve()
    suffix = str(roi.get("mask_suffix") or "_mask.png")
    candidate = directory / f"{image_path.stem}{suffix}"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"No matching tissue mask for {image_path.name}; expected {candidate}"
        )
    roi["mode"] = "mask"
    roi["mask_path"] = str(candidate)


def _animal_summary(images: pd.DataFrame, plaques: pd.DataFrame) -> pd.DataFrame:
    if images.empty or "mouse_id" not in images.columns or images["mouse_id"].isna().all():
        return pd.DataFrame(
            columns=[
                "genotype",
                "mouse_id",
                "region",
                "sex",
                "image_count",
                "roi_area_um2",
                "roi_area_mm2",
                "plaque_count_all",
                "plaque_count_boundary",
                "plaque_count_interior",
                "plaque_density_all_per_mm2",
            ]
        )
    group_keys = [
        key
        for key in ("genotype", "mouse_id", "region", "sex")
        if key in images.columns and not images[key].isna().all()
    ]
    if "mouse_id" not in group_keys:
        return pd.DataFrame(columns=["mouse_id", "image_count"])
    records: list[dict[str, Any]] = []
    for group_values, frame in images.groupby(group_keys, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        record = dict(zip(group_keys, group_values))
        record["image_count"] = int(len(frame))
        roi_area = float(frame["roi_area_um2"].sum())
        record["roi_area_um2"] = roi_area
        record["roi_area_mm2"] = roi_area / 1_000_000.0
        for count_column in ("plaque_count_all", "plaque_count_boundary", "plaque_count_interior"):
            if count_column in frame.columns:
                record[count_column] = int(frame[count_column].sum())
        record["plaque_density_all_per_mm2"] = (
            record.get("plaque_count_all", 0) / record["roi_area_mm2"]
            if record["roi_area_mm2"] > 0
            else np.nan
        )
        additive = [
            column
            for column in frame.columns
            if column.endswith("_area_um2") and column not in {"roi_area_um2"}
        ]
        for column in additive:
            record[column] = float(frame[column].sum())
        for role in ("abeta", "iba1", "cd68"):
            area_col = f"{role}_positive_area_um2"
            if area_col in record:
                record[f"{role}_positive_fraction"] = record[area_col] / roi_area if roi_area else np.nan
        if "cd68_in_iba1_area_um2" in record and record.get("iba1_positive_area_um2", 0) > 0:
            record["cd68_in_iba1_fraction_of_iba1"] = (
                record["cd68_in_iba1_area_um2"] / record["iba1_positive_area_um2"]
            )
        ring_area_columns = [
            column for column in record if column.startswith("ring_") and column.endswith("_area_um2")
        ]
        for ring_area_column in ring_area_columns:
            if "_positive_" in ring_area_column or "_in_" in ring_area_column:
                continue
            prefix = ring_area_column[: -len("_area_um2")]
            ring_area = record[ring_area_column]
            for role in ("abeta", "iba1", "cd68"):
                area_col = f"{prefix}_{role}_positive_area_um2"
                if area_col in record:
                    record[f"{prefix}_{role}_positive_fraction"] = (
                        record[area_col] / ring_area if ring_area else np.nan
                    )
            intersection = f"{prefix}_cd68_in_iba1_area_um2"
            iba1_area = f"{prefix}_iba1_positive_area_um2"
            if intersection in record and record.get(iba1_area, 0) > 0:
                record[f"{prefix}_cd68_in_iba1_fraction_of_iba1"] = (
                    record[intersection] / record[iba1_area]
                )
        if not plaques.empty:
            plaque_frame = plaques.copy()
            for key, value in record.items():
                if key in group_keys and key in plaque_frame.columns:
                    if pd.isna(value):
                        plaque_frame = plaque_frame[plaque_frame[key].isna()]
                    else:
                        plaque_frame = plaque_frame[plaque_frame[key] == value]
            if not plaque_frame.empty and "plaque_area_um2" in plaque_frame:
                record["median_interior_plaque_area_um2"] = float(
                    plaque_frame["plaque_area_um2"].median()
                )
                record["mean_interior_plaque_area_um2"] = float(
                    plaque_frame["plaque_area_um2"].mean()
                )
        records.append(record)
    return pd.DataFrame.from_records(records)


def run_batch_analysis(
    image_paths: Iterable[str | Path],
    config_or_path: dict[str, Any] | str | Path,
    output_root: str | Path,
    *,
    per_file_configs: dict[str, dict[str, Any]] | None = None,
    progress: ProgressCallback | None = print,
) -> dict[str, Any]:
    """Analyze multiple images without changing the original cell-analyzer package."""

    paths = _normalize_paths(image_paths)
    template = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else copy.deepcopy(config_or_path)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = _metadata_table(template.get("batch", {}).get("metadata_csv"))
    continue_on_error = bool(template.get("batch", {}).get("continue_on_error", True))
    specific_configs = {
        str(Path(key).expanduser().resolve()).casefold(): copy.deepcopy(value)
        for key, value in (per_file_configs or {}).items()
    }
    selected_path = root / "selected_image_files.txt"
    selected_path.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
    config_path = root / "batch_parameters.yaml"
    batch_parameter_document: dict[str, Any] = {
        "format_version": 2,
        "selected_image_files": [str(path) for path in paths],
        "output_root": str(root),
        "template_config": copy.deepcopy(template),
        "per_file_overrides": {
            str(path): copy.deepcopy(specific_configs[str(path).casefold()])
            for path in paths
            if str(path).casefold() in specific_configs
        },
        "effective_file_configs": {},
    }
    save_config(batch_parameter_document, config_path)

    records: list[dict[str, Any]] = []
    image_tables: list[pd.DataFrame] = []
    plaque_tables: list[pd.DataFrame] = []
    candidate_tables: list[pd.DataFrame] = []
    marker_tables: list[pd.DataFrame] = []
    microglia_tables: list[pd.DataFrame] = []
    ring_tables: list[pd.DataFrame] = []
    used_names: dict[str, int] = {}
    for index, path in enumerate(paths, start=1):
        base = _safe_name(path)
        used_names[base.casefold()] = used_names.get(base.casefold(), 0) + 1
        suffix = "" if used_names[base.casefold()] == 1 else f"_{used_names[base.casefold()]}"
        output_dir = root / f"{base}{suffix}"
        file_template = specific_configs.get(str(path).casefold(), template)
        file_config = copy.deepcopy(file_template)
        file_config.setdefault("input", {})["image_path"] = str(path)
        file_config["input"]["output_dir"] = str(output_dir)
        file_config.setdefault("metadata", {}).update(_metadata_for(path, metadata))
        try:
            _resolve_mask(file_config, path)
            batch_parameter_document["effective_file_configs"][str(path)] = copy.deepcopy(
                file_config
            )
            _notify(progress, f"[{index}/{len(paths)}] {path.name}: starting...")
            result = run_analysis(
                file_config,
                progress=lambda message, prefix=path.name: _notify(progress, f"{prefix}: {message}"),
                include_tables=True,
            )
            tables = result.pop("_tables")
            image_table = _export_table(tables["image_summary"], file_config, reverse=True)
            plaque_table = _export_table(tables["primary_objects"], file_config, reverse=True)
            candidate_table = _export_table(tables["candidate_qc"], file_config, reverse=True)
            marker_table = _export_table(tables["channel_objects"], file_config, reverse=True)
            microglia_table = _export_table(tables["cells"], file_config, reverse=True)
            ring_table = _export_table(tables["ring_metrics"], file_config, reverse=True)
            image_table.insert(0, "batch_file_index", index)
            image_table.insert(1, "source_name", path.name)
            if not plaque_table.empty:
                plaque_table.insert(0, "batch_file_index", index)
                plaque_table.insert(1, "source_name", path.name)
            if not candidate_table.empty:
                candidate_table.insert(0, "batch_file_index", index)
                candidate_table.insert(1, "source_name", path.name)
            if not marker_table.empty:
                marker_table.insert(0, "batch_file_index", index)
                marker_table.insert(1, "source_name", path.name)
            for table in (microglia_table, ring_table):
                if not table.empty:
                    table.insert(0, "batch_file_index", index)
                    table.insert(1, "source_name", path.name)
            image_tables.append(image_table)
            plaque_tables.append(plaque_table)
            candidate_tables.append(candidate_table)
            marker_tables.append(marker_table)
            microglia_tables.append(microglia_table)
            ring_tables.append(ring_table)
            records.append(
                {
                    "batch_file_index": index,
                    "source_file": str(path),
                    "source_name": path.name,
                    "status": "complete",
                    "primary_object_count_all": result["plaque_count_all"],
                    "output_dir": str(output_dir),
                    "error": "",
                }
            )
        except Exception as error:
            records.append(
                {
                    "batch_file_index": index,
                    "source_file": str(path),
                    "source_name": path.name,
                    "status": "failed",
                    "primary_object_count_all": 0,
                    "output_dir": str(output_dir),
                    "error": str(error),
                }
            )
            _notify(progress, f"[{index}/{len(paths)}] {path.name}: failed: {error}")
            if not continue_on_error:
                raise

    save_config(batch_parameter_document, config_path)

    batch_summary = pd.DataFrame.from_records(records)
    combined_images = pd.concat(image_tables, ignore_index=True, sort=False) if image_tables else pd.DataFrame()
    nonempty_plaques = [table for table in plaque_tables if not table.empty]
    combined_plaques = (
        pd.concat(nonempty_plaques, ignore_index=True, sort=False)
        if nonempty_plaques
        else (
            plaque_tables[0].iloc[0:0].copy()
            if plaque_tables
            else pd.DataFrame(columns=["source_file", "plaque_id", "plaque_area_um2"])
        )
    )
    nonempty_candidates = [table for table in candidate_tables if not table.empty]
    combined_candidates = (
        pd.concat(nonempty_candidates, ignore_index=True, sort=False)
        if nonempty_candidates
        else (
            candidate_tables[0].iloc[0:0].copy()
            if candidate_tables
            else pd.DataFrame(columns=["source_file", "candidate_id", "neuron_like_excluded"])
        )
    )
    nonempty_markers = [table for table in marker_tables if not table.empty]
    combined_markers = (
        pd.concat(nonempty_markers, ignore_index=True, sort=False)
        if nonempty_markers
        else (
            marker_tables[0].iloc[0:0].copy()
            if marker_tables
            else pd.DataFrame(
                columns=["source_file", "marker", "component_id", "accepted"]
            )
        )
    )
    combined_microglia = (
        pd.concat([table for table in microglia_tables if not table.empty], ignore_index=True, sort=False)
        if any(not table.empty for table in microglia_tables)
        else (microglia_tables[0].iloc[0:0].copy() if microglia_tables else pd.DataFrame())
    )
    combined_rings = (
        pd.concat([table for table in ring_tables if not table.empty], ignore_index=True, sort=False)
        if any(not table.empty for table in ring_tables)
        else (ring_tables[0].iloc[0:0].copy() if ring_tables else pd.DataFrame())
    )
    animals = _animal_summary(combined_images, combined_plaques)
    batch_summary_path = root / "batch_summary.csv"
    image_path = root / "combined_image_summary.csv"
    plaque_path = root / "combined_primary_object_measurements.csv"
    candidate_path = root / "combined_channel_1_candidate_qc.csv"
    marker_path = root / "combined_channel_object_qc.csv"
    microglia_path = root / "combined_cells.csv"
    ring_path = root / "combined_primary_object_ring_metrics.csv"
    animal_path = root / "animal_summary.csv"
    excel_path = root / "cell_analysis_batch_results.xlsx"
    public_images = _export_table(combined_images, template)
    public_objects = _export_table(combined_plaques, template)
    public_candidates = _export_table(combined_candidates, template)
    public_channels = _export_table(combined_markers, template)
    public_cells = _export_table(combined_microglia, template)
    public_rings = _export_table(combined_rings, template)
    public_animals = _export_table(animals, template)
    batch_summary.to_csv(batch_summary_path, index=False)
    output = template.get("output", {})
    result_files = {
        "selected_image_files": str(selected_path),
        "batch_parameters": str(config_path),
        "batch_summary": str(batch_summary_path),
    }
    combined_outputs = (
        ("save_image_summary_csv", "combined_image_summary", image_path, public_images),
        ("save_primary_objects_csv", "combined_plaque_measurements", plaque_path, public_objects),
        ("save_candidate_qc_csv", "combined_abeta_candidate_qc", candidate_path, public_candidates),
        ("save_channel_objects_csv", "combined_marker_component_qc", marker_path, public_channels),
        ("save_cells_csv", "combined_microglia_cells", microglia_path, public_cells),
        ("save_ring_metrics_csv", "combined_plaque_ring_metrics", ring_path, public_rings),
        ("save_animal_summary_csv", "animal_summary", animal_path, public_animals),
    )
    for flag, key, path, table in combined_outputs:
        if bool(output.get(flag, True)):
            table.to_csv(path, index=False)
            result_files[key] = str(path)
    if bool(output.get("save_excel", True)):
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            batch_summary.to_excel(writer, sheet_name="Batch Summary", index=False)
            public_images.to_excel(writer, sheet_name="Image Summary", index=False)
            public_objects.to_excel(writer, sheet_name="Primary Objects", index=False)
            public_candidates.to_excel(writer, sheet_name="Channel 1 QC", index=False)
            public_channels.to_excel(writer, sheet_name="Channel Objects", index=False)
            public_cells.to_excel(writer, sheet_name="Cells", index=False)
            public_rings.to_excel(writer, sheet_name="Object Ring Metrics", index=False)
            public_animals.to_excel(writer, sheet_name="Animal Summary", index=False)
        result_files["excel"] = str(excel_path)
    result = {
        "output_root": str(root),
        "file_count": len(paths),
        "completed": int((batch_summary["status"] == "complete").sum()),
        "failed": int((batch_summary["status"] == "failed").sum()),
        "per_file_parameter_count": len(specific_configs),
        "files": result_files,
        "records": records,
    }
    json_path = root / "batch_analysis_summary.json"
    result["files"]["json"] = str(json_path)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _notify(progress, f"Batch complete: {result['completed']} completed, {result['failed']} failed.")
    return result
