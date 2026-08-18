"""ROI labels, ImageJ ROI ZIP, GeoJSON, and preview image export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image
from roifile import ImagejRoi, roiwrite
from skimage import measure, segmentation, transform


def _largest_contour(mask: np.ndarray, tolerance: float) -> np.ndarray | None:
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    if tolerance > 0:
        contour = measure.approximate_polygon(contour, tolerance=tolerance)
    if len(contour) < 3:
        return None
    # Convert row-column coordinates to ImageJ and GeoJSON x-y coordinates.
    return np.column_stack([contour[:, 1], contour[:, 0]])


def export_rois(
    labels: np.ndarray,
    output_dir: str | Path,
    *,
    save_label_image: bool,
    save_imagej_rois: bool,
    save_geojson: bool,
    simplify_tolerance_px: float,
) -> dict[str, str]:
    """Export every label as interoperable raster and vector ROI formats."""

    destination = Path(output_dir)
    paths: dict[str, str] = {}
    if save_label_image:
        label_path = destination / "roi_labels.tiff"
        tifffile.imwrite(label_path, labels.astype(np.uint32), compression="zlib")
        paths["label_image"] = str(label_path)

    contours: list[tuple[int, np.ndarray]] = []
    for roi_id in range(1, int(labels.max()) + 1):
        contour = _largest_contour(labels == roi_id, simplify_tolerance_px)
        if contour is not None:
            contours.append((roi_id, contour))

    if save_imagej_rois and contours:
        imagej_rois = [
            ImagejRoi.frompoints(points, name=f"ROI_{roi_id:05d}")
            for roi_id, points in contours
        ]
        roi_path = destination / "imagej_rois.zip"
        roiwrite(roi_path, imagej_rois, mode="w")
        paths["imagej_rois"] = str(roi_path)

    if save_geojson:
        features: list[dict[str, Any]] = []
        for roi_id, contour in contours:
            coordinates = contour.tolist()
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            features.append(
                {
                    "type": "Feature",
                    "properties": {"roi_id": roi_id, "coordinate_space": "analysis_pixels"},
                    "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                }
            )
        geojson_path = destination / "rois.geojson"
        geojson_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
            encoding="utf-8",
        )
        paths["geojson"] = str(geojson_path)
    return paths


def save_overlay(
    analysis_image: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    max_dimension: int,
) -> None:
    """Save a memory-bounded red-boundary segmentation preview."""

    height, width = analysis_image.shape
    scale = min(1.0, float(max_dimension) / max(height, width))
    if scale < 1.0:
        shape = (max(1, round(height * scale)), max(1, round(width * scale)))
        image = transform.resize(
            analysis_image,
            shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
        label_preview = transform.resize(
            labels,
            shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.int32)
    else:
        image = analysis_image
        label_preview = labels
    finite = image[np.isfinite(image)]
    if finite.size:
        low = float(np.min(finite))
        high = float(np.max(finite))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    base = np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=2)
    boundaries = segmentation.find_boundaries(label_preview, mode="outer")
    rgb[boundaries] = np.array([255, 48, 48], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)
