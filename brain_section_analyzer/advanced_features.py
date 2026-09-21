"""Optional 2-D measurements on existing masks; no new segmentation dependency."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure, morphology

FEATURE_DEFAULTS = {
    "cell_measurements_enabled": False,
    "radial_profiles_enabled": False,
    "spatial_distribution_enabled": False,
    "skeleton_enabled": False,
    "cell_expansion_um": 3.0,
    "radial_reference_channel": "abeta",
    "radial_max_um": 30.0,
    "radial_step_um": 5.0,
    "spatial_radius_um": 30.0,
    "object_channels": ["abeta", "iba1", "cd68"],
}
ADVANCED_TABLES = {
    "advanced_metrics": ("Advanced Metrics", "save_advanced_metrics_csv"),
    "object_distances": ("Object Distances", "save_object_distances_csv"),
    "cell_measurements": ("Cell Measurements", "save_cell_measurements_csv"),
    "radial_profiles": ("Radial Profiles", "save_radial_profiles_csv"),
    "spatial_objects": ("Spatial Objects", "save_spatial_objects_csv"),
    "skeleton_objects": ("Skeleton Objects", "save_skeleton_objects_csv"),
}
FEATURE_IMAGES = {
    "advanced_cell_regions": "Advanced: per-cell measurement regions",
    "advanced_radial_regions": "Advanced: radial measurement regions",
    "advanced_radial_plot": "Advanced: radial intensity profiles",
    "advanced_spatial_map": "Advanced: nearest-neighbour maps",
    "advanced_spatial_histogram": "Advanced: nearest-neighbour histograms",
    "advanced_skeleton": "Advanced: skeletons, endpoints and junction pixels",
}


def normalize_features(config, roles):
    advanced = config["advanced"]
    for key in FEATURE_DEFAULTS:
        if key.endswith("_enabled"):
            advanced[key] = bool(advanced[key])
    if advanced["cell_measurements_enabled"] and not config["microglia_count"]["enabled"]:
        raise ValueError("Per-cell measurements require Cell Counting.")
    for key in ("cell_expansion_um", "radial_max_um", "radial_step_um", "spatial_radius_um"):
        value = float(advanced[key])
        if not math.isfinite(value) or value < 0 or (key == "radial_step_um" and value == 0):
            raise ValueError(f"Advanced {key} must be finite and nonnegative (radial step > 0).")
        advanced[key] = value
    if math.ceil(advanced["radial_max_um"] / advanced["radial_step_um"]) > 256:
        raise ValueError("Radial profiles support at most 256 external bins; increase the bin width.")
    valid_roles = tuple(config["channels"])
    reference = advanced["radial_reference_channel"]
    if reference not in (roles if advanced["radial_profiles_enabled"] else valid_roles):
        if advanced["radial_profiles_enabled"]:
            raise ValueError("Radial profiles must select an enabled reference channel.")
        advanced["radial_reference_channel"] = valid_roles[0]
    selected = advanced["object_channels"]
    enabled = advanced["spatial_distribution_enabled"] or advanced["skeleton_enabled"]
    if not isinstance(selected, (list, tuple)):
        raise ValueError("Spatial / skeleton channels must select enabled channels.")
    if enabled and any(role not in roles for role in selected):
        raise ValueError("Spatial / skeleton channels must select enabled channels.")
    advanced["object_channels"] = list(dict.fromkeys(role for role in selected if role in valid_roles))
    if enabled and not selected:
        raise ValueError("Select at least one channel for spatial / skeleton analysis.")


def _territories(labels, domain, radius, pixel_x, pixel_y):
    """Nearest-mask territories, including interiors; ties use SciPy EDT's owner."""
    if not np.any(labels):
        return np.zeros(labels.shape, dtype=np.int32), np.full(labels.shape, np.nan)
    distances, indices = ndi.distance_transform_edt(labels == 0, sampling=(pixel_y, pixel_x), return_indices=True)
    owners = labels[indices[0], indices[1]]
    return np.where(domain & (distances <= radius), owners, 0), distances


def _intensity_record(image, positive, coords, background, pixel_area):
    values = np.asarray(image[coords], dtype=float)
    valid = np.isfinite(values)
    values = np.maximum(values[valid] - background, 0)
    selected = np.asarray(positive[coords])[valid]
    positive_values = values[selected]
    return {
        "finite_pixel_count": int(valid.sum()),
        "valid_area_um2": int(valid.sum()) * pixel_area,
        "mean_intensity": float(values.mean()) if values.size else np.nan,
        "integrated_intensity": float(values.sum()) if values.size else np.nan,
        "intensity_area_integral": float(values.sum()) * pixel_area if values.size else np.nan,
        "positive_fraction": float(selected.mean()) if selected.size else np.nan,
        "positive_area_um2": int(selected.sum()) * pixel_area,
        "positive_mean_intensity": float(positive_values.mean()) if positive_values.size else np.nan,
    }


def _skeleton_graph(mask, pixel_x, pixel_y):
    skeleton = morphology.skeletonize(mask)
    neighbors = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    endpoints = skeleton & (neighbors == 1)
    junctions = skeleton & (neighbors >= 3)
    # This is the calibrated 8-neighbour pixel graph, not fitted/subpixel branches.
    horizontal = np.count_nonzero(skeleton[:, :-1] & skeleton[:, 1:]) * pixel_x
    vertical = np.count_nonzero(skeleton[:-1] & skeleton[1:]) * pixel_y
    diagonal = (np.count_nonzero(skeleton[:-1, :-1] & skeleton[1:, 1:])
                + np.count_nonzero(skeleton[:-1, 1:] & skeleton[1:, :-1])) * math.hypot(pixel_x, pixel_y)
    return skeleton, endpoints, junctions, {
        "skeleton_pixel_count": int(skeleton.sum()),
        "skeleton_graph_length_um": float(horizontal + vertical + diagonal),
        "endpoint_count": int(endpoints.sum()),
        "junction_cluster_count": int(ndi.label(junctions, np.ones((3, 3)))[1]),
        "junction_pixel_count": int(junctions.sum()),
        "isolated_pixel_count": int((skeleton & (neighbors == 0)).sum()),
    }


def analyze_features(images, marker_labels, cell_labels, domain, config, pixel_x, pixel_y, source_file):
    """All regions are clipped to domain; expanded regions never overlap each other."""
    advanced = config["advanced"]
    tables, maps, summaries = {}, {}, []
    metadata = {"source_file": source_file, **config.get("metadata", {}),
                "scope": advanced["scope"], "dimensions": "2D"}
    intensity_columns = list(_intensity_record(np.zeros((1, 1)), np.zeros((1, 1), bool),
                                              (np.array([], int), np.array([], int)), 0, 1))
    channels = list(images)
    area = pixel_x * pixel_y
    positive = {role: marker_labels[role] > 0 for role in channels}
    # Only objects touching the scope act as seeds; retain their IDs after clipping.
    clipped = {role: np.where(domain, labels, 0) for role, labels in marker_labels.items()}
    if advanced["cell_measurements_enabled"]:
        seeds = np.where(domain, cell_labels, 0)
        regions, _ = _territories(seeds, domain, advanced["cell_expansion_um"], pixel_x, pixel_y)
        rows = []
        for region in measure.regionprops(regions):
            coords = tuple(region.coords.T)
            for role in channels:
                rows.append({**metadata, "cell_id": int(region.label), "channel": config["channels"][role]["alias"],
                    "expansion_um": advanced["cell_expansion_um"], "region_area_um2": region.area * area,
                    "background": advanced["backgrounds"][role],
                    **_intensity_record(images[role], positive[role], coords, advanced["backgrounds"][role], area)})
        columns = [*metadata, "cell_id", "channel", "expansion_um", "region_area_um2", "background", *intensity_columns]
        tables["cell_measurements"] = pd.DataFrame(rows, columns=columns)
        maps["advanced_cell_regions"] = ("cell_regions", regions, config["microglia_count"]["nucleus_channel"])
        summaries.append({**metadata, "analysis_type": "cell_measurements", "cell_count": len(measure.regionprops(regions)),
                          "cell_expansion_um": advanced["cell_expansion_um"]})
    if advanced["radial_profiles_enabled"]:
        reference = advanced["radial_reference_channel"]
        seeds = clipped[reference]
        owners, distance = _territories(seeds, domain, advanced["radial_max_um"], pixel_x, pixel_y)
        # Interiors get bin 0; exterior bins are (inner, outer] in calibrated um.
        bins = np.where(np.isfinite(distance), np.ceil(np.nan_to_num(distance) / advanced["radial_step_um"]), 0).astype(int)
        rows = []
        for region in measure.regionprops(owners):
            yy, xx = region.coords.T
            for index in np.unique(bins[yy, xx]):
                selected = bins[yy, xx] == index
                coords = (yy[selected], xx[selected])
                for role in channels:
                    rows.append({**metadata, "reference_channel": config["channels"][reference]["alias"],
                        "reference_object_id": int(region.label), "channel": config["channels"][role]["alias"],
                        "bin_index": int(index), "is_interior": bool(index == 0),
                        "bin_inner_um": max(0, (index - 1) * advanced["radial_step_um"]),
                        "bin_outer_um": min(index * advanced["radial_step_um"], advanced["radial_max_um"]),
                        "region_area_um2": int(selected.sum()) * area, "background": advanced["backgrounds"][role],
                        **_intensity_record(images[role], positive[role], coords, advanced["backgrounds"][role], area)})
        columns = [*metadata, "reference_channel", "reference_object_id", "channel", "bin_index", "is_interior",
                   "bin_inner_um", "bin_outer_um", "region_area_um2", "background", *intensity_columns]
        tables["radial_profiles"] = pd.DataFrame(rows, columns=columns)
        maps["advanced_radial_regions"] = ("radial_regions", np.where(owners > 0, bins + 1, 0), reference)
        maps["advanced_radial_plot"] = ("profile", tables["radial_profiles"], reference)
        summaries.append({**metadata, "analysis_type": "radial_profiles", "channel": config["channels"][reference]["alias"],
                          "reference_object_count": len(measure.regionprops(seeds)), "radial_max_um": advanced["radial_max_um"]})
    if advanced["spatial_distribution_enabled"] or advanced["skeleton_enabled"]:
        # Pad so the image boundary is treated as outside even for a full-image ROI.
        edge_distance = ndi.distance_transform_edt(np.pad(domain, 1), sampling=(pixel_y, pixel_x))[1:-1, 1:-1]
        spatial_rows, skeleton_rows = [], []
        for role in advanced["object_channels"]:
            labels = clipped[role]
            regions = measure.regionprops(labels)
            alias = config["channels"][role]["alias"]
            base = {**metadata, "channel": alias}
            if advanced["spatial_distribution_enabled"]:
                coords = np.asarray([r.centroid for r in regions], dtype=float).reshape(-1, 2) * [pixel_y, pixel_x]
                tree = cKDTree(coords) if regions else None
                nearest = tree.query(coords, k=2)[0][:, 1] if len(regions) > 1 else np.full(len(regions), np.nan)
                neighbors = tree.query_ball_point(coords, advanced["spatial_radius_um"], return_length=True) - 1 if tree else []
                lookup = np.full(int(labels.max()) + 1, np.nan)
                for index, region in enumerate(regions):
                    center = tuple(np.rint(region.centroid).astype(int))
                    spatial_rows.append({**base, "object_id": int(region.label),
                        "centroid_x_um": float(coords[index, 1]), "centroid_y_um": float(coords[index, 0]),
                        "nearest_neighbour_distance_um": float(nearest[index]), "neighbours_within_radius": int(neighbors[index]),
                        "radius_um": advanced["spatial_radius_um"],
                        "radius_roi_truncated": bool(edge_distance[center] <= advanced["spatial_radius_um"])})
                    lookup[region.label] = nearest[index]
                key = f"channel_{channels.index(role) + 1}"
                maps[f"advanced_spatial_map_{key}"] = ("spatial_heatmap", lookup[labels], role)
                maps[f"advanced_spatial_histogram_{key}"] = ("spatial_histogram", nearest[np.isfinite(nearest)], role)
                summaries.append({**base, "analysis_type": "spatial_distribution", "object_count": len(regions),
                    "density_per_mm2": len(regions) / (domain.sum() * area / 1e6) if domain.any() else np.nan,
                    "median_nearest_neighbour_distance_um": float(np.nanmedian(nearest)) if np.isfinite(nearest).any() else np.nan})
            if advanced["skeleton_enabled"]:
                full_skeleton, full_endpoints, full_junctions = (np.zeros(domain.shape, bool) for _ in range(3))
                for region in regions:
                    skeleton, endpoints, junctions, record = _skeleton_graph(region.image, pixel_x, pixel_y)
                    full_skeleton[region.slice] |= skeleton
                    full_endpoints[region.slice] |= endpoints
                    full_junctions[region.slice] |= junctions
                    skeleton_rows.append({**base, "object_id": int(region.label), "object_area_um2": region.area * area, **record})
                maps[f"advanced_skeleton_channel_{channels.index(role) + 1}"] = ("skeleton", (full_skeleton, full_endpoints, full_junctions), role)
        if advanced["spatial_distribution_enabled"]:
            tables["spatial_objects"] = pd.DataFrame(spatial_rows, columns=[*metadata, "channel", "object_id", "centroid_x_um",
                "centroid_y_um", "nearest_neighbour_distance_um", "neighbours_within_radius", "radius_um", "radius_roi_truncated"])
        if advanced["skeleton_enabled"]:
            tables["skeleton_objects"] = pd.DataFrame(skeleton_rows, columns=[*metadata, "channel", "object_id", "object_area_um2",
                "skeleton_pixel_count", "skeleton_graph_length_um", "endpoint_count", "junction_cluster_count",
                "junction_pixel_count", "isolated_pixel_count"])
    return tables, maps, pd.DataFrame(summaries)
