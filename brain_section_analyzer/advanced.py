"""Optional 2-D channel relationships; reuse SciPy and scikit-image, not plugin code."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from .advanced_features import FEATURE_DEFAULTS, normalize_features

ROLES = ("abeta", "iba1", "cd68", "dapi")
CHANNEL_COLORS = ("#00FF33", "#FF00FF", "#FF5900", "#0033FF", "#00FFFF", "#FFFF00")


def roles_for_count(count: int) -> tuple[str, ...]:
    """Return stable internal channel keys while preserving old four-channel sessions."""

    return tuple((*ROLES, *(f"channel_{index}" for index in range(5, count + 1)))[:count])
ADVANCED_DEFAULTS = {
    **FEATURE_DEFAULTS,
    "neighbour_enabled": False,
    "colocalization_enabled": False,
    "object_distances_enabled": False,
    "pairs": [],
    "scope": "roi",
    "backgrounds": {role: 0.0 for role in ROLES},
    "proximity_um": 10.0,
}
DRAWING_DEFAULTS = {
    "low_percentile": 1.0, "high_percentile": 99.7,
    "gamma": 1.0, "gain": 1.0, "opacity": 1.0,
    "show_object_ids": False, "font_size": 10, "dpi": 160,
    "background": "#FFFFFF", "heatmap_cmap": "viridis", "histogram_bins": 50, "density_log_scale": True,
    "boundaries": {
        name: {"color": color, "width_px": 1}
        for name, color in {
            "abeta": "#00FFFF", "iba1": "#00FF00", "cd68": "#FFFF00", "dapi": "#FFFFFF",
            "reference": "#00FFFF", "excluded": "#FF00FF", "no_nearby": "#FF8000",
            "nucleus": "#FFFFFF", "cell": "#FFFF00", "ring": "#FFFF00",
            "tissue": "#FF00FF", "overlap": "#FFFFFF",
            "skeleton": "#FFFF00", "endpoint": "#00FFFF", "junction": "#FF3300", "profile": "#FFFF00",
        }.items()
    },
}
STAGE_LABELS = {
    **{
        f"stage_channel_{index}_{stage}": f"Channel {index}: {title}"
        for index in range(1, 5)
        for stage, title in {
            "gaussian": "Gaussian image", "threshold": "Threshold mask",
            "morphology": "Morphology mask", "candidates": "Candidate labels",
            "accepted": "Accepted mask", "rejected": "Rejected mask",
        }.items()
    },
    **{
        f"stage_{name}": title for name, title in {
            "reference_cleaned": "Reference: morphology mask",
            "reference_after_exclusion": "Reference: after round/hollow exclusion",
            "reference_distance": "Reference: watershed distance map (px)",
            "reference_seeds": "Reference: watershed seed labels",
            "reference_watershed": "Reference: candidate / watershed labels",
            "reference_final": "Reference: final accepted labels",
            "reference_distance_um": "Reference: external distance map (um)",
            "nucleus_cleaned": "Nuclei: morphology mask",
            "nucleus_distance": "Nuclei: watershed distance map (px)",
            "nucleus_seeds": "Nuclei: watershed seed labels",
            "nucleus_candidates": "Nuclei: candidate / watershed labels",
            "nucleus_accepted": "Nuclei: accepted labels",
            "cell_accepted": "Cells: confirmed labels",
            "tissue": "Analysis ROI mask",
        }.items()
    },
}


def normalize_advanced(config, roles):
    advanced = config["advanced"]
    normalize_features(config, roles)
    for key in ("neighbour_enabled", "colocalization_enabled", "object_distances_enabled"):
        advanced[key] = bool(advanced[key])
    if advanced["scope"] not in {"roi", "cells", "reference_objects"}:
        raise ValueError("Advanced analysis scope must be roi, cells or reference_objects.")
    enabled = advanced["colocalization_enabled"] or advanced["object_distances_enabled"]
    pairs = advanced["pairs"]
    if not isinstance(pairs, (list, tuple)):
        raise ValueError("Advanced channel pairs must be a list.")
    cleaned = []
    configured_roles = tuple(config["channels"])
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("Each advanced pair must select two different enabled channels.")
        if any(role not in (roles if enabled else configured_roles) for role in pair):
            if enabled:
                raise ValueError("Each advanced pair must select two different enabled channels.")
            continue
        if list(pair) not in cleaned:
            cleaned.append(list(pair))
    advanced["pairs"] = cleaned
    if enabled and not cleaned:
        raise ValueError("Add at least one channel pair for advanced analysis.")
    scoped_enabled = enabled or any(advanced[key] for key in FEATURE_DEFAULTS if key.endswith("_enabled"))
    if scoped_enabled and advanced["scope"] == "cells" and not config["microglia_count"]["enabled"]:
        raise ValueError("Cell-scope advanced analysis requires cell counting.")
    if scoped_enabled and advanced["scope"] == "reference_objects" and not advanced["neighbour_enabled"]:
        raise ValueError("Reference-object scope requires neighbour analysis.")
    for role in configured_roles:
        value = float(advanced["backgrounds"].get(role, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Advanced intensity backgrounds must be finite and nonnegative.")
        advanced["backgrounds"][role] = value
    advanced["proximity_um"] = float(advanced["proximity_um"])
    if not math.isfinite(advanced["proximity_um"]) or advanced["proximity_um"] < 0:
        raise ValueError("Proximity distance must be finite and nonnegative.")


def analyze_relationships(images, marker_labels, domain, config, pixel_x, pixel_y, source_file, thresholds):
    """Intensity PCC/M1/M2 use raw background-subtracted pixels; overlap uses filtered masks.

    M1=sum(A where B>threshold)/sum(A); M2 is the converse. These are
    threshold-mask Manders, NOT Costes automatic thresholds or significance tests.
    Distances are directed A->B, calibrated centroid distances and pixel-center
    minimum mask distances (zero for overlap); no 3-D or subpixel surface claim.
    """
    advanced = config["advanced"]
    rows, object_rows, maps = [], [], {}
    if not (advanced["colocalization_enabled"] or advanced["object_distances_enabled"]):
        return pd.DataFrame(), pd.DataFrame(), maps
    channel_order = list(images)
    for a, b in advanced["pairs"]:
        mask_a, mask_b = (marker_labels[role] > 0 for role in (a, b))
        finite = domain & np.isfinite(images[a]) & np.isfinite(images[b])
        ai = np.maximum(np.asarray(images[a], dtype=float) - advanced["backgrounds"][a], 0)
        bi = np.maximum(np.asarray(images[b], dtype=float) - advanced["backgrounds"][b], 0)
        ma, mb = mask_a & finite, mask_b & finite
        overlap = ma & mb
        key = f"pair_{channel_order.index(a) + 1}_{channel_order.index(b) + 1}"
        record = {
            "source_file": source_file, **config.get("metadata", {}),
            "channel_a": config["channels"][a]["alias"], "channel_b": config["channels"][b]["alias"],
            "scope": advanced["scope"], "dimensions": "2D", "pixel_count": int(finite.sum()),
            "threshold_a": thresholds[a], "threshold_method_a": config["channels"][a]["threshold"]["method"],
            "threshold_b": thresholds[b], "threshold_method_b": config["channels"][b]["threshold"]["method"],
            "background_a": advanced["backgrounds"][a], "background_b": advanced["backgrounds"][b],
        }
        def ratio(num, den):
            return float(num / den) if den > 0 else float("nan")
        if advanced["colocalization_enabled"]:
            x, y = ai[finite], bi[finite]
            valid = x.size >= 2 and np.std(x) > 0 and np.std(y) > 0
            record.update({
                "pearson_r": float(measure.pearson_corr_coeff(ai, bi, mask=finite)[0]) if valid else float("nan"),
                "manders_m1": float(measure.manders_coloc_coeff(ai, mb, mask=finite)) if x.sum() > 0 else float("nan"),
                "manders_m2": float(measure.manders_coloc_coeff(bi, ma, mask=finite)) if y.sum() > 0 else float("nan"),
                "overlap_area_um2": int(overlap.sum()) * pixel_x * pixel_y,
                "overlap_fraction_of_a": ratio(overlap.sum(), ma.sum()),
                "overlap_fraction_of_b": ratio(overlap.sum(), mb.sum()),
                "jaccard": ratio(overlap.sum(), (ma | mb).sum()),
                "dice": ratio(2 * overlap.sum(), ma.sum() + mb.sum()),
            })
            maps[f"advanced_{key}_overlap"] = ("overlap", overlap, a)
            maps[f"advanced_{key}_scatter"] = ("scatter", (ai[finite], bi[finite]), (a, b))
        if advanced["object_distances_enabled"]:
            # Keep each object's original identity even when the ROI clips it.
            labels_a = np.where(domain, marker_labels[a], 0)
            labels_b = np.where(domain, marker_labels[b], 0)
            regions_a, regions_b = measure.regionprops(labels_a), measure.regionprops(labels_b)
            distance_map = ndi.distance_transform_edt(labels_b == 0, sampling=(pixel_y, pixel_x)) if regions_b else np.full(domain.shape, np.nan)
            tree = cKDTree(np.asarray([r.centroid for r in regions_b]) * [pixel_y, pixel_x]) if regions_b else None
            distances = []
            for region in regions_a:
                distance, index = tree.query(np.asarray(region.centroid) * [pixel_y, pixel_x]) if tree else (np.nan, None)
                edge = float(np.min(distance_map[region.coords[:, 0], region.coords[:, 1]]))
                distances.append(edge)
                object_rows.append({
                    **{k: record[k] for k in ("source_file", "channel_a", "channel_b", "scope", "dimensions")},
                    **config.get("metadata", {}), "object_a_id": int(region.label),
                    "nearest_centroid_object_b_id": int(regions_b[index].label) if index is not None else np.nan,
                    "nearest_centroid_distance_um": float(distance), "minimum_mask_distance_um": edge,
                    "overlap_fraction_of_object_a": float(np.mean(mask_b[region.coords[:, 0], region.coords[:, 1]])),
                    "within_proximity": bool(np.isfinite(edge) and edge <= advanced["proximity_um"]),
                    "target_available": bool(regions_b),
                })
            values = np.asarray(distances)
            valid_values = values[np.isfinite(values)]
            record.update({"object_a_count": len(regions_a), "object_b_count": len(regions_b),
                "median_minimum_mask_distance_um": float(np.median(valid_values)) if valid_values.size else np.nan,
                "proximity_um": advanced["proximity_um"],
                "fraction_a_within_proximity": float(np.mean(valid_values <= advanced["proximity_um"])) if valid_values.size else np.nan})
            maps[f"advanced_{key}_distance"] = ("heatmap", np.where(domain, distance_map, np.nan), b)
            maps[f"advanced_{key}_distance_histogram"] = ("histogram", valid_values, a)
        rows.append(record)
    object_columns = ["source_file", "channel_a", "channel_b", "scope", "dimensions", *config.get("metadata", {}).keys(),
        "object_a_id", "nearest_centroid_object_b_id", "nearest_centroid_distance_um",
        "minimum_mask_distance_um", "overlap_fraction_of_object_a", "within_proximity", "target_available"]
    return pd.DataFrame(rows), pd.DataFrame(object_rows, columns=object_columns), maps
