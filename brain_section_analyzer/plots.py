"""Shared rendering for saved processing images and the Outputs preview."""
from __future__ import annotations
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, LogNorm
import numpy as np
from scipy import ndimage as ndi
from skimage import segmentation
from PIL import Image, ImageDraw, ImageFont
from .advanced import DRAWING_DEFAULTS, ROLES, STAGE_LABELS
from .advanced_features import FEATURE_IMAGES


def _display_scale(image, settings=None):
    settings = settings or DRAWING_DEFAULTS
    values = np.asarray(image, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=float)
    positive = finite[finite > 0]
    basis = positive if positive.size else finite
    low, high = np.percentile(basis, [settings["low_percentile"], settings["high_percentile"]])
    if high <= low:
        low = 0.0
        high = max(float(high), 1.0)
    scaled = np.clip(np.nan_to_num((values - low) / (high - low), nan=0), 0, 1)
    return np.clip(scaled ** (1.0 / settings["gamma"]) * settings["gain"], 0, 1)


def _tint(image, color):
    return np.clip(np.asarray(image)[..., None] * np.asarray(color), 0, 1)


def _overlay(image, boundary, color, width_px=1, opacity=1.0):
    result = np.asarray(image).copy()
    width_px = max(1, int(width_px))
    if width_px > 1:
        boundary = ndi.maximum_filter(boundary.astype(np.uint8), size=width_px) > 0
    result[boundary] = result[boundary] * (1 - opacity) + np.asarray(color) * opacity
    return result


def _selected_qc_panels(panels, selected):
    selected = set(selected)
    return [panel for panel in panels if panel[0] in selected or
            any(panel[0].startswith(key) for key in selected & FEATURE_IMAGES.keys())]


def _with_ids(panel, labels, settings):
    if not settings["show_object_ids"]:
        return panel
    image = Image.fromarray(np.uint8(np.clip(panel, 0, 1) * 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", settings["font_size"])
    except OSError:
        font = ImageFont.load_default()
    for label, slices in enumerate(ndi.find_objects(labels.astype(int)), start=1):
        if slices is None:
            continue
        yy, xx = np.where(labels[slices] == label)
        if yy.size:
            position = (int(xx.mean() + slices[1].start), int(yy.mean() + slices[0].start))
            draw.text(position, str(label), fill="white", font=font, stroke_width=1, stroke_fill="black")
    return np.asarray(image) / 255.0


def save_qc(path, raw_images, products, max_dimension, channel_colors, channel_names,
            cell_count_config, reference_role, selected_qc_panels, processing_dir=None,
            save_raw_channels=True, save_composite=True, save_segmentation=True,
            save_mask_images=True, ring_boundary_width_px=1, save_stages=True,
            save_advanced=True, drawing=None, processing_steps=None):
    settings = drawing or DRAWING_DEFAULTS
    shape = products.tissue_mask.shape
    stride = max(1, int(math.ceil(max(shape) / max_dimension)))
    sl = (slice(None, None, stride), slice(None, None, stride))
    tinted = {role: _tint(_display_scale(image, settings)[sl], channel_colors[role]) for role, image in raw_images.items()}
    composite = np.clip(sum(tinted.values()), 0, 1)
    panels = []
    saved = {}
    selected = None if processing_steps is None else set(processing_steps)
    qc_selected = set(selected_qc_panels)
    group_enabled = {
        "raw": save_raw_channels,
        "composite": save_composite,
        "segmentation": save_segmentation,
        "mask": save_mask_images,
        "stage": save_stages,
        "advanced": save_advanced,
    }

    def requested(name, group, selection_key=None):
        qc = path is not None and (
            name in qc_selected
            or (selection_key or name) in qc_selected
            or any(
                name.startswith(key)
                for key in qc_selected & FEATURE_IMAGES.keys()
            )
        )
        processing = (
            processing_dir is not None
            and group_enabled[group]
            and (selected is None or (selection_key or name) in selected)
        )
        return qc, processing

    def boundary(image, labels, name):
        style = settings["boundaries"][name]
        width = style["width_px"]
        if drawing is None and name == "ring":
            width = ring_boundary_width_px
        result = _overlay(image, segmentation.find_boundaries(labels, mode="inner"), to_rgb(style["color"]), width, settings["opacity"])
        return _with_ids(result, labels, settings) if labels.dtype != bool else result

    def add(name, title, image, group, selection_key=None):
        qc, processing = requested(name, group, selection_key)
        if not (qc or processing):
            return
        image = image() if callable(image) else image
        if qc:
            panels.append((name, title, image))
        if processing:
            processing_dir.mkdir(parents=True, exist_ok=True)
            filename = processing_dir / f"{name}.png"
            Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).save(filename, dpi=(settings["dpi"], settings["dpi"]))
            saved[f"processing_{name}"] = str(filename)

    for index, role in enumerate(tinted, 1):
        add(
            f"raw_channel_{index}",
            f"{channel_names[role]}: raw",
            lambda role=role: tinted[role],
            "raw",
        )
    add("05_composite", "Composite", lambda: composite, "composite")

    def reference_panel():
        panel = boundary(tinted[reference_role], products.plaque_labels[sl], "reference")
        panel = boundary(panel, products.neuron_like_mask[sl], "excluded")
        return boundary(panel, products.microglia_absent_plaque_mask[sl], "no_nearby")

    if products.neighbour_enabled:
        add("06_primary_object_segmentation", f"{channel_names[reference_role]}: neighbour reference objects", reference_panel, "segmentation")
    for index, role in enumerate(tinted, 1):
        add(f"07_channel_{index}_objects", f"{channel_names[role]}: accepted objects",
            lambda role=role: boundary(tinted[role], products.marker_labels[role][sl], role), "segmentation")
        add(f"mask_channel_{index}_object_filter_mask", f"{channel_names[role]}: accepted mask",
            lambda role=role: _tint((products.marker_labels[role][sl] > 0).astype(float), channel_colors[role]), "mask")
    if products.neighbour_enabled:
        for name, mask, style in (
            ("neighbour_reference_object_mask", products.plaque_labels > 0, "reference"),
            ("neighbour_excluded_object_mask", products.neuron_like_mask, "excluded"),
            ("neighbour_no_nearby_cell_excluded", products.microglia_absent_plaque_mask, "no_nearby"),
        ):
            add(f"mask_{name}", name.replace("_", " "),
                lambda mask=mask, style=style: _tint(mask[sl].astype(float), to_rgb(settings["boundaries"][style]["color"])), "mask")
    nucleus_role = cell_count_config.get("nucleus_channel")
    if cell_count_config.get("enabled") and nucleus_role in tinted:
        def cell_panel():
            panel = boundary(tinted[nucleus_role], products.nucleus_labels[sl], "nucleus")
            return boundary(panel, products.microglia_labels[sl], "cell")

        add("09_cell_counting", f"{channel_names[nucleus_role]}: detected / confirmed cells", cell_panel, "segmentation")
        for name, labels, style in (("nucleus_mask", products.nucleus_labels, "nucleus"), ("cell_mask", products.microglia_labels, "cell")):
            add(f"mask_{name}", name.replace("_", " "),
                lambda labels=labels, style=style: _tint((labels[sl] > 0).astype(float), to_rgb(settings["boundaries"][style]["color"])), "mask")
    if products.ring_masks:
        union = np.zeros(shape, dtype=bool)
        for name, mask in products.ring_masks.items():
            union |= mask
            add(f"mask_{name}", f"{name}: includes object interior",
                lambda mask=mask: _tint(mask[sl].astype(float), to_rgb(settings["boundaries"]["ring"]["color"])), "mask", "individual_rings")
            add(f"overlay_{name}", name,
                lambda mask=mask: boundary(composite, mask[sl], "ring"), "segmentation", "individual_rings")
        add("10_primary_object_distance_rings", "Cumulative neighbour ranges (interior included)",
            lambda: boundary(composite, union[sl], "ring"), "segmentation")

    def tissue_panel():
        panel = composite.copy()
        panel[~products.tissue_mask[sl]] *= 0.15
        return boundary(panel, products.tissue_mask[sl], "tissue")

    add("11_tissue_roi", "Analysis ROI", tissue_panel, "segmentation")

    for name, (kind, array, role) in {**products.stages, **products.advanced_maps}.items():
        title = STAGE_LABELS.get(name, name.replace("_", " ").title())
        group = "advanced" if name.startswith("advanced_") else "stage"
        choice = name
        if group == "advanced":
            choice = next((key for key in FEATURE_IMAGES if name.startswith(key)), name)
        if kind == "heatmap":
            choice = "advanced_distance" if group == "advanced" else name
        elif kind == "overlap":
            choice = "advanced_overlap"
        elif kind == "scatter":
            choice = "advanced_scatter"
        elif kind == "histogram":
            choice = "advanced_histogram"
        if not any(requested(name, group, choice)):
            continue
        if kind == "image":
            panel = _tint(_display_scale(array, settings)[sl], channel_colors[role])
        elif kind == "mask":
            color = channel_colors[role] if role else (1, 1, 1)
            panel = _tint(array[sl].astype(float), color)
        elif kind == "labels":
            labels = array[sl]
            panel = plt.get_cmap("tab20")((labels % 20) / 19)[..., :3]
            panel[labels == 0] = 0
            panel = _with_ids(boundary(panel, labels, role or "reference"), labels, settings)
        elif kind == "cell_regions":
            panel = boundary(composite, array[sl], "cell")
            title = "Per-cell measurement regions (nucleus + expansion)"
        elif kind == "radial_regions":
            labels = array[sl]
            limit = max(int(array.max()), 1)
            colors = plt.get_cmap(settings["heatmap_cmap"])(labels / limit)[..., :3]
            panel = composite.copy()
            mask = labels > 0
            panel[mask] = panel[mask] * (1 - settings["opacity"]) + colors[mask] * settings["opacity"]
            panel = boundary(panel, labels, "profile")
            title = "Radial bins: object interior + external nearest-reference regions"
        elif kind == "skeleton":
            panel = tinted[role].copy()
            for mask, name_style in zip(array, ("skeleton", "endpoint", "junction")):
                style = settings["boundaries"][name_style]
                width = style["width_px"]
                panel = _overlay(panel, mask[sl], to_rgb(style["color"]), width, settings["opacity"])
            title = channel_names[role] + ": skeleton / endpoints / junction pixels"
        elif kind in {"heatmap", "spatial_heatmap"}:
            values = np.asarray(array[sl], dtype=float)
            valid = np.isfinite(values)
            limit = float(np.max(values[valid])) if valid.any() else 1.0
            panel = plt.get_cmap(settings["heatmap_cmap"])(np.nan_to_num(values / max(limit, 1e-12)))[..., :3]
            panel[~valid] = 0
            title += f" [0 - {limit:g}]"
            if kind == "heatmap":
                choice = "advanced_distance" if group == "advanced" else name
            else:
                title = channel_names[role] + f": nearest-neighbour distance [0 - {limit:g} um]"
        elif kind == "overlap":
            panel = boundary(composite, array[sl], "overlap")
            color = to_rgb(settings["boundaries"]["overlap"]["color"])
            mask = array[sl]
            panel[mask] = panel[mask] * (1 - settings["opacity"]) + np.asarray(color) * settings["opacity"]
            choice = "advanced_overlap"
        else:
            # A density histogram shows every valid pixel without random subsampling.
            figure, axis = plt.subplots(figsize=(5, 4), dpi=settings["dpi"])
            figure.set_facecolor(settings["background"])
            axis.set_facecolor(settings["background"])
            if kind == "profile":
                for channel_role, alias in channel_names.items():
                    values = array.loc[array["channel"] == alias]
                    means = values.groupby("bin_outer_um")["mean_intensity"].mean()
                    if not means.empty:
                        axis.plot(means.index, means.values, marker="o", label=alias, color=channel_colors[channel_role],
                                  linewidth=settings["boundaries"]["profile"]["width_px"])
                if axis.lines:
                    axis.legend(fontsize=settings["font_size"])
                axis.set_xlabel("External bin edge (um); 0 = object interior")
                axis.set_ylabel("Mean background-corrected intensity")
                title = "Radial profiles: equal weight per reference object"
            elif kind == "scatter":
                x, y = array
                if x.size:
                    _, _, _, density = axis.hist2d(x, y, bins=settings["histogram_bins"], cmap=settings["heatmap_cmap"],
                        cmin=1, norm=LogNorm() if settings["density_log_scale"] else None)
                    figure.colorbar(density, ax=axis, label="Pixel count")
                axis.set_xlabel(channel_names[role[0]] + " (background corrected)")
                axis.set_ylabel(channel_names[role[1]] + " (background corrected)")
                choice = "advanced_scatter"
            else:
                axis.hist(array, bins=settings["histogram_bins"], color=channel_colors[role])
                axis.set_xlabel("Nearest-neighbour centroid distance (um)" if kind == "spatial_histogram" else "Minimum mask distance (um)")
                axis.set_ylabel("Object count")
                if kind != "spatial_histogram":
                    choice = "advanced_histogram"
            axis.set_title(title, fontsize=settings["font_size"])
            axis.tick_params(labelsize=settings["font_size"])
            figure.tight_layout()
            figure.canvas.draw()
            panel = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy() / 255.0
            plt.close(figure)
        add(name, title, panel, group, choice)

    if path is not None:
        chosen = panels
        if not chosen:
            # An optional disabled analysis should not fail an otherwise valid run.
            chosen = [("05_composite", "Composite", composite)]
        columns = min(4, len(chosen))
        rows = int(math.ceil(len(chosen) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4.5 * rows), squeeze=False)
        figure.set_facecolor(settings["background"])
        for axis, (_, title, panel) in zip(axes.ravel(), chosen):
            axis.imshow(panel)
            axis.set_title(title, fontsize=settings["font_size"])
            axis.axis("off")
        for axis in axes.ravel()[len(chosen):]:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(path, dpi=settings["dpi"], bbox_inches="tight", facecolor=settings["background"])
        plt.close(figure)
        saved["qc"] = str(path)
    if processing_dir is not None:
        saved["processing_images_dir"] = str(processing_dir)
    return saved
