"""Notebook widgets for previewing and tuning cell fluorescence analysis."""

from __future__ import annotations

import copy
import html
import math
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from skimage import segmentation

from cell_analyzer.image_io import (
    MICROSCOPY_FILE_PATTERN,
    SUPPORTED_IMAGE_SUFFIXES,
    inspect_image,
    read_image_channels,
)

from .analysis import _channel_colors, _overlay, analyze_arrays
from .batch import _resolve_mask, run_batch_analysis
from .config import (
    DEFAULT_CONFIG,
    create_default_config,
    load_config,
    normalize_config,
    save_config,
)


ROLE_ORDER = ("abeta", "iba1", "cd68", "dapi")
ROLE_LABELS = {"abeta": "Aβ", "iba1": "Iba1", "cd68": "CD68", "dapi": "DAPI"}


def _display_scale(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=float)
    positive = finite[finite > 0]
    basis = positive if positive.size else finite
    low, high = np.percentile(basis, [1, 99.7])
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low), 0, 1)


def _optional_float(value: str) -> float | None:
    text = str(value).strip()
    return None if not text else float(text)


class BrainSectionBatchTuningPanel:
    """Per-file preview, parameter memory, and batch runner for notebooks."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        image_paths: Iterable[str | Path] | None = None,
        output_root: str | Path | None = None,
        max_preview_dimension: int = 1400,
    ) -> None:
        self.config = copy.deepcopy(config or DEFAULT_CONFIG)
        self.image_paths: list[Path] = []
        self.file_configs: dict[str, dict[str, Any]] = {}
        self.current_info = None
        self.preview_products = None
        self.batch_result: dict[str, Any] | None = None
        self.max_preview_dimension = int(max_preview_dimension)
        self._switching = False
        self._channel_mapping_initialized = False
        initial_output = output_root or Path.cwd() / "cell_analysis_results"
        self._build(initial_output)
        self._apply_config_to_controls(self.config)
        if image_paths:
            self._set_paths(list(image_paths))

    def _build(self, output_root: str | Path) -> None:
        self.path_text = widgets.Textarea(
            description="Image files",
            placeholder="One absolute microscopy image path per line",
            layout=widgets.Layout(width="100%", height="105px"),
            style={"description_width": "100px"},
        )
        self.add_button = widgets.Button(
            description="Add images",
            button_style="info",
            icon="folder-open",
            tooltip="Click repeatedly to add microscopy images from different folders.",
        )
        self.apply_paths_button = widgets.Button(description="Apply file list", icon="check")
        self.apply_all_button = widgets.Button(
            description="Apply current settings to all",
            icon="copy",
            tooltip="Copy every current analysis parameter to every selected file.",
        )
        self.file_selector = widgets.Dropdown(
            description="Preview file",
            options=(),
            layout=widgets.Layout(width="72%"),
            style={"description_width": "100px"},
        )
        self.previous_button = widgets.Button(description="Previous", icon="arrow-left")
        self.next_button = widgets.Button(description="Next", icon="arrow-right")

        self.output_root_widget = widgets.Text(
            description="Output root",
            value=str(Path(output_root).expanduser()),
            layout=widgets.Layout(width="100%"),
            style={"description_width": "100px"},
        )
        self.metadata_csv = widgets.Text(
            description="Metadata CSV",
            value="",
            placeholder="Optional: source_name, mouse_id, genotype, region, section_id, sex",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "100px"},
        )
        self.roi_mode = widgets.Dropdown(
            description="Tissue ROI",
            options=("full_image", "mask_directory"),
            value="full_image",
            style={"description_width": "100px"},
        )
        self.mask_directory = widgets.Text(
            description="Mask folder",
            value="",
            layout=widgets.Layout(width="72%"),
            style={"description_width": "100px"},
        )
        self.mask_suffix = widgets.Text(
            description="Mask suffix",
            value="_mask.png",
            style={"description_width": "100px"},
        )
        self.invert_mask = widgets.Checkbox(description="Invert ROI mask", value=False)

        self.scene = widgets.IntText(description="Scene", value=0)
        self.time_index = widgets.IntText(description="Time", value=0)
        self.z_projection = widgets.Dropdown(
            description="Z projection", options=("max", "mean", "single"), value="max"
        )
        self.z_index = widgets.IntText(description="Z index", value=0)
        self.preview_zoom = widgets.BoundedFloatText(
            description="Preview zoom", value=0.25, min=0.01, max=1.0, step=0.05
        )
        self.final_zoom = widgets.BoundedFloatText(
            description="Final zoom", value=1.0, min=0.01, max=1.0, step=0.05
        )
        for control in (
            self.scene,
            self.time_index,
            self.z_projection,
            self.z_index,
            self.preview_zoom,
            self.final_zoom,
        ):
            control.style = {"description_width": "100px"}

        self.channel_controls: dict[str, dict[str, widgets.Widget]] = {}
        channel_rows: list[widgets.Widget] = []
        header = widgets.HTML(
            "<b style='display:grid;grid-template-columns:85px 210px 105px 130px 115px 105px 105px;gap:8px'>"
            "<span>Marker</span><span>CZI channel</span><span>Gaussian σ</span>"
            "<span>Threshold</span><span>Manual value</span><span>Auto scale</span>"
            "<span>Percentile</span></b>"
        )
        channel_rows.append(header)
        for role in ROLE_ORDER:
            controls: dict[str, widgets.Widget] = {
                "index": widgets.Dropdown(options=(), layout=widgets.Layout(width="210px")),
                "sigma": widgets.BoundedFloatText(value=1.0, min=0, max=20, step=0.25, layout=widgets.Layout(width="105px")),
                "method": widgets.Dropdown(
                    options=("manual", "otsu", "yen", "triangle", "percentile"),
                    value="otsu",
                    layout=widgets.Layout(width="130px"),
                ),
                "value": widgets.FloatText(value=0, layout=widgets.Layout(width="115px")),
                "scale": widgets.BoundedFloatText(value=1.0, min=0.01, max=10, step=0.05, layout=widgets.Layout(width="105px")),
                "percentile": widgets.BoundedFloatText(value=95, min=0, max=100, step=0.5, layout=widgets.Layout(width="105px")),
            }
            self.channel_controls[role] = controls
            channel_rows.append(
                widgets.HBox(
                    (
                        widgets.HTML(f"<b>{ROLE_LABELS[role]}</b>", layout=widgets.Layout(width="85px")),
                        controls["index"],
                        controls["sigma"],
                        controls["method"],
                        controls["value"],
                        controls["scale"],
                        controls["percentile"],
                    ),
                    layout=widgets.Layout(gap="8px"),
                )
            )

        self.marker_filter_controls: dict[str, dict[str, widgets.Widget]] = {}
        marker_filter_rows: list[widgets.Widget] = []
        for role in ("iba1", "cd68"):
            controls = {
                "enabled": widgets.Checkbox(description="Enable filter", value=True),
                "opening": widgets.IntText(description="Opening px", value=0),
                "closing": widgets.IntText(description="Closing px", value=0),
                "fill_holes": widgets.Checkbox(description="Fill all holes", value=False),
                "max_hole_area": widgets.FloatText(description="Max hole µm²", value=0.0),
                "min_area": widgets.FloatText(description="Min area µm²", value=0.0),
                "max_area": widgets.Text(description="Max area µm²", value=""),
                "min_circularity": widgets.BoundedFloatText(
                    description="Min circularity", value=0.0, min=0.0, max=1.0, step=0.05
                ),
                "min_solidity": widgets.BoundedFloatText(
                    description="Min solidity", value=0.0, min=0.0, max=1.0, step=0.05
                ),
                "max_eccentricity": widgets.BoundedFloatText(
                    description="Max eccentricity", value=1.0, min=0.0, max=1.0, step=0.05
                ),
            }
            self.marker_filter_controls[role] = controls
            for control in controls.values():
                control.style = {"description_width": "105px"}
            marker_filter_rows.extend(
                (
                    widgets.HTML(f"<h4>{ROLE_LABELS[role]} connected-object filters</h4>"),
                    widgets.HBox(
                        (
                            controls["enabled"],
                            controls["opening"],
                            controls["closing"],
                            controls["fill_holes"],
                            controls["max_hole_area"],
                        )
                    ),
                    widgets.HBox(
                        (
                            controls["min_area"],
                            controls["max_area"],
                            controls["min_circularity"],
                            controls["min_solidity"],
                            controls["max_eccentricity"],
                        )
                    ),
                )
            )

        self.microglia_enabled = widgets.Checkbox(
            description="Count DAPI+ / Iba1+ microglia", value=False,
            layout=widgets.Layout(width="280px"),
        )
        self.nucleus_opening = widgets.IntText(description="Opening px", value=0)
        self.nucleus_closing = widgets.IntText(description="Closing px", value=1)
        self.nucleus_fill_holes = widgets.Checkbox(description="Fill nuclei", value=True)
        self.nucleus_min_area = widgets.FloatText(
            description="Min nucleus µm²", value=10.0
        )
        self.nucleus_max_area = widgets.Text(
            description="Max nucleus µm²", value="150"
        )
        self.nucleus_min_circularity = widgets.BoundedFloatText(
            description="Min circularity", value=0.20, min=0, max=1, step=0.05
        )
        self.nucleus_min_solidity = widgets.BoundedFloatText(
            description="Min solidity", value=0.70, min=0, max=1, step=0.05
        )
        self.nucleus_max_eccentricity = widgets.BoundedFloatText(
            description="Max eccentricity", value=0.98, min=0, max=1, step=0.01
        )
        self.nucleus_split_touching = widgets.Checkbox(
            description="Split touching nuclei", value=True
        )
        self.nucleus_peak_distance = widgets.IntText(
            description="Peak distance px", value=3
        )
        self.perinuclear_radius = widgets.FloatText(
            description="Iba1 radius µm", value=3.0
        )
        self.min_perinuclear_iba1_fraction = widgets.BoundedFloatText(
            description="Min Iba1 fraction", value=0.15, min=0, max=1, step=0.05
        )
        for control in (
            self.nucleus_opening,
            self.nucleus_closing,
            self.nucleus_min_area,
            self.nucleus_max_area,
            self.nucleus_min_circularity,
            self.nucleus_min_solidity,
            self.nucleus_max_eccentricity,
            self.nucleus_peak_distance,
            self.perinuclear_radius,
            self.min_perinuclear_iba1_fraction,
        ):
            control.style = {"description_width": "115px"}

        self.min_plaque_area = widgets.FloatText(description="Min plaque µm²", value=10.0)
        self.max_plaque_area = widgets.Text(description="Max plaque µm²", value="")
        self.opening_radius = widgets.IntText(description="Opening px", value=0)
        self.closing_radius = widgets.IntText(description="Closing px", value=1)
        self.fill_holes = widgets.Checkbox(description="Fill plaque holes", value=False)
        self.max_hole_area = widgets.FloatText(description="Max hole µm²", value=0.0)
        self.min_circularity = widgets.BoundedFloatText(
            description="Min circularity", value=0.0, min=0.0, max=1.0, step=0.05
        )
        self.min_solidity = widgets.BoundedFloatText(
            description="Min solidity", value=0.0, min=0.0, max=1.0, step=0.05
        )
        self.max_eccentricity = widgets.BoundedFloatText(
            description="Max eccentricity", value=1.0, min=0.0, max=1.0, step=0.05
        )
        self.neuron_exclusion_mode = widgets.Dropdown(
            description="Soma exclusion",
            options=("off", "shape", "shape_and_dark_center"),
            value="shape_and_dark_center",
        )
        self.neuron_detection_threshold_scale = widgets.FloatText(
            description="Soma threshold ×", value=0.75
        )
        self.neuron_min_diameter = widgets.FloatText(
            description="Soma min µm", value=8.0
        )
        self.neuron_max_diameter = widgets.FloatText(
            description="Soma max µm", value=28.0
        )
        self.neuron_min_circularity = widgets.BoundedFloatText(
            description="Soma min circ", value=0.35, min=0.0, max=1.0, step=0.05
        )
        self.neuron_min_solidity = widgets.BoundedFloatText(
            description="Soma min solidity", value=0.60, min=0.0, max=1.0, step=0.05
        )
        self.neuron_min_hole_fraction = widgets.BoundedFloatText(
            description="Min hole fraction", value=0.03, min=0.0, max=1.0, step=0.01
        )
        self.neuron_max_center_shell_ratio = widgets.FloatText(
            description="Max center/shell", value=0.95
        )
        self.require_nearby_microglia = widgets.Checkbox(
            description="Require nearby microglia", value=False,
            layout=widgets.Layout(width="250px"),
        )
        self.nearby_microglia_radius = widgets.FloatText(
            description="Microglia radius µm", value=30.0
        )
        self.min_nearby_microglia_count = widgets.IntText(
            description="Min nearby microglia", value=1
        )
        self.split_touching = widgets.Checkbox(description="Split touching plaques", value=False)
        self.min_peak_distance = widgets.IntText(description="Peak distance px", value=8)
        self.watershed_min_height = widgets.FloatText(description="Peak height px", value=0.0)
        self.watershed_compactness = widgets.FloatText(description="WS compactness", value=0.0)
        self.exclude_boundary = widgets.Checkbox(
            description="Exclude boundary plaques from plaque table", value=True,
            layout=widgets.Layout(width="330px"),
        )
        self.boundary_margin = widgets.FloatText(description="Boundary µm", value=0.0)
        self.ring_edges = widgets.Text(description="Cumulative radii µm", value="0,30")
        self.ring_boundary_width = widgets.BoundedIntText(
            description="Ring line width px", value=1, min=1, max=20
        )
        self.save_masks = widgets.Checkbox(description="Save masks", value=True)
        self.save_qc = widgets.Checkbox(description="Save QC", value=True)
        self.save_processing_images = widgets.Checkbox(
            description="Save all processing images", value=True,
            layout=widgets.Layout(width="240px"),
        )
        for control in (
            self.min_plaque_area,
            self.max_plaque_area,
            self.opening_radius,
            self.closing_radius,
            self.max_hole_area,
            self.min_circularity,
            self.min_solidity,
            self.max_eccentricity,
            self.neuron_exclusion_mode,
            self.neuron_detection_threshold_scale,
            self.neuron_min_diameter,
            self.neuron_max_diameter,
            self.neuron_min_circularity,
            self.neuron_min_solidity,
            self.neuron_min_hole_fraction,
            self.neuron_max_center_shell_ratio,
            self.nearby_microglia_radius,
            self.min_nearby_microglia_count,
            self.min_peak_distance,
            self.watershed_min_height,
            self.watershed_compactness,
            self.boundary_margin,
            self.ring_edges,
            self.ring_boundary_width,
        ):
            control.style = {"description_width": "120px"}

        self.preview_button = widgets.Button(
            description="Update preview", button_style="warning", icon="image"
        )
        self.save_button = widgets.Button(description="Save session", icon="save")
        self.load_button = widgets.Button(description="Load session", icon="folder-open")
        self.run_button = widgets.Button(
            description="Run all files", button_style="success", icon="play"
        )
        self.status = widgets.HTML(value="<b>Add one or more microscopy images.</b>")
        self.metadata_label = widgets.HTML()
        self.preview_image = widgets.Image(format="png", layout=widgets.Layout(width="100%"))
        self.preview_metrics = widgets.HTML()
        self.result_table = widgets.HTML()

        self.add_button.on_click(self._choose_files)
        self.apply_paths_button.on_click(self._apply_text_paths)
        self.apply_all_button.on_click(self._apply_current_to_all)
        self.previous_button.on_click(lambda button: self._move(-1))
        self.next_button.on_click(lambda button: self._move(1))
        self.file_selector.observe(self._change_file, names="value")
        self.preview_button.on_click(self._update_preview)
        self.save_button.on_click(self._save_parameters)
        self.load_button.on_click(self._choose_session)
        self.run_button.on_click(self._start_batch)

        files_box = widgets.VBox(
            (
                widgets.HTML(
                    "<h3>Cell Aβ / Iba1 / CD68 batch analysis</h3>"
                    "<p>Add files from any number of folders. Each file remembers its own settings. "
                    "Tune a representative image, then click <b>Apply current settings to all</b> "
                    "to standardize the batch before checking every file.</p>"
                ),
                self.path_text,
                widgets.HBox((self.add_button, self.apply_paths_button, self.apply_all_button)),
                widgets.HBox((self.file_selector, self.previous_button, self.next_button)),
                self.output_root_widget,
                self.metadata_csv,
                widgets.HBox((self.roi_mode, self.mask_directory, self.mask_suffix, self.invert_mask)),
                self.metadata_label,
            )
        )
        plane_box = widgets.VBox(
            (
                widgets.HTML("<b>Image plane and resolution</b>"),
                widgets.HBox((self.scene, self.time_index, self.z_projection, self.z_index)),
                widgets.HBox((self.preview_zoom, self.final_zoom)),
            )
        )
        channel_box = widgets.VBox(
            tuple(channel_rows)
            + (
                widgets.HTML(
                    "<small>Manual value is used only for manual thresholding. "
                    "Keep acquisition settings and thresholds fixed within one staining batch.</small>"
                ),
            )
        )
        marker_filter_box = widgets.VBox(
            (
                widgets.HTML(
                    "<b>Iba1 / CD68 positive-object filters</b>"
                    "<p><small>Applied independently after each intensity threshold. "
                    "Area is calibrated in µm²; opening/closing are pixel parameters. "
                    "The accepted objects determine marker area and overlap measurements, "
                    "but do not define Aβ plaques. At 20×, an Iba1 connected component "
                    "may contain touching processes and is not automatically one cell.</small></p>"
                ),
                *marker_filter_rows,
            )
        )
        microglia_box = widgets.VBox(
            (
                widgets.HTML(
                    "<b>Microglia count: DAPI nucleus + local Iba1 confirmation</b>"
                    "<p><small>Each accepted nucleus is counted once. Its centroid is "
                    "assigned to the nearest Aβ plaque and the configured distance ring. "
                    "Yellow DAPI boundaries in preview are counted microglia.</small></p>"
                ),
                widgets.HBox(
                    (
                        self.microglia_enabled,
                        self.nucleus_opening,
                        self.nucleus_closing,
                        self.nucleus_fill_holes,
                    )
                ),
                widgets.HBox(
                    (
                        self.nucleus_min_area,
                        self.nucleus_max_area,
                        self.nucleus_min_circularity,
                        self.nucleus_min_solidity,
                        self.nucleus_max_eccentricity,
                    )
                ),
                widgets.HBox(
                    (
                        self.nucleus_split_touching,
                        self.nucleus_peak_distance,
                        self.perinuclear_radius,
                        self.min_perinuclear_iba1_fraction,
                    )
                ),
            )
        )
        plaque_box = widgets.VBox(
            (
                widgets.HTML("<b>Plaque segmentation and spatial rings</b>"),
                widgets.HBox((self.min_plaque_area, self.max_plaque_area, self.opening_radius, self.closing_radius)),
                widgets.HBox((self.fill_holes, self.max_hole_area, self.exclude_boundary)),
                widgets.HBox((self.min_circularity, self.min_solidity, self.max_eccentricity)),
                widgets.HTML(
                    "<small><b>Neuron-like Aβ exclusion uses Aβ shape only.</b> "
                    "CD68 is deliberately not used to define plaques because it is a biological outcome.</small>"
                ),
                widgets.HBox((self.neuron_exclusion_mode, self.neuron_detection_threshold_scale, self.neuron_min_diameter, self.neuron_max_diameter)),
                widgets.HBox((self.neuron_min_circularity, self.neuron_min_solidity, self.neuron_min_hole_fraction, self.neuron_max_center_shell_ratio)),
                widgets.HBox((self.require_nearby_microglia, self.nearby_microglia_radius, self.min_nearby_microglia_count)),
                widgets.HBox((self.split_touching, self.min_peak_distance, self.watershed_min_height, self.watershed_compactness)),
                widgets.HBox(
                    (self.boundary_margin, self.ring_edges, self.ring_boundary_width)
                ),
                widgets.HBox((self.save_masks, self.save_qc, self.save_processing_images)),
            )
        )
        action_box = widgets.VBox(
            (
                widgets.HBox(
                    (
                        self.preview_button,
                        self.save_button,
                        self.load_button,
                        self.run_button,
                    )
                ),
                self.status,
                self.preview_metrics,
                self.result_table,
            )
        )
        accordion = widgets.Accordion(
            children=(plane_box, channel_box, marker_filter_box, microglia_box, plaque_box)
        )
        accordion.set_title(0, "Image plane")
        accordion.set_title(1, "Channel thresholds")
        accordion.set_title(2, "Iba1 / CD68 object filters")
        accordion.set_title(3, "Microglia count")
        accordion.set_title(4, "Plaques and distance rings")
        for index in range(5):
            accordion.selected_index = 1 if index == 1 else accordion.selected_index
        self.widget = widgets.VBox((files_box, accordion, action_box, self.preview_image))

    def _apply_config_to_controls(self, config: dict[str, Any]) -> None:
        input_config = config.get("input", {})
        self.scene.value = int(input_config.get("scene", 0))
        self.time_index.value = int(input_config.get("time_index", 0))
        self.z_projection.value = str(input_config.get("z_projection", "max"))
        self.z_index.value = int(input_config.get("z_index", 0))
        self.final_zoom.value = float(input_config.get("zoom", 1.0))
        for role in ROLE_ORDER:
            item = config.get("channels", {}).get(role, {})
            controls = self.channel_controls[role]
            requested_index = item.get("index")
            available_indices = {value for _, value in controls["index"].options}
            if requested_index is not None and int(requested_index) in available_indices:
                controls["index"].value = int(requested_index)
            controls["sigma"].value = float(item.get("gaussian_sigma_px", 1.0))
            threshold = item.get("threshold", {})
            controls["method"].value = str(threshold.get("method", "otsu"))
            controls["value"].value = float(threshold.get("value", 0.0))
            controls["scale"].value = float(threshold.get("scale", 1.0))
            controls["percentile"].value = float(threshold.get("percentile", 95.0))
            if role in self.marker_filter_controls:
                object_filter = item.get("object_filter", {})
                marker_controls = self.marker_filter_controls[role]
                marker_controls["enabled"].value = bool(
                    object_filter.get("enabled", True)
                )
                marker_controls["opening"].value = int(
                    object_filter.get("opening_radius_px", 0)
                )
                marker_controls["closing"].value = int(
                    object_filter.get("closing_radius_px", 0)
                )
                marker_controls["fill_holes"].value = bool(
                    object_filter.get("fill_holes", False)
                )
                marker_controls["max_hole_area"].value = float(
                    object_filter.get("max_hole_area_um2", 0.0)
                )
                marker_controls["min_area"].value = float(
                    object_filter.get("min_area_um2", 0.0)
                )
                marker_controls["max_area"].value = (
                    ""
                    if object_filter.get("max_area_um2") is None
                    else str(object_filter["max_area_um2"])
                )
                marker_controls["min_circularity"].value = float(
                    object_filter.get("min_circularity", 0.0)
                )
                marker_controls["min_solidity"].value = float(
                    object_filter.get("min_solidity", 0.0)
                )
                marker_controls["max_eccentricity"].value = float(
                    object_filter.get("max_eccentricity", 1.0)
                )
        microglia = config.get("microglia_count", {})
        self.microglia_enabled.value = bool(
            microglia.get("enabled", False)
            and config.get("channels", {}).get("dapi", {}).get("enabled", False)
        )
        self.nucleus_opening.value = int(microglia.get("opening_radius_px", 0))
        self.nucleus_closing.value = int(microglia.get("closing_radius_px", 1))
        self.nucleus_fill_holes.value = bool(microglia.get("fill_holes", True))
        self.nucleus_min_area.value = float(
            microglia.get("min_nucleus_area_um2", 10.0)
        )
        self.nucleus_max_area.value = (
            ""
            if microglia.get("max_nucleus_area_um2") is None
            else str(microglia["max_nucleus_area_um2"])
        )
        self.nucleus_min_circularity.value = float(
            microglia.get("min_circularity", 0.20)
        )
        self.nucleus_min_solidity.value = float(
            microglia.get("min_solidity", 0.70)
        )
        self.nucleus_max_eccentricity.value = float(
            microglia.get("max_eccentricity", 0.98)
        )
        self.nucleus_split_touching.value = bool(
            microglia.get("split_touching", True)
        )
        self.nucleus_peak_distance.value = int(
            microglia.get("min_peak_distance_px", 3)
        )
        self.perinuclear_radius.value = float(
            microglia.get("perinuclear_radius_um", 3.0)
        )
        self.min_perinuclear_iba1_fraction.value = float(
            microglia.get("min_iba1_positive_fraction", 0.15)
        )
        roi = config.get("tissue_roi", {})
        mode = str(roi.get("mode", "full_image"))
        self.roi_mode.value = mode if mode in {"full_image", "mask_directory"} else "full_image"
        self.mask_directory.value = str(roi.get("mask_directory") or "")
        self.mask_suffix.value = str(roi.get("mask_suffix", "_mask.png"))
        self.invert_mask.value = bool(roi.get("invert_mask", False))
        plaque = config.get("plaque", {})
        self.min_plaque_area.value = float(plaque.get("min_area_um2", 10.0))
        self.max_plaque_area.value = "" if plaque.get("max_area_um2") is None else str(plaque["max_area_um2"])
        self.opening_radius.value = int(plaque.get("opening_radius_px", 0))
        self.closing_radius.value = int(plaque.get("closing_radius_px", 1))
        self.fill_holes.value = bool(plaque.get("fill_holes", False))
        self.max_hole_area.value = float(plaque.get("max_hole_area_um2", 0.0))
        self.min_circularity.value = float(plaque.get("min_circularity", 0.0))
        self.min_solidity.value = float(plaque.get("min_solidity", 0.0))
        self.max_eccentricity.value = float(plaque.get("max_eccentricity", 1.0))
        self.neuron_exclusion_mode.value = str(
            plaque.get("neuron_exclusion_mode", "shape_and_dark_center")
        )
        self.neuron_min_diameter.value = float(
            plaque.get("neuron_min_diameter_um", 8.0)
        )
        self.neuron_detection_threshold_scale.value = float(
            plaque.get("neuron_detection_threshold_scale", 0.75)
        )
        self.neuron_max_diameter.value = float(
            plaque.get("neuron_max_diameter_um", 28.0)
        )
        self.neuron_min_circularity.value = float(
            plaque.get("neuron_min_circularity", 0.35)
        )
        self.neuron_min_solidity.value = float(
            plaque.get("neuron_min_solidity", 0.60)
        )
        self.neuron_min_hole_fraction.value = float(
            plaque.get("neuron_min_hole_fraction", 0.03)
        )
        self.neuron_max_center_shell_ratio.value = float(
            plaque.get("neuron_max_center_shell_ratio", 0.95)
        )
        self.require_nearby_microglia.value = bool(
            plaque.get("require_nearby_microglia", False)
        )
        self.nearby_microglia_radius.value = float(
            plaque.get("nearby_microglia_radius_um", 30.0)
        )
        self.min_nearby_microglia_count.value = int(
            plaque.get("min_nearby_microglia_count", 1)
        )
        self.split_touching.value = bool(plaque.get("split_touching", False))
        self.min_peak_distance.value = int(plaque.get("min_peak_distance_px", 8))
        self.watershed_min_height.value = float(
            plaque.get("watershed_min_peak_height_px", 0.0)
        )
        self.watershed_compactness.value = float(
            plaque.get("watershed_compactness", 0.0)
        )
        self.exclude_boundary.value = bool(plaque.get("exclude_boundary_plaques_from_table", True))
        self.boundary_margin.value = float(plaque.get("boundary_margin_um", 0.0))
        self.ring_edges.value = ",".join(str(value) for value in config.get("spatial", {}).get("ring_edges_um", [0, 30]))
        self.metadata_csv.value = str(config.get("batch", {}).get("metadata_csv") or "")
        output = config.get("output", {})
        self.save_masks.value = bool(output.get("save_masks", True))
        self.save_qc.value = bool(output.get("save_qc", True))
        self.save_processing_images.value = bool(
            output.get("save_processing_images", True)
        )
        self.ring_boundary_width.value = int(
            output.get("ring_boundary_width_px", 1)
        )

    @staticmethod
    def _validated_paths(values: Iterable[str | Path]) -> list[Path]:
        paths: list[Path] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip().strip('"').strip("'")
            if not text:
                continue
            path = Path(text).expanduser().resolve()
            key = str(path).casefold()
            if key in seen:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            if path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
                supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
                raise ValueError(f"Unsupported image format: {path}; expected {supported}")
            paths.append(path)
            seen.add(key)
        if not paths:
            raise ValueError("Select at least one supported microscopy image.")
        return paths

    def _set_paths(self, values: Iterable[str | Path]) -> None:
        try:
            paths = self._validated_paths(values)
            self._snapshot_current()
            current = str(self.file_selector.value or "")
            options = [(f"{index}. {path.name}", str(path)) for index, path in enumerate(paths, 1)]
            selected = current if current in {value for _, value in options} else options[0][1]
            self._switching = True
            try:
                self.image_paths = paths
                valid_keys = {str(path).casefold() for path in paths}
                self.file_configs = {
                    key: value for key, value in self.file_configs.items() if key in valid_keys
                }
                self.path_text.value = "\n".join(str(path) for path in paths)
                self.file_selector.options = options
                self.file_selector.value = selected
            finally:
                self._switching = False
            self._load_current_metadata()
        except Exception as error:
            self.status.value = (
                f"<span style='color:#b00020'><b>File selection error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def _choose_files(self, button: widgets.Button) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update()
            try:
                selected = filedialog.askopenfilenames(
                    parent=root,
                    title="Select multichannel cell-analysis images",
                    filetypes=(
                        ("All supported microscopy images", MICROSCOPY_FILE_PATTERN),
                        ("Zeiss CZI", "*.czi"),
                        ("Leica", "*.lif *.lei *.scn *.lof *.xlef"),
                        ("Olympus", "*.oir *.vsi *.oib *.oif"),
                        ("OME/TIFF", "*.ome.tif *.ome.tiff *.tif *.tiff"),
                        ("Nikon ND2", "*.nd2"),
                        ("All files", "*.*"),
                    ),
                )
            finally:
                root.destroy()
            if selected:
                self._set_paths([*self.image_paths, *selected])
        except Exception as error:
            self.status.value = (
                f"<span style='color:#b00020'><b>File dialog error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def _apply_text_paths(self, button: widgets.Button) -> None:
        self._set_paths(self.path_text.value.splitlines())

    def _change_file(self, change: dict[str, Any]) -> None:
        if self._switching or not change.get("new"):
            return
        self._snapshot_current(path_override=change.get("old"))
        self._load_current_metadata()

    def _move(self, step: int) -> None:
        values = [value for _, value in self.file_selector.options]
        if not values:
            return
        current = values.index(self.file_selector.value)
        self.file_selector.value = values[(current + step) % len(values)]

    def _snapshot_current(self, path_override: str | Path | None = None) -> None:
        selected = path_override or self.file_selector.value
        if not selected:
            return
        try:
            config = self._current_config(preview=False, path_override=selected)
        except Exception:
            return
        self.file_configs[str(Path(selected).resolve()).casefold()] = copy.deepcopy(config)

    def _apply_current_to_all(self, button: widgets.Button) -> None:
        try:
            if not self.image_paths:
                raise ValueError("Select at least one CZI file first.")
            current = self._current_config(preview=False)
            self.config = copy.deepcopy(current)
            copied: dict[str, dict[str, Any]] = {}
            output_root = Path(self.output_root_widget.value).expanduser().resolve()
            for path in self.image_paths:
                item = copy.deepcopy(current)
                item["input"]["image_path"] = str(path)
                item["input"]["output_dir"] = str(output_root / path.stem)
                item["tissue_roi"]["mask_path"] = None
                copied[str(path).casefold()] = item
            self.file_configs = copied
            self.status.value = (
                f"<b>Applied current settings to all {len(self.image_paths)} files.</b> "
                "You may now switch files and fine-tune an individual file if necessary. "
                "Every per-file setting will be recorded in batch_parameters.yaml."
            )
        except Exception as error:
            self.status.value = (
                f"<span style='color:#b00020'><b>Apply-all error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def _load_current_metadata(self) -> None:
        selected = self.file_selector.value
        if not selected:
            return
        try:
            info = inspect_image(selected)
            if len(info.channels) < 3:
                raise ValueError(
                    f"At least three channels are required; {Path(selected).name} has {len(info.channels)}."
                )
            detected = create_default_config(selected)
            options = tuple(
                (f"C{channel.index}: {channel.name}", channel.index)
                for channel in info.channels
            )
            available = {value for _, value in options}
            for role in ROLE_ORDER:
                control = self.channel_controls[role]["index"]
                control.options = options
            saved = self.file_configs.get(str(Path(selected).resolve()).casefold())
            if saved is not None:
                self._apply_config_to_controls(saved)
            else:
                for role in ROLE_ORDER:
                    control = self.channel_controls[role]["index"]
                    if (
                        role == "dapi"
                        and not detected["channels"]["dapi"].get("enabled", False)
                    ):
                        control.value = (
                            control.value
                            if control.value in available
                            else options[-1][1]
                        )
                        continue
                    if not self._channel_mapping_initialized:
                        selected_index = int(detected["channels"][role]["index"])
                    else:
                        current = control.value
                        selected_index = int(current) if current in available else int(detected["channels"][role]["index"])
                    control.value = selected_index
                self.microglia_enabled.value = bool(
                    detected["microglia_count"].get("enabled", False)
                )
            self._channel_mapping_initialized = True
            self.current_info = info
            dimensions = info.dimensions
            channel_text = ", ".join(f"C{item.index}: {item.name}" for item in info.channels)
            self.metadata_label.value = (
                f"<b>{html.escape(Path(selected).name)}</b> | "
                f"{dimensions.get('X', (0, 0))[1]} × {dimensions.get('Y', (0, 0))[1]} px | "
                f"Z={dimensions.get('Z', (0, 1))[1] - dimensions.get('Z', (0, 1))[0]} | "
                f"{html.escape(channel_text)} | pixel size: "
                f"{info.pixel_size_um_x} × {info.pixel_size_um_y} µm"
            )
            position = next(index for index, path in enumerate(self.image_paths, 1) if str(path) == str(selected))
            self.status.value = (
                f"<b>Previewing {position}/{len(self.image_paths)}.</b> "
                "This file's saved settings have been restored. Click <b>Update preview</b>."
            )
        except Exception as error:
            self.current_info = None
            self.status.value = (
                f"<span style='color:#b00020'><b>Metadata error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def _ring_edge_values(self) -> list[float]:
        values = [float(value.strip()) for value in self.ring_edges.value.split(",") if value.strip()]
        if len(values) < 2 or values[0] != 0 or any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(
                "Cumulative radii must start at 0 and increase; "
                "0,15,30 creates plaque+0–15 and plaque+0–30 µm regions."
            )
        return values

    def _current_config(
        self,
        *,
        preview: bool,
        path_override: str | Path | None = None,
    ) -> dict[str, Any]:
        selected = path_override or self.file_selector.value
        if not selected:
            raise ValueError("Select a CZI file first.")
        output_root = Path(self.output_root_widget.value).expanduser().resolve()
        config = create_default_config(selected, output_dir=output_root / Path(selected).stem)
        pixel_parameter_scale = (
            float(self.preview_zoom.value) / float(self.final_zoom.value)
            if preview
            else 1.0
        )
        config["input"].update(
            {
                "scene": int(self.scene.value),
                "time_index": int(self.time_index.value),
                "z_projection": str(self.z_projection.value),
                "z_index": int(self.z_index.value),
                "zoom": float(self.preview_zoom.value if preview else self.final_zoom.value),
            }
        )
        indices: list[int] = []
        for role in ROLE_ORDER:
            controls = self.channel_controls[role]
            if role == "dapi" and not self.microglia_enabled.value:
                config["channels"]["dapi"]["enabled"] = False
                continue
            if controls["index"].value is None:
                raise ValueError(f"Select the {ROLE_LABELS[role]} channel.")
            index = int(controls["index"].value)
            indices.append(index)
            config["channels"][role].update(
                {
                    "index": index,
                    "alias": ROLE_LABELS[role],
                    "gaussian_sigma_px": float(controls["sigma"].value)
                    * pixel_parameter_scale,
                    "threshold": {
                        "method": str(controls["method"].value),
                        "value": float(controls["value"].value),
                        "scale": float(controls["scale"].value),
                        "percentile": float(controls["percentile"].value),
                    },
                }
            )
            if role == "dapi":
                config["channels"]["dapi"]["enabled"] = True
            if role in self.marker_filter_controls:
                marker_controls = self.marker_filter_controls[role]
                config["channels"][role]["object_filter"] = {
                    "enabled": bool(marker_controls["enabled"].value),
                    "opening_radius_px": max(
                        0,
                        int(
                            round(
                                float(marker_controls["opening"].value)
                                * pixel_parameter_scale
                            )
                        ),
                    ),
                    "closing_radius_px": max(
                        0,
                        int(
                            round(
                                float(marker_controls["closing"].value)
                                * pixel_parameter_scale
                            )
                        ),
                    ),
                    "fill_holes": bool(marker_controls["fill_holes"].value),
                    "max_hole_area_um2": max(
                        0.0, float(marker_controls["max_hole_area"].value)
                    ),
                    "min_area_um2": max(
                        0.0, float(marker_controls["min_area"].value)
                    ),
                    "max_area_um2": _optional_float(
                        marker_controls["max_area"].value
                    ),
                    "min_circularity": float(
                        marker_controls["min_circularity"].value
                    ),
                    "min_solidity": float(marker_controls["min_solidity"].value),
                    "max_eccentricity": float(
                        marker_controls["max_eccentricity"].value
                    ),
                }
        if len(set(indices)) != len(indices):
            raise ValueError("Enabled marker roles must use different channels.")
        config["tissue_roi"].update(
            {
                "mode": str(self.roi_mode.value),
                "mask_directory": self.mask_directory.value.strip() or None,
                "mask_suffix": self.mask_suffix.value.strip() or "_mask.png",
                "invert_mask": bool(self.invert_mask.value),
            }
        )
        if config["tissue_roi"]["mode"] == "mask_directory" and not config["tissue_roi"]["mask_directory"]:
            raise ValueError("Choose a mask folder or change Tissue ROI to full_image.")
        config["plaque"].update(
            {
                "opening_radius_px": max(
                    0, int(round(float(self.opening_radius.value) * pixel_parameter_scale))
                ),
                "closing_radius_px": max(
                    0, int(round(float(self.closing_radius.value) * pixel_parameter_scale))
                ),
                "fill_holes": bool(self.fill_holes.value),
                "max_hole_area_um2": max(0.0, float(self.max_hole_area.value)),
                "min_area_um2": max(0.0, float(self.min_plaque_area.value)),
                "max_area_um2": _optional_float(self.max_plaque_area.value),
                "min_circularity": float(self.min_circularity.value),
                "min_solidity": float(self.min_solidity.value),
                "max_eccentricity": float(self.max_eccentricity.value),
                "neuron_exclusion_mode": str(self.neuron_exclusion_mode.value),
                "neuron_detection_threshold_scale": max(
                    0.01, float(self.neuron_detection_threshold_scale.value)
                ),
                "neuron_min_diameter_um": max(
                    0.0, float(self.neuron_min_diameter.value)
                ),
                "neuron_max_diameter_um": max(
                    0.0, float(self.neuron_max_diameter.value)
                ),
                "neuron_min_circularity": float(
                    self.neuron_min_circularity.value
                ),
                "neuron_min_solidity": float(self.neuron_min_solidity.value),
                "neuron_min_hole_fraction": float(
                    self.neuron_min_hole_fraction.value
                ),
                "neuron_max_center_shell_ratio": max(
                    0.0, float(self.neuron_max_center_shell_ratio.value)
                ),
                "require_nearby_microglia": bool(
                    self.require_nearby_microglia.value
                ),
                "nearby_microglia_radius_um": max(
                    0.0, float(self.nearby_microglia_radius.value)
                ),
                "min_nearby_microglia_count": max(
                    1, int(self.min_nearby_microglia_count.value)
                ),
                "split_touching": bool(self.split_touching.value),
                "min_peak_distance_px": max(
                    1,
                    int(round(float(self.min_peak_distance.value) * pixel_parameter_scale)),
                ),
                "watershed_min_peak_height_px": max(
                    0.0,
                    float(self.watershed_min_height.value) * pixel_parameter_scale,
                ),
                "watershed_compactness": max(
                    0.0, float(self.watershed_compactness.value)
                ),
                "exclude_boundary_plaques_from_table": bool(self.exclude_boundary.value),
                "boundary_margin_um": max(0.0, float(self.boundary_margin.value)),
            }
        )
        config["microglia_count"].update(
            {
                "enabled": bool(self.microglia_enabled.value),
                "opening_radius_px": max(
                    0,
                    int(
                        round(
                            float(self.nucleus_opening.value)
                            * pixel_parameter_scale
                        )
                    ),
                ),
                "closing_radius_px": max(
                    0,
                    int(
                        round(
                            float(self.nucleus_closing.value)
                            * pixel_parameter_scale
                        )
                    ),
                ),
                "fill_holes": bool(self.nucleus_fill_holes.value),
                "min_nucleus_area_um2": max(
                    0.0, float(self.nucleus_min_area.value)
                ),
                "max_nucleus_area_um2": _optional_float(
                    self.nucleus_max_area.value
                ),
                "min_circularity": float(self.nucleus_min_circularity.value),
                "min_solidity": float(self.nucleus_min_solidity.value),
                "max_eccentricity": float(
                    self.nucleus_max_eccentricity.value
                ),
                "split_touching": bool(self.nucleus_split_touching.value),
                "min_peak_distance_px": max(
                    1,
                    int(
                        round(
                            float(self.nucleus_peak_distance.value)
                            * pixel_parameter_scale
                        )
                    ),
                ),
                "perinuclear_radius_um": max(
                    0.0, float(self.perinuclear_radius.value)
                ),
                "min_iba1_positive_fraction": float(
                    self.min_perinuclear_iba1_fraction.value
                ),
            }
        )
        config["spatial"]["ring_edges_um"] = self._ring_edge_values()
        config["batch"]["metadata_csv"] = self.metadata_csv.value.strip() or None
        config["metadata"] = copy.deepcopy(self.config.get("metadata", config["metadata"]))
        config["output"]["save_masks"] = bool(self.save_masks.value)
        config["output"]["save_qc"] = bool(self.save_qc.value)
        config["output"]["save_processing_images"] = bool(
            self.save_processing_images.value
        )
        config["output"]["ring_boundary_width_px"] = int(
            self.ring_boundary_width.value
        )
        return config

    def _set_busy(self, busy: bool) -> None:
        for control in (
            self.add_button,
            self.apply_paths_button,
            self.apply_all_button,
            self.file_selector,
            self.previous_button,
            self.next_button,
            self.preview_button,
            self.save_button,
            self.load_button,
            self.run_button,
        ):
            control.disabled = busy

    def _update_preview(self, button: widgets.Button | None = None) -> None:
        try:
            self._set_busy(True)
            self.status.value = "<b>Reading channels and calculating preview...</b>"
            self._snapshot_current()
            config = self._current_config(preview=True)
            source = Path(config["input"]["image_path"])
            if config["tissue_roi"]["mode"] == "mask_directory":
                _resolve_mask(config, source)
            input_config = config["input"]
            info = inspect_image(source)
            config = normalize_config(config, info)
            roles = ["abeta", "iba1", "cd68"]
            if bool(config["channels"]["dapi"].get("enabled", False)):
                roles.append("dapi")
            role_indices = {
                role: int(config["channels"][role]["index"]) for role in roles
            }
            channels = read_image_channels(
                source,
                info,
                scene=input_config["scene"],
                time_index=input_config["time_index"],
                z_projection=input_config["z_projection"],
                z_index=input_config["z_index"],
                zoom=input_config["zoom"],
                channel_indices=sorted(set(role_indices.values())),
            )
            images = {role: channels[index] for role, index in role_indices.items()}
            if info.pixel_size_um_x is None or info.pixel_size_um_y is None:
                raise ValueError("CZI X/Y pixel calibration is required.")
            products = analyze_arrays(
                images,
                config,
                pixel_size_um_x=float(info.pixel_size_um_x) / input_config["zoom"],
                pixel_size_um_y=float(info.pixel_size_um_y) / input_config["zoom"],
                source_file=str(source),
            )
            self.preview_products = products
            self._render_preview(images, products, config, info)
            self.status.value = (
                f"<b>Preview ready:</b> {Path(source).name}. "
                "If this is the representative setting, apply it to all files and then inspect each QC preview."
            )
        except Exception as error:
            self.status.value = (
                f"<span style='color:#b00020'><b>Preview error:</b> "
                f"{html.escape(str(error))}</span>"
            )
        finally:
            self._set_busy(False)

    def _render_preview(
        self,
        images: dict[str, np.ndarray],
        products: Any,
        config: dict[str, Any],
        info: Any,
    ) -> None:
        shape = products.tissue_mask.shape
        stride = max(1, int(math.ceil(max(shape) / self.max_preview_dimension)))
        sl = (slice(None, None, stride), slice(None, None, stride))
        scaled = {role: _display_scale(image)[sl] for role, image in images.items()}
        masks = {
            role: mask[sl] for role, mask in products.positive_masks.items()
        }
        colors = _channel_colors(info, config, set(images))
        colored = {
            role: scaled[role][..., None] * np.asarray(colors[role])
            for role in scaled
        }
        tissue = products.tissue_mask[sl]
        labels = products.plaque_labels[sl]
        neuron_like = products.neuron_like_mask[sl]
        no_microglia = products.microglia_absent_plaque_mask[sl]
        plaque_boundary = segmentation.find_boundaries(labels, mode="outer")
        neuron_boundary = segmentation.find_boundaries(neuron_like, mode="outer")
        no_microglia_boundary = segmentation.find_boundaries(
            no_microglia, mode="outer"
        )
        iba1_boundary = segmentation.find_boundaries(masks["iba1"], mode="outer")
        cd68_boundary = segmentation.find_boundaries(masks["cd68"], mode="outer")
        tissue_boundary = segmentation.find_boundaries(tissue, mode="inner")
        ring_union = np.zeros(tissue.shape, dtype=bool)
        for mask in products.ring_masks.values():
            ring_union |= mask[sl]
        ring_boundary = segmentation.find_boundaries(ring_union, mode="outer")

        composite = np.clip(
            sum(
                (colored[role] for role in colored),
                np.zeros((*tissue.shape, 3), dtype=float),
            ),
            0.0,
            1.0,
        )
        plaque_overlay = _overlay(colored["abeta"], plaque_boundary, (0.0, 1.0, 1.0))
        plaque_overlay[neuron_boundary] = np.asarray((1.0, 0.0, 1.0))
        plaque_overlay[no_microglia_boundary] = np.asarray((1.0, 0.5, 0.0))
        iba1_overlay = _overlay(colored["iba1"], iba1_boundary, (0.0, 1.0, 0.0))
        cd68_overlay = _overlay(colored["cd68"], cd68_boundary, (1.0, 1.0, 0.0))
        ring_overlay = _overlay(
            composite,
            ring_boundary,
            (1.0, 1.0, 0.0),
            width_px=int(config["output"].get("ring_boundary_width_px", 1)),
        )
        tissue_overlay = composite.copy()
        tissue_overlay[~tissue] *= 0.15
        tissue_overlay[tissue_boundary] = np.asarray((1.0, 0.0, 1.0))
        dapi_overlay = None
        if "dapi" in colored:
            nucleus_boundary = segmentation.find_boundaries(
                products.nucleus_labels[sl], mode="outer"
            )
            microglia_boundary = segmentation.find_boundaries(
                products.microglia_labels[sl], mode="outer"
            )
            dapi_overlay = _overlay(
                colored["dapi"], nucleus_boundary, (1.0, 1.0, 1.0)
            )
            dapi_overlay[microglia_boundary] = np.asarray((1.0, 1.0, 0.0))

        figure, axes = plt.subplots(3, 3, figsize=(16, 15))
        panels = [
            (colored["abeta"], None, f"Aβ original color | threshold {products.thresholds['abeta']:.1f}"),
            (
                plaque_overlay,
                None,
                "Aβ: plaques cyan / soma magenta / no nearby microglia orange",
            ),
            (iba1_overlay, None, f"Iba1 positive mask | threshold {products.thresholds['iba1']:.1f}"),
            (cd68_overlay, None, f"CD68 positive mask | threshold {products.thresholds['cd68']:.1f}"),
            (composite, None, "Composite: original CZI display colors"),
            (ring_overlay, None, "Plaque-neighborhood outer boundaries (yellow)"),
            (tissue_overlay, None, "Tissue/anatomical ROI (magenta boundary)"),
        ]
        if dapi_overlay is not None:
            panels.insert(
                4,
                (
                    dapi_overlay,
                    None,
                    f"DAPI nuclei white / microglia yellow | threshold {products.thresholds['dapi']:.1f}",
                ),
            )
        for axis, (image, cmap, title) in zip(axes.ravel(), panels):
            axis.imshow(image, cmap=cmap)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        summary_axis = axes.ravel()[-1]
        for axis in axes.ravel()[len(panels) : -1]:
            axis.axis("off")
        summary_axis.axis("off")
        summary = products.image_summary.iloc[0]
        ring_key = next(iter(products.ring_masks), None)
        lines = [
            f"Plaques (all): {int(summary['plaque_count_all'])}",
            f"Interior plaques: {int(summary['plaque_count_interior'])}",
            f"Soma-like excluded: {int(summary['abeta_neuron_like_excluded_count'])}",
            (
                "No-nearby-microglia excluded: "
                f"{int(summary['plaque_without_nearby_microglia_excluded_count'])}"
            ),
            (
                "Iba1 objects: "
                f"{int(summary['iba1_component_accepted_count'])} accepted / "
                f"{int(summary['iba1_component_excluded_count'])} excluded"
            ),
            (
                "CD68 objects: "
                f"{int(summary['cd68_component_accepted_count'])} accepted / "
                f"{int(summary['cd68_component_excluded_count'])} excluded"
            ),
            f"Aβ burden: {summary['abeta_positive_fraction']:.4f}",
            f"Iba1 area fraction: {summary['iba1_positive_fraction']:.4f}",
            f"CD68 area fraction: {summary['cd68_positive_fraction']:.4f}",
            f"CD68∩Iba1 / Iba1: {summary['cd68_in_iba1_fraction_of_iba1']:.4f}",
        ]
        if ring_key:
            lines.extend(
                (
                    f"{ring_key} Iba1 fraction: {summary[f'{ring_key}_iba1_positive_fraction']:.4f}",
                    f"{ring_key} CD68∩Iba1/Iba1: {summary[f'{ring_key}_cd68_in_iba1_fraction_of_iba1']:.4f}",
                )
            )
        if bool(summary["microglia_count_available"]):
            lines.insert(
                3, f"Microglia in ROI: {int(summary['microglia_count_roi'])}"
            )
            if ring_key:
                lines.append(
                    f"{ring_key} microglia: {int(summary[f'{ring_key}_microglia_count'])}"
                )
        summary_axis.text(
            0.02,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=12,
            family="monospace",
        )
        summary_axis.set_title("Preview measurements", fontsize=10)
        figure.tight_layout()
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=125, bbox_inches="tight")
        self.preview_image.value = buffer.getvalue()
        plt.close(figure)

        metrics = [
            ("Plaque count", int(summary["plaque_count_all"])),
            (
                "Soma-like excluded",
                int(summary["abeta_neuron_like_excluded_count"]),
            ),
            (
                "Iba1 accepted / excluded",
                f"{int(summary['iba1_component_accepted_count'])} / "
                f"{int(summary['iba1_component_excluded_count'])}",
            ),
            (
                "CD68 accepted / excluded",
                f"{int(summary['cd68_component_accepted_count'])} / "
                f"{int(summary['cd68_component_excluded_count'])}",
            ),
            ("Aβ area fraction", f"{summary['abeta_positive_fraction']:.5f}"),
            ("Iba1 area fraction", f"{summary['iba1_positive_fraction']:.5f}"),
            ("CD68/Iba1 area ratio", f"{summary['cd68_in_iba1_fraction_of_iba1']:.5f}"),
        ]
        if bool(summary["microglia_count_available"]):
            metrics.insert(
                1, ("Microglia count", int(summary["microglia_count_roi"]))
            )
        self.preview_metrics.value = (
            "<table style='border-collapse:collapse'><tr>"
            + "".join(
                f"<td style='padding:5px 16px;border:1px solid #ccc'><b>{html.escape(label)}</b><br>{value}</td>"
                for label, value in metrics
            )
            + "</tr></table>"
        )

    def _save_parameters(self, button: widgets.Button) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            if not self.image_paths:
                raise ValueError("Select at least one CZI file first.")
            self._snapshot_current()
            config = self._current_config(preview=False)
            document = {
                "format_version": 2,
                "selected_image_files": [
                    str(path) for path in self.image_paths
                ],
                "output_root": str(
                    Path(self.output_root_widget.value).expanduser().resolve()
                ),
                "template_config": copy.deepcopy(config),
                "per_file_overrides": {
                    str(path): copy.deepcopy(
                        self.file_configs[str(path).casefold()]
                    )
                    for path in self.image_paths
                    if str(path).casefold() in self.file_configs
                },
            }
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update()
            try:
                selected = filedialog.asksaveasfilename(
                    parent=root,
                    title="Save cell-analysis session",
                    initialfile="cell_analysis_session.yaml",
                    defaultextension=".yaml",
                    filetypes=(("YAML", "*.yaml"), ("All files", "*.*")),
                )
            finally:
                root.destroy()
            if selected:
                save_config(document, selected)
                files_path = Path(selected).with_suffix(".files.txt")
                files_path.write_text(
                    "\n".join(document["selected_image_files"]) + "\n",
                    encoding="utf-8",
                )
                self.config = copy.deepcopy(config)
                self.status.value = (
                    f"<b>Session saved:</b> <code>{html.escape(selected)}</code><br>"
                    f"<b>File list:</b> <code>{html.escape(str(files_path))}</code>"
                )
        except Exception as error:
            self.status.value = (
                f"<span style='color:#b00020'><b>Save error:</b> {html.escape(str(error))}</span>"
            )

    def load_session(self, session_path: str | Path) -> None:
        document = load_config(session_path)
        template = copy.deepcopy(document.get("template_config", document))
        paths = document.get("selected_image_files") or [
            template.get("input", {}).get("image_path")
        ]
        paths = self._validated_paths(paths)
        raw_configs = (
            document.get("effective_file_configs")
            or document.get("per_file_overrides")
            or {}
        )
        output_root = document.get("output_root")
        if not output_root:
            output_dir = template.get("input", {}).get("output_dir")
            output_root = (
                str(Path(output_dir).expanduser().resolve().parent)
                if output_dir
                else str(Path.cwd() / "cell_analysis_results")
            )
        file_configs: dict[str, dict[str, Any]] = {}
        for path in paths:
            source = next(
                (
                    value
                    for key, value in raw_configs.items()
                    if str(Path(key).expanduser().resolve()).casefold()
                    == str(path).casefold()
                ),
                template,
            )
            item = copy.deepcopy(source)
            item.setdefault("input", {})["image_path"] = str(path)
            item["input"]["output_dir"] = str(Path(output_root) / path.stem)
            file_configs[str(path).casefold()] = item
        self.image_paths = []
        self.file_selector.options = ()
        self.file_configs = file_configs
        self.config = copy.deepcopy(template)
        self.output_root_widget.value = str(Path(output_root).expanduser().resolve())
        self._channel_mapping_initialized = True
        self._set_paths(paths)
        self.status.value = (
            f"<b>Session loaded:</b> {len(paths)} files from "
            f"<code>{html.escape(str(session_path))}</code>"
        )

    def _choose_session(self, button: widgets.Button) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update()
            try:
                selected = filedialog.askopenfilename(
                    parent=root,
                    title="Load cell-analysis session",
                    filetypes=(("YAML", "*.yaml *.yml"), ("All files", "*.*")),
                )
            finally:
                root.destroy()
            if selected:
                self.load_session(selected)
        except Exception as error:
            self.status.value = (
                f"<span style='color:#b00020'><b>Load error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def _render_result_table(self, result: dict[str, Any]) -> None:
        rows = []
        for record in result["records"]:
            rows.append(
                "<tr>"
                f"<td>{record['batch_file_index']}</td>"
                f"<td>{html.escape(record['source_name'])}</td>"
                f"<td>{html.escape(record['status'])}</td>"
                f"<td>{record['plaque_count_all']}</td>"
                f"<td>{html.escape(str(record.get('error', '')))}</td>"
                "</tr>"
            )
        self.result_table.value = (
            "<table style='border-collapse:collapse;width:100%'>"
            "<thead><tr><th>#</th><th>File</th><th>Status</th><th>Plaques</th><th>Error</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>"
        )

    def _start_batch(self, button: widgets.Button) -> None:
        try:
            if not self.image_paths:
                raise ValueError("Select at least one supported microscopy image.")
            self._snapshot_current()
            template = self._current_config(preview=False)
            output_root = Path(self.output_root_widget.value).expanduser().resolve()
            paths = list(self.image_paths)
            per_file_configs = copy.deepcopy(self.file_configs)
            self.config = copy.deepcopy(template)
            self.batch_result = None
            self.result_table.value = ""
            self._set_busy(True)
            self.status.value = "<b>Starting batch analysis...</b>"

            def worker() -> None:
                try:
                    result = run_batch_analysis(
                        paths,
                        copy.deepcopy(template),
                        output_root,
                        per_file_configs=per_file_configs,
                        progress=lambda message: setattr(
                            self.status, "value", f"<b>{html.escape(str(message))}</b>"
                        ),
                    )
                    self.batch_result = result
                    self._render_result_table(result)
                    self.status.value = (
                        f"<b>Batch complete:</b> {result['completed']} completed, "
                        f"{result['failed']} failed. Results: "
                        f"<code>{html.escape(result['output_root'])}</code>"
                    )
                except Exception as error:
                    self.status.value = (
                        f"<span style='color:#b00020'><b>Batch error:</b> "
                        f"{html.escape(str(error))}</span>"
                    )
                finally:
                    self._set_busy(False)

            threading.Thread(target=worker, daemon=True).start()
        except Exception as error:
            self._set_busy(False)
            self.status.value = (
                f"<span style='color:#b00020'><b>Batch setup error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def run_batch(self, progress: Any = print) -> dict[str, Any]:
        """Run the current settings synchronously and return the result dictionary."""

        self._snapshot_current()
        template = self._current_config(preview=False)
        result = run_batch_analysis(
            self.image_paths,
            template,
            self.output_root_widget.value,
            per_file_configs=copy.deepcopy(self.file_configs),
            progress=progress,
        )
        self.config = copy.deepcopy(template)
        self.batch_result = result
        self._render_result_table(result)
        return result

    def _ipython_display_(self) -> None:
        display(self.widget)


def launch_brain_section_tuning_widget(
    config: dict[str, Any] | None = None,
    image_paths: Iterable[str | Path] | None = None,
    output_root: str | Path | None = None,
    max_preview_dimension: int = 1400,
) -> BrainSectionBatchTuningPanel:
    """Return the notebook file selector, preview tuner, and batch runner."""

    return BrainSectionBatchTuningPanel(
        config=config,
        image_paths=image_paths,
        output_root=output_root,
        max_preview_dimension=max_preview_dimension,
    )
