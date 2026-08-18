"""Live notebook controls for channel smoothing and cell segmentation."""

from __future__ import annotations

import copy
import html
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from skimage import measure, segmentation

from .batch import prepare_batch_config, run_batch_analysis
from .config import normalize_config
from .czi_io import inspect_czi, read_czi_channels
from .preprocessing import preprocess_channel_steps
from .segmentation import segment_cells_steps, threshold_image


def _display_scale(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image)]
    nonzero = values[values > 0]
    values = nonzero if nonzero.size >= 32 else values
    if values.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(values, [1.0, 99.8])
    high = high if high > low else low + 1.0
    return np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def _preview_channel_config(config: dict[str, Any], scale: float) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["gaussian_sigma_px"] = float(result["gaussian_sigma_px"]) * scale
    return result


def _preview_segmentation_config(config: dict[str, Any], scale: float) -> dict[str, Any]:
    result = copy.deepcopy(config)
    linear_keys = (
        "opening_radius_px",
        "closing_radius_px",
        "local_contrast_ring_px",
        "min_peak_distance_px",
        "adaptive_block_size_px",
        "watershed_min_peak_height_px",
        "watershed_min_peak_prominence_px",
        "border_exclusion_margin_px",
    )
    for key in linear_keys:
        minimum = 1 if key in {"local_contrast_ring_px", "min_peak_distance_px"} else 0
        result[key] = max(minimum, round(float(result.get(key, 0)) * scale))
    result["min_hole_area_px"] = max(
        0, round(float(result.get("min_hole_area_px", 0)) * scale**2)
    )
    result["min_area_px"] = max(
        1, round(float(result.get("min_area_px", 1)) * scale**2)
    )
    if result.get("max_area_px") is not None:
        result["max_area_px"] = max(1, round(float(result["max_area_px"]) * scale**2))
    return result


class LiveTuningPanel:
    """Config-mutating live preview for Jupyter notebooks."""

    def __init__(self, config: dict[str, Any], max_preview_dimension: int = 1400) -> None:
        self.config = config
        self.info = inspect_czi(config["input"]["czi_path"])
        normalized_config = normalize_config(config, self.info)
        self.config.clear()
        self.config.update(normalized_config)
        self.analysis_zoom = float(self.config["input"]["zoom"])
        scene_index = int(self.config["input"]["scene"])
        scene = next(item for item in self.info.scenes if item.index == scene_index)
        preview_cap = max_preview_dimension / max(scene.width, scene.height)
        self.preview_zoom = max(0.01, min(self.analysis_zoom, preview_cap, 1.0))
        self.pixel_scale = self.preview_zoom / self.analysis_zoom
        self.scene_index = scene_index
        self.images: dict[int, np.ndarray] | None = None
        self._suspend = False
        self._build()
        self._load_channel(int(self.parameter_channel.value))
        self._connect()
        self.status.value = (
            "<b>Ready.</b> Click <b>Load preview</b> to read the selected CZI plane."
        )

    @staticmethod
    def _float_slider(
        label: str, value: float, minimum: float, maximum: float, step: float
    ) -> widgets.FloatSlider:
        return widgets.FloatSlider(
            description=label,
            value=float(np.clip(value, minimum, maximum)),
            min=minimum,
            max=maximum,
            step=step,
            continuous_update=False,
            style={"description_width": "165px"},
            layout=widgets.Layout(width="430px"),
        )

    @staticmethod
    def _int_slider(label: str, value: int, maximum: int) -> widgets.IntSlider:
        return widgets.IntSlider(
            description=label,
            value=int(np.clip(value, 0, maximum)),
            min=0,
            max=maximum,
            continuous_update=False,
            style={"description_width": "165px"},
            layout=widgets.Layout(width="430px"),
        )

    @staticmethod
    def _dropdown(label: str, options: Any, value: Any) -> widgets.Dropdown:
        return widgets.Dropdown(
            description=label,
            options=options,
            value=value,
            style={"description_width": "165px"},
            layout=widgets.Layout(width="430px"),
        )

    def _build(self) -> None:
        channel_options = [
            (f"C{channel.index}: {channel.name}", channel.index)
            for channel in self.info.channels
        ]
        selected = int(self.config["input"]["segmentation_channel"])
        self.parameter_channel = self._dropdown(
            "Parameter/image channel", channel_options, selected
        )
        self.segmentation_channel = self._dropdown(
            "Segmentation channel", channel_options, selected
        )
        first = self.config["channels"][str(selected)]
        measurement = first["measurement_threshold"]
        self.sigma = self._float_slider(
            "Gaussian sigma (px)", first["gaussian_sigma_px"], 0, 12, 0.1
        )
        signal_methods = ("none", "otsu", "yen", "triangle", "percentile")
        self.signal_method = self._dropdown(
            "Signal area threshold", signal_methods, measurement.get("method", "otsu")
        )
        self.signal_percentile = self._float_slider(
            "Signal percentile", measurement.get("percentile", 95), 50, 100, 0.5
        )

        segment = self.config["segmentation"]
        cell_methods = ("otsu", "yen", "triangle", "percentile", "adaptive")
        self.cell_method = self._dropdown(
            "Cell threshold", cell_methods, segment["threshold_method"]
        )
        self.threshold_scale = self._float_slider(
            "Automatic threshold multiplier",
            segment.get("threshold_scale", 1.0),
            0.2,
            1.5,
            0.02,
        )
        self.cell_percentile = self._float_slider(
            "Cell percentile", segment.get("threshold_percentile", 90), 50, 100, 0.5
        )
        self.adaptive_block = self._int_slider(
            "Adaptive block size", segment.get("adaptive_block_size_px", 51), 301
        )
        self.adaptive_offset = self._float_slider(
            "Adaptive threshold offset", segment.get("adaptive_offset", 0), -0.5, 0.5, 0.005
        )
        self.opening = self._int_slider(
            "Opening radius (px)", segment["opening_radius_px"], 30
        )
        self.closing = self._int_slider(
            "Closing radius (px)", segment["closing_radius_px"], 30
        )
        self.fill_holes = widgets.Checkbox(
            description="Fill all enclosed holes",
            value=bool(segment.get("fill_all_holes", True)),
            indent=False,
        )
        self.hole_area = widgets.BoundedIntText(
            description="Small-hole area (px)",
            value=int(segment.get("min_hole_area_px", 32)),
            min=0,
            max=100_000_000,
            style={"description_width": "165px"},
        )
        self.min_area = widgets.BoundedIntText(
            description="Minimum cell area (px^2)",
            value=int(segment["min_area_px"]),
            min=1,
            max=100_000_000,
            style={"description_width": "190px"},
        )
        self.max_area = widgets.BoundedIntText(
            description="Maximum cell area (px^2; 0=none)",
            value=int(segment.get("max_area_px") or 0),
            min=0,
            max=100_000_000,
            style={"description_width": "230px"},
        )
        self.circularity = self._float_slider(
            "Minimum circularity", segment["min_circularity"], 0, 1, 0.01
        )
        self.contrast = self._float_slider(
            "Minimum local contrast", segment["min_local_contrast_ratio"], 0, 5, 0.05
        )
        self.split = widgets.Checkbox(
            description="Split touching cells", value=bool(segment["split_touching"]), indent=False
        )
        self.peak = self._int_slider(
            "Watershed peak distance", segment["min_peak_distance_px"], 100
        )
        self.peak_height = self._float_slider(
            "Watershed minimum height",
            segment.get("watershed_min_peak_height_px", 0),
            0,
            100,
            0.5,
        )
        self.peak_prominence = self._float_slider(
            "Watershed peak prominence",
            segment.get("watershed_min_peak_prominence_px", 0),
            0,
            50,
            0.5,
        )
        self.compactness = self._float_slider(
            "Watershed compactness",
            segment.get("watershed_compactness", 0),
            0,
            10,
            0.1,
        )
        self.clear_border = widgets.Checkbox(
            description="Clear border ROIs", value=bool(segment["clear_border"]), indent=False
        )
        self.border_margin = widgets.BoundedIntText(
            description="Border exclusion margin (px)",
            value=int(segment.get("border_exclusion_margin_px", 10)),
            min=0,
            max=100_000,
            style={"description_width": "190px"},
        )

        channel_box = widgets.VBox(
            [
                self.parameter_channel,
                self.sigma,
                self.signal_method,
                self.signal_percentile,
            ]
        )
        segmentation_box = widgets.VBox(
            [
                self.segmentation_channel,
                widgets.HTML(
                    "<b>Cell size filters</b> — candidates outside this area range "
                    "are removed from the accepted ROI preview."
                ),
                self.min_area,
                self.max_area,
                self.cell_method,
                self.threshold_scale,
                self.cell_percentile,
                self.adaptive_block,
                self.adaptive_offset,
                self.opening,
                self.closing,
                self.fill_holes,
                self.hole_area,
                self.circularity,
                self.contrast,
                self.split,
                self.peak,
                self.peak_height,
                self.peak_prominence,
                self.compactness,
                self.clear_border,
                self.border_margin,
            ]
        )
        self.accordion = widgets.Accordion(children=(channel_box, segmentation_box))
        self.accordion.set_title(0, "Per-channel smoothing and signal area")
        self.accordion.set_title(1, "Cell segmentation and size filters")
        self.accordion.selected_index = 1
        self.load_button = widgets.Button(
            description="Load preview",
            button_style="primary",
            icon="image",
            tooltip="Read the configured CZI plane and draw the live preview",
        )
        self.status = widgets.HTML()
        self.output = widgets.Image(
            format="png",
            layout=widgets.Layout(border="1px solid #cccccc", width="100%"),
        )
        self.widget = widgets.VBox(
            (self.accordion, self.load_button, self.status, self.output)
        )

    def _controls(self) -> tuple[widgets.Widget, ...]:
        return (
            self.segmentation_channel,
            self.sigma,
            self.signal_method,
            self.signal_percentile,
            self.cell_method,
            self.threshold_scale,
            self.cell_percentile,
            self.adaptive_block,
            self.adaptive_offset,
            self.opening,
            self.closing,
            self.fill_holes,
            self.hole_area,
            self.min_area,
            self.max_area,
            self.circularity,
            self.contrast,
            self.split,
            self.peak,
            self.peak_height,
            self.peak_prominence,
            self.compactness,
            self.clear_border,
            self.border_margin,
        )

    def _connect(self) -> None:
        self.load_button.on_click(self._load_preview)
        self.parameter_channel.observe(self._change_parameter_channel, names="value")
        for control in self._controls():
            control.observe(self._changed, names="value")

    def _change_parameter_channel(self, change: dict[str, Any]) -> None:
        if not self._suspend:
            self._load_channel(int(change["new"]))
            self.refresh()

    def _changed(self, change: dict[str, Any]) -> None:
        if not self._suspend:
            self.refresh()

    def _load_preview(self, button: widgets.Button) -> None:
        self.load_button.disabled = True
        self.status.value = "<b>Loading CZI preview...</b>"
        try:
            self.images = read_czi_channels(
                self.info.path,
                self.info,
                scene=self.scene_index,
                time_index=int(self.config["input"]["time_index"]),
                z_projection=str(self.config["input"]["z_projection"]),
                z_index=int(self.config["input"]["z_index"]),
                zoom=self.preview_zoom,
            )
            self.load_button.description = "Reload preview"
            self.refresh()
        except Exception as error:
            self.status.value = (
                f"<span style='color:#b00020'><b>Load error:</b> {error}</span>"
            )
        finally:
            self.load_button.disabled = False

    def _load_channel(self, channel_index: int) -> None:
        item = self.config["channels"][str(channel_index)]
        measurement = item["measurement_threshold"]
        self._suspend = True
        try:
            self.sigma.value = float(item["gaussian_sigma_px"])
            self.signal_method.value = str(measurement.get("method", "otsu"))
            self.signal_percentile.value = float(measurement.get("percentile", 95))
        finally:
            self._suspend = False

    def _store(self) -> None:
        channel_index = int(self.parameter_channel.value)
        channel = self.config["channels"][str(channel_index)]
        channel.update(
            {
                "gaussian_sigma_px": float(self.sigma.value),
                "measurement_threshold": {
                    "method": str(self.signal_method.value),
                    "percentile": float(self.signal_percentile.value),
                },
            }
        )
        self.config["input"]["segmentation_channel"] = int(self.segmentation_channel.value)
        segment = self.config["segmentation"]
        segment.update(
            {
                "threshold_method": str(self.cell_method.value),
                "threshold_scale": float(self.threshold_scale.value),
                "threshold_percentile": float(self.cell_percentile.value),
                "adaptive_block_size_px": max(3, int(self.adaptive_block.value)),
                "adaptive_offset": float(self.adaptive_offset.value),
                "opening_radius_px": int(self.opening.value),
                "closing_radius_px": int(self.closing.value),
                "fill_all_holes": bool(self.fill_holes.value),
                "min_hole_area_px": int(self.hole_area.value),
                "min_area_px": int(self.min_area.value),
                "max_area_px": int(self.max_area.value) or None,
                "min_circularity": float(self.circularity.value),
                "min_local_contrast_ratio": float(self.contrast.value),
                "split_touching": bool(self.split.value),
                "min_peak_distance_px": max(1, int(self.peak.value)),
                "watershed_min_peak_height_px": float(self.peak_height.value),
                "watershed_min_peak_prominence_px": float(
                    self.peak_prominence.value
                ),
                "watershed_compactness": float(self.compactness.value),
                "clear_border": bool(self.clear_border.value),
                "border_exclusion_margin_px": int(self.border_margin.value),
            }
        )

    def refresh(self) -> None:
        """Save control values in config and redraw the preview."""

        try:
            self._store()
            if self.images is None:
                self.status.value = (
                    "<b>Ready.</b> Click <b>Load preview</b> to read the selected CZI plane."
                )
                return
            parameter_index = int(self.parameter_channel.value)
            segmentation_index = int(self.segmentation_channel.value)
            parameter_processed, _ = preprocess_channel_steps(
                self.images[parameter_index],
                _preview_channel_config(
                    self.config["channels"][str(parameter_index)], self.pixel_scale
                ),
            )
            if parameter_index == segmentation_index:
                segmentation_processed = parameter_processed
            else:
                segmentation_processed, _ = preprocess_channel_steps(
                    self.images[segmentation_index],
                    _preview_channel_config(
                        self.config["channels"][str(segmentation_index)], self.pixel_scale
                    ),
                )
            labels, diagnostics, segmentation_steps = segment_cells_steps(
                segmentation_processed.analysis_image,
                _preview_segmentation_config(self.config["segmentation"], self.pixel_scale),
            )
            measurement = self.config["channels"][str(parameter_index)][
                "measurement_threshold"
            ]
            positive, signal_threshold = threshold_image(
                parameter_processed.analysis_image,
                method=measurement["method"],
                percentile=float(measurement.get("percentile", 95)),
            )
            self._draw(
                parameter_index,
                segmentation_index,
                parameter_processed,
                segmentation_processed,
                labels,
                positive,
                segmentation_steps,
            )
            zoom_note = (
                f"preview zoom {self.preview_zoom:.4g}, analysis zoom {self.analysis_zoom:.4g}"
            )
            self.status.value = (
                f"<b>{int(labels.max())} ROIs</b> | "
                f"{diagnostics.get('connected_component_count')} cleaned objects | cell threshold "
                f"{diagnostics.get('threshold')} | signal threshold {signal_threshold} | "
                f"rejected by area {diagnostics.get('rejected_area')} | "
                f"rejected by border {diagnostics.get('rejected_border')} | "
                f"{zoom_note}. Controls update <code>config</code> automatically."
            )
        except Exception as error:
            self.status.value = (
                f"<span style='color:#b00020'><b>Preview error:</b> {error}</span>"
            )

    def _draw(
        self,
        parameter_index: int,
        segmentation_index: int,
        parameter_processed: Any,
        segmentation_processed: Any,
        labels: np.ndarray,
        positive: np.ndarray,
        segmentation_steps: dict[str, np.ndarray],
    ) -> None:
        raw = _display_scale(self.images[parameter_index])
        smoothed = _display_scale(parameter_processed.analysis_image)
        segment_image = _display_scale(segmentation_processed.analysis_image)
        boundaries = segmentation.find_boundaries(labels, mode="outer")
        roi_overlay = np.repeat(segment_image[..., None], 3, axis=2)
        roi_overlay[boundaries] = np.array([1.0, 0.1, 0.1])
        signal_overlay = np.repeat(
            smoothed[..., None], 3, axis=2
        )
        positive_inside = positive & (labels > 0)
        signal_overlay[positive_inside] = (
            0.35 * signal_overlay[positive_inside]
            + 0.65 * np.array([0.0, 1.0, 1.0])
        )
        signal_overlay[boundaries] = np.array([1.0, 0.1, 0.1])
        candidate_overlay = np.repeat(
            segment_image[..., None], 3, axis=2
        )
        candidate_boundaries = segmentation.find_boundaries(
            segmentation_steps["candidate_labels"], mode="outer"
        )
        candidate_overlay[candidate_boundaries] = np.array([1.0, 0.85, 0.0])
        names = {channel.index: channel.name for channel in self.info.channels}

        figure, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.ravel()
        panels = [
            (raw, "gray", f"1. Raw: C{parameter_index} {names[parameter_index]}", 0, 1),
            (
                smoothed,
                "gray",
                "2. Gaussian smoothed",
                0,
                1,
            ),
            (
                segment_image,
                "gray",
                f"3. Segmentation input: C{segmentation_index}",
                0,
                1,
            ),
            (
                segmentation_steps["initial_threshold_mask"],
                "gray",
                "4. Initial threshold mask",
                0,
                1,
            ),
            (
                segmentation_steps["cleaned_mask"],
                "gray",
                "5. Morphology-cleaned mask",
                0,
                1,
            ),
            (candidate_overlay, None, "6. Watershed candidates", None, None),
            (roi_overlay, None, "7. Accepted cell boundaries", None, None),
            (signal_overlay, None, "8. Positive signal inside ROIs", None, None),
        ]
        for axis, (image, cmap, title, minimum, maximum) in zip(axes, panels):
            axis.imshow(image, cmap=cmap, vmin=minimum, vmax=maximum)
            axis.set_title(title, fontsize=11)
        if int(labels.max()) <= 100:
            for region in measure.regionprops(labels):
                y, x = region.centroid
                axes[6].text(
                    x,
                    y,
                    str(region.label),
                    color="yellow",
                    fontsize=7,
                    ha="center",
                    va="center",
                )
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        self.output.value = buffer.getvalue()
        plt.close(figure)

    def _ipython_display_(self) -> None:
        display(self.widget)


def launch_tuning_widget(
    config: dict[str, Any], max_preview_dimension: int = 1400
) -> LiveTuningPanel:
    """Load the configured CZI plane once and return the live tuning panel."""

    return LiveTuningPanel(config, max_preview_dimension=max_preview_dimension)


class BatchTuningPanel:
    """Multi-file notebook interface with shared parameters and per-file previews."""

    def __init__(
        self,
        config: dict[str, Any],
        czi_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
        output_root: str | Path | None = None,
        max_preview_dimension: int = 1400,
    ) -> None:
        self.config = config
        self.max_preview_dimension = int(max_preview_dimension)
        self.current_panel: LiveTuningPanel | None = None
        self.batch_result: dict[str, Any] | None = None
        self.czi_paths: list[Path] = []
        self.file_configs: dict[str, dict[str, Any]] = {}
        self._switching = False
        configured_path = config.get("input", {}).get("czi_path")
        initial_paths = list(czi_paths or ([configured_path] if configured_path else []))
        configured_output = config.get("input", {}).get("output_dir")
        initial_output = output_root or configured_output or Path.cwd() / "batch_results"
        self._build(initial_output)
        self._set_paths(initial_paths)

    def _build(self, output_root: str | Path) -> None:
        self.path_text = widgets.Textarea(
            description="CZI files",
            placeholder="One absolute .czi path per line",
            layout=widgets.Layout(width="100%", height="110px"),
            style={"description_width": "90px"},
        )
        self.select_button = widgets.Button(
            description="Select CZI files",
            button_style="info",
            icon="folder-open",
        )
        self.apply_paths_button = widgets.Button(
            description="Apply file list",
            icon="check",
        )
        self.apply_all_button = widgets.Button(
            description="Apply current settings to all",
            icon="copy",
            tooltip="Use the current image settings as the template for every file",
        )
        self.file_selector = widgets.Dropdown(
            description="Preview file",
            options=(),
            style={"description_width": "90px"},
            layout=widgets.Layout(width="70%"),
        )
        self.previous_button = widgets.Button(description="Previous", icon="arrow-left")
        self.next_button = widgets.Button(description="Next", icon="arrow-right")
        self.output_root_widget = widgets.Text(
            description="Output root",
            value=str(Path(output_root).expanduser()),
            style={"description_width": "90px"},
            layout=widgets.Layout(width="100%"),
        )
        self.run_button = widgets.Button(
            description="Run all files",
            button_style="success",
            icon="play",
        )
        self.batch_status = widgets.HTML(value="<b>Select one or more CZI files.</b>")
        self.result_table = widgets.HTML()
        self.preview_container = widgets.VBox()

        self.select_button.on_click(self._choose_files)
        self.apply_paths_button.on_click(self._apply_text_paths)
        self.apply_all_button.on_click(self._apply_current_to_all)
        self.previous_button.on_click(lambda button: self._move(-1))
        self.next_button.on_click(lambda button: self._move(1))
        self.run_button.on_click(self._start_batch)
        self.file_selector.observe(self._change_file, names="value")

        file_buttons = widgets.HBox(
            (self.select_button, self.apply_paths_button, self.apply_all_button)
        )
        navigation = widgets.HBox(
            (self.file_selector, self.previous_button, self.next_button)
        )
        batch_controls = widgets.VBox(
            (
                widgets.HTML(
                    "<h3>Multi-file CZI analysis</h3>"
                    "<p>Each file remembers its own parameters. Select a file to preview "
                    "and tune it, or copy the current settings to the complete batch.</p>"
                ),
                self.path_text,
                file_buttons,
                navigation,
                self.output_root_widget,
                self.run_button,
                self.batch_status,
                self.result_table,
            )
        )
        self.widget = widgets.VBox((batch_controls, self.preview_container))

    @staticmethod
    def _validated_paths(values: list[str | Path]) -> list[Path]:
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
                raise FileNotFoundError(f"CZI file not found: {path}")
            if path.suffix.casefold() != ".czi":
                raise ValueError(f"Expected a .czi file: {path}")
            paths.append(path)
            seen.add(key)
        if not paths:
            raise ValueError("Select at least one CZI file.")
        return paths

    def _set_paths(self, values: list[str | Path]) -> None:
        try:
            paths = self._validated_paths(values)
            self._snapshot_current()
            current = str(self.file_selector.value or "")
            options = [
                (f"{index}. {path.name}", str(path))
                for index, path in enumerate(paths, start=1)
            ]
            selected = current if current in {value for _, value in options} else options[0][1]
            self._switching = True
            try:
                self.czi_paths = paths
                valid_keys = {str(path).casefold() for path in paths}
                self.file_configs = {
                    key: value
                    for key, value in self.file_configs.items()
                    if key in valid_keys
                }
                self.path_text.value = "\n".join(str(path) for path in paths)
                self.file_selector.options = options
                self.file_selector.value = selected
            finally:
                self._switching = False
            self._load_selected_panel()
        except Exception as error:
            self.batch_status.value = (
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
                    title="Select CZI files",
                    filetypes=(("CZI images", "*.czi"), ("All files", "*.*")),
                )
            finally:
                root.destroy()
            if selected:
                self._set_paths(list(selected))
        except Exception as error:
            self.batch_status.value = (
                f"<span style='color:#b00020'><b>File dialog error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def _apply_text_paths(self, button: widgets.Button) -> None:
        self._set_paths(self.path_text.value.splitlines())

    def _snapshot_current(self) -> None:
        if self.current_panel is None:
            return
        self.current_panel._store()
        current = self.current_panel.config
        key = str(Path(current["input"]["czi_path"]).resolve()).casefold()
        self.file_configs[key] = copy.deepcopy(current)

    def _copy_parameters_to_template(self, current: dict[str, Any]) -> None:
        self.config.setdefault("input", {})
        for key in (
            "scene",
            "time_index",
            "z_projection",
            "z_index",
            "zoom",
            "segmentation_channel",
        ):
            self.config["input"][key] = copy.deepcopy(current["input"][key])
        self.config["input"]["czi_path"] = current["input"]["czi_path"]
        self.config["segmentation"] = copy.deepcopy(current["segmentation"])
        self.config["output"] = copy.deepcopy(current["output"])
        template_channels = self.config.setdefault("channels", {})
        for key, value in current["channels"].items():
            template_channels[key] = copy.deepcopy(value)

    def _apply_current_to_all(self, button: widgets.Button) -> None:
        try:
            self._snapshot_current()
            if self.current_panel is None:
                raise ValueError("Select and load a preview file first.")
            self._copy_parameters_to_template(self.current_panel.config)
            current_key = str(
                Path(self.current_panel.config["input"]["czi_path"]).resolve()
            ).casefold()
            current_config = copy.deepcopy(self.current_panel.config)
            self.file_configs.clear()
            self.file_configs[current_key] = current_config
            self.batch_status.value = (
                "<b>Current settings will be used as the template for all files.</b> "
                "Files can still be adjusted individually afterward."
            )
        except Exception as error:
            self.batch_status.value = (
                f"<span style='color:#b00020'><b>Parameter copy error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def _load_selected_panel(self) -> None:
        selected = self.file_selector.value
        if not selected:
            self.preview_container.children = ()
            return
        try:
            output_root = Path(self.output_root_widget.value).expanduser()
            preview_output = output_root / Path(selected).stem
            saved_config = self.file_configs.get(str(Path(selected).resolve()).casefold())
            current_config = prepare_batch_config(
                selected,
                saved_config or self.config,
                preview_output,
            )
            self.current_panel = LiveTuningPanel(
                current_config,
                max_preview_dimension=self.max_preview_dimension,
            )
            self.preview_container.children = (self.current_panel.widget,)
            position = next(
                index
                for index, path in enumerate(self.czi_paths, start=1)
                if str(path) == str(selected)
            )
            self.batch_status.value = (
                f"<b>Previewing {position}/{len(self.czi_paths)}:</b> "
                f"{html.escape(Path(selected).name)}. Click <b>Load preview</b> below."
            )
        except Exception as error:
            self.current_panel = None
            self.preview_container.children = ()
            self.batch_status.value = (
                f"<span style='color:#b00020'><b>Preview setup error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def _change_file(self, change: dict[str, Any]) -> None:
        if self._switching or not change.get("new"):
            return
        try:
            self._snapshot_current()
            self._load_selected_panel()
        except Exception as error:
            self.batch_status.value = (
                f"<span style='color:#b00020'><b>File switch error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def _move(self, step: int) -> None:
        values = [value for _, value in self.file_selector.options]
        if not values:
            return
        current = values.index(self.file_selector.value)
        self.file_selector.value = values[(current + step) % len(values)]

    def _set_busy(self, busy: bool) -> None:
        for control in (
            self.select_button,
            self.apply_paths_button,
            self.apply_all_button,
            self.file_selector,
            self.previous_button,
            self.next_button,
            self.output_root_widget,
            self.run_button,
        ):
            control.disabled = busy

    def _render_result_table(self, result: dict[str, Any]) -> None:
        rows = []
        for record in result["records"]:
            error = html.escape(str(record.get("error", "")))
            rows.append(
                "<tr>"
                f"<td>{record['batch_file_index']}</td>"
                f"<td>{html.escape(record['source_name'])}</td>"
                f"<td>{html.escape(record['status'])}</td>"
                f"<td>{record['roi_count']}</td>"
                f"<td>{error}</td>"
                "</tr>"
            )
        self.result_table.value = (
            "<table style='border-collapse:collapse;width:100%'>"
            "<thead><tr><th>#</th><th>File</th><th>Status</th>"
            "<th>ROIs</th><th>Error</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    def _start_batch(self, button: widgets.Button) -> None:
        try:
            self._snapshot_current()
            paths = list(self.czi_paths)
            if not paths:
                raise ValueError("Select at least one CZI file.")
            output_root = Path(self.output_root_widget.value).expanduser().resolve()
            template = copy.deepcopy(self.config)
            per_file_configs = copy.deepcopy(self.file_configs)
            self._set_busy(True)
            self.batch_result = None
            self.result_table.value = ""
            self.batch_status.value = "<b>Starting batch analysis...</b>"

            def worker() -> None:
                try:
                    result = run_batch_analysis(
                        paths,
                        template,
                        output_root,
                        per_file_configs=per_file_configs,
                        progress=lambda message: setattr(
                            self.batch_status,
                            "value",
                            f"<b>{html.escape(str(message))}</b>",
                        ),
                    )
                    self.batch_result = result
                    self._render_result_table(result)
                    self.batch_status.value = (
                        f"<b>Batch complete:</b> {result['completed']} completed, "
                        f"{result['failed']} failed, {result['total_roi_count']} total ROIs. "
                        f"Results: <code>{html.escape(result['output_root'])}</code>"
                    )
                except Exception as error:
                    self.batch_status.value = (
                        f"<span style='color:#b00020'><b>Batch error:</b> "
                        f"{html.escape(str(error))}</span>"
                    )
                finally:
                    self._set_busy(False)

            threading.Thread(target=worker, daemon=True).start()
        except Exception as error:
            self._set_busy(False)
            self.batch_status.value = (
                f"<span style='color:#b00020'><b>Batch setup error:</b> "
                f"{html.escape(str(error))}</span>"
            )

    def run_batch(self, progress: Any = print) -> dict[str, Any]:
        """Run the current batch synchronously and return its result dictionary."""

        self._snapshot_current()
        result = run_batch_analysis(
            self.czi_paths,
            copy.deepcopy(self.config),
            self.output_root_widget.value,
            per_file_configs=copy.deepcopy(self.file_configs),
            progress=progress,
        )
        self.batch_result = result
        self._render_result_table(result)
        return result

    def _ipython_display_(self) -> None:
        display(self.widget)


def launch_batch_tuning_widget(
    config: dict[str, Any],
    czi_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    output_root: str | Path | None = None,
    max_preview_dimension: int = 1400,
) -> BatchTuningPanel:
    """Return a multi-file selector, preview browser, and batch runner."""

    return BatchTuningPanel(
        config,
        czi_paths=czi_paths,
        output_root=output_root,
        max_preview_dimension=max_preview_dimension,
    )
