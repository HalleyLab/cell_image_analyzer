"""Standalone Tk interface for cell fluorescence analysis."""

from __future__ import annotations

import copy
import os
import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any

from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

from cell_analyzer.image_io import (
    CACHE_DIRECTORY,
    MICROSCOPY_FILE_PATTERN,
    SUPPORTED_IMAGE_SUFFIXES,
    configure_cache_directory,
    inspect_image,
)

from .analysis import run_analysis, _channel_colors
from .advanced import DRAWING_DEFAULTS, ROLES, STAGE_LABELS, roles_for_count
from .advanced_features import ADVANCED_TABLES, FEATURE_DEFAULTS, FEATURE_IMAGES
from .plots import save_qc
from .batch import _resolve_mask, run_batch_analysis
from .config import (
    DEFAULT_CONFIG,
    OUTPUT_SELECTION_DEFAULTS,
    QC_PANEL_DEFAULTS,
    QC_PANEL_CHOICES,
    PROCESSING_CHOICES,
    default_channel,
    load_config,
    normalize_config,
    processing_choices,
    qc_panel_choices,
    save_config,
)


ROLE_LABELS = dict(zip(ROLES, ("Channel 1", "Channel 2", "Channel 3", "Channel 4")))
ROLE_BY_LABEL = {label: role for role, label in ROLE_LABELS.items()}
THRESHOLD_METHODS = ("manual", "otsu", "yen", "triangle", "percentile")
PREVIEW_IMAGE_OUTPUT_KEYS = (
    "save_qc",
    "save_raw_channel_images",
    "save_composite_image",
    "save_segmentation_images",
    "save_mask_images",
    "save_stage_images",
    "save_advanced_images",
)
QC_PANEL_LABELS = {
    "05_composite": "Composite",
    "06_primary_object_segmentation": "Neighbour-reference segmentation",
    "07_channel_1_objects": "Channel 1 object filter",
    "07_channel_2_objects": "Channel 2 object filter",
    "07_channel_3_objects": "Channel 3 object filter",
    "07_channel_4_objects": "Channel 4 object filter",
    "09_cell_counting": "Cell counting",
    "10_primary_object_distance_rings": "Distance rings",
    "11_tissue_roi": "Tissue ROI",
}
QC_PANEL_LABELS.update(STAGE_LABELS)
QC_PANEL_LABELS.update(FEATURE_IMAGES)
QC_PANEL_LABELS.update({f"raw_channel_{i}": f"Channel {i}: raw image" for i in range(1, 5)})

TABLE_FILE_OUTPUT_KEYS = {
    "save_excel",
    "save_image_summary_csv",
    "save_primary_objects_csv",
    "save_candidate_qc_csv",
    "save_channel_objects_csv",
    "save_cells_csv",
    "save_ring_metrics_csv",
    "save_animal_summary_csv",
    "save_advanced_metrics_csv",
    "save_object_distances_csv",
}
OUTPUT_TABLE_LABELS = {
    "batch_summary": "Batch Summary",
    "image_summary": "Image Summary",
    "primary_objects": "Neighbour Objects",
    "candidate_qc": "Neighbour Candidate QC",
    "channel_objects": "Channel Objects",
    "cells": "Cells",
    "ring_metrics": "Neighbour Ring Metrics",
    "animal_summary": "Animal Summary",
    "thresholds": "Thresholds (per-image Excel)",
    "advanced_metrics": "Advanced Metrics",
    "object_distances": "Object Distances",
}
OUTPUT_TABLE_LABELS.update({name: title for name, (title, _) in ADVANCED_TABLES.items()})
TABLE_FILE_OUTPUT_KEYS.update(flag for _, flag in ADVANCED_TABLES.values())
BATCH_SUMMARY_COLUMNS = (
    "batch_file_index",
    "source_file",
    "source_name",
    "status",
    "reference_object_count_all",
    "output_dir",
    "error",
)


def _optional_float(variable: tk.StringVar) -> float | None:
    value = variable.get().strip()
    return None if not value else float(value)


def _safe_name(path: Path) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", path.stem).strip(" ._") or "image"


def _preview_image_choices(files: dict[str, Any]) -> dict[str, Path]:
    choices: dict[str, Path] = {}
    for key, value in files.items():
        path = Path(value)
        if path.is_file() and path.suffix.casefold() in {".png", ".tif", ".tiff"}:
            label = "Overview QC" if key == "qc" else key.removeprefix("processing_").replace("_", " ").title()
            choices[label] = path
    return choices


class BrainSectionGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cell Analyzer")
        self.root.geometry("1380x980")
        self.image_paths: list[Path] = []
        self.info = None
        self.active_roles = list(roles_for_count(4))
        ROLE_LABELS.clear()
        ROLE_LABELS.update({role: f"Channel {index}" for index, role in enumerate(self.active_roles, 1)})
        ROLE_BY_LABEL.clear()
        ROLE_BY_LABEL.update({label: role for role, label in ROLE_LABELS.items()})
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.channel_vars: dict[str, dict[str, tk.Variable]] = {}
        self.marker_vars: dict[str, dict[str, tk.Variable]] = {}
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_path: Path | None = None
        self.preview_files: dict[str, Path] = {}
        self.preview_context = None
        self.available_output_columns: dict[str, list[str]] = {}
        self.selected_output_columns: dict[str, list[str]] = {}
        self._build()
        self._apply_config(copy.deepcopy(DEFAULT_CONFIG))
        self.root.bind("<KeyPress>", self._preview_key, add="+")
        self.root.after(100, self._poll_messages)

    @staticmethod
    def _entry_grid(
        parent: ttk.Widget,
        variables: dict[str, tk.Variable],
        fields: list[tuple[str, str, tuple[str, ...] | None]],
        *,
        column: int = 0,
    ) -> dict[str, ttk.Widget]:
        widgets: dict[str, ttk.Widget] = {}
        for row, (label, key, choices) in enumerate(fields):
            variable = variables[key]
            if isinstance(variable, tk.BooleanVar):
                widget = ttk.Checkbutton(parent, text=label, variable=variable)
                widget.grid(
                    row=row, column=column, columnspan=2, sticky="w", padx=5, pady=3
                )
                widgets[key] = widget
                continue
            ttk.Label(parent, text=label, wraplength=210).grid(row=row, column=column, sticky="w", padx=5, pady=3)
            if choices:
                widget = ttk.Combobox(
                    parent, textvariable=variable, values=choices, state="readonly", width=24
                )
            else:
                widget = ttk.Entry(parent, textvariable=variable, width=20)
            widget.grid(row=row, column=column + 1, sticky="ew", padx=5, pady=3)
            widgets[key] = widget
        return widgets

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        files_frame = ttk.LabelFrame(outer, text="Input microscopy images", padding=8)
        files_frame.pack(fill="x")
        self.file_list = tk.Listbox(files_frame, height=6, selectmode="extended")
        self.file_list.grid(row=0, column=0, rowspan=5, sticky="nsew")
        scrollbar = ttk.Scrollbar(files_frame, orient="vertical", command=self.file_list.yview)
        scrollbar.grid(row=0, column=1, rowspan=5, sticky="ns")
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.drop_target_register(DND_FILES)
        self.file_list.dnd_bind("<<Drop>>", self._drop_images)
        ttk.Button(files_frame, text="Add images", command=self._add_images).grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Button(files_frame, text="Inspect selected image", command=self._inspect_selected).grid(row=1, column=2, sticky="ew", padx=6)
        ttk.Button(files_frame, text="Remove selected", command=self._remove_images).grid(row=2, column=2, sticky="ew", padx=6)
        ttk.Button(files_frame, text="Clear", command=self._clear_images).grid(row=3, column=2, sticky="ew", padx=6)
        files_frame.columnconfigure(0, weight=1)

        output_frame = ttk.LabelFrame(outer, text="Output and sample metadata", padding=8)
        output_frame.pack(fill="x", pady=(8, 0))
        self.output_root = tk.StringVar()
        self.cache_root = tk.StringVar()
        self.metadata_csv = tk.StringVar()
        ttk.Label(output_frame, text="Output folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(output_frame, textvariable=self.output_root).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(output_frame, text="Browse", command=self._browse_output).grid(row=0, column=2)
        ttk.Label(output_frame, text="Cache folder").grid(row=1, column=0, sticky="w")
        ttk.Entry(output_frame, textvariable=self.cache_root).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(output_frame, text="Browse", command=self._browse_cache).grid(row=1, column=2)
        ttk.Label(output_frame, text="Sample metadata CSV (optional)").grid(row=2, column=0, sticky="w")
        ttk.Entry(output_frame, textvariable=self.metadata_csv).grid(row=2, column=1, sticky="ew", padx=4)
        ttk.Button(output_frame, text="Browse", command=self._browse_metadata).grid(row=2, column=2)
        output_frame.columnconfigure(1, weight=1)

        self.metadata_label = ttk.Label(
            outer, text="Add images, then inspect one to load channels and pixel size.", wraplength=1320
        )
        self.metadata_label.pack(fill="x", pady=(6, 2))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook
        input_tab = ttk.Frame(notebook, padding=8)
        channels_tab = ttk.Frame(notebook, padding=8)
        plaque_tab = ttk.Frame(notebook, padding=8)
        microglia_tab = ttk.Frame(notebook, padding=8)
        output_tab = ttk.Frame(notebook, padding=8)
        notebook.add(input_tab, text="Image & ROI")
        notebook.add(channels_tab, text="Channels")
        notebook.add(microglia_tab, text="Cell Counting")
        notebook.add(plaque_tab, text="Advanced Analysis")
        notebook.add(output_tab, text="Outputs")
        self.output_tab = output_tab
        self.channels_tab = channels_tab
        self._build_input_tab(input_tab)
        self._build_channels_tab(channels_tab)
        self._build_advanced_tab(plaque_tab)
        self._build_microglia_tab(microglia_tab)
        self._build_output_tab(output_tab)

        controls = ttk.Frame(outer, padding=(0, 6))
        controls.pack(fill="x", pady=(8, 0))
        style = ttk.Style(self.root)
        style.configure(
            "Session.TButton", anchor="center", justify="center",
            padding=(12, 8, 12, 8), width=18,
        )
        style.layout("Session.TButton", [
            ("Button.button", {"sticky": "nswe", "children": [
                ("Button.focus", {"sticky": "nswe", "children": [
                    ("Button.padding", {"sticky": "nswe", "children": [
                        ("Button.label", {"sticky": ""}),
                    ]}),
                ]}),
            ]}),
        ])
        self.session_buttons = []
        for label, command in (
            ("Load parameters", self._load_session),
            ("Save parameters", self._save_parameters),
            ("Save session", self._save_session),
        ):
            button = ttk.Button(controls, text=label, style="Session.TButton", command=command)
            button.pack(side="left", padx=3)
            self.session_buttons.append(button)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status).pack(side="left", padx=12)

        self.log = tk.Text(outer, height=6, wrap="word", state="disabled")
        self.log.pack(fill="x", pady=(6, 0))

    def _build_input_tab(self, parent: ttk.Frame) -> None:
        self.input_vars = {
            "scene": tk.StringVar(value="0"),
            "time_index": tk.StringVar(value="0"),
            "z_projection": tk.StringVar(value="max"),
            "z_index": tk.StringVar(value="0"),
            "zoom": tk.StringVar(value="1.0"),
            "preview_zoom": tk.StringVar(value="0.25"),
            "pixel_size_um_x": tk.StringVar(),
            "pixel_size_um_y": tk.StringVar(),
            "image_width_um": tk.StringVar(),
            "image_height_um": tk.StringVar(),
        }
        image_frame = ttk.LabelFrame(parent, text="Image plane and calibration", padding=8)
        image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._entry_grid(
            image_frame,
            self.input_vars,
            [
                ("Scene", "scene", None),
                ("Time index", "time_index", None),
                ("Z projection", "z_projection", ("single", "max", "mean")),
                ("Z index", "z_index", None),
                ("Analysis zoom", "zoom", None),
                ("Preview zoom", "preview_zoom", None),
                ("X µm/pixel (if metadata is missing)", "pixel_size_um_x", None),
                ("Y µm/pixel (if metadata is missing)", "pixel_size_um_y", None),
                ("Whole-image width µm (optional)", "image_width_um", None),
                ("Whole-image height µm (optional)", "image_height_um", None),
            ],
        )

        self.roi_vars = {
            "mode": tk.StringVar(value="full_image"),
            "mask_directory": tk.StringVar(),
            "mask_suffix": tk.StringVar(value="_mask.png"),
            "invert_mask": tk.BooleanVar(value=False),
        }
        roi_frame = ttk.LabelFrame(parent, text="Analysis ROI", padding=8)
        roi_frame.grid(row=0, column=1, sticky="nsew")
        self._entry_grid(
            roi_frame,
            self.roi_vars,
            [
                ("ROI mode", "mode", ("full_image", "mask_directory")),
                ("Mask directory", "mask_directory", None),
                ("Mask filename suffix", "mask_suffix", None),
                ("Invert mask", "invert_mask", None),
            ],
        )
        ttk.Button(roi_frame, text="Browse mask directory", command=self._browse_mask_directory).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=5, pady=8
        )
        ttk.Label(
            roi_frame,
            text="Microscope pixel-size metadata is preferred; manual calibration is used only when metadata is missing.",
            wraplength=500,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=5)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    @staticmethod
    def _pick_color(variable: tk.StringVar) -> None:
        selected = colorchooser.askcolor(color=variable.get(), title="Select channel color")[1]
        if selected:
            variable.set(selected.upper())

    def _build_channels_tab(self, parent: ttk.Frame) -> None:
        self.channel_vars = {}
        self.marker_vars = {}
        headers = (
            "Channel", "Enabled", "Image channel", "Name", "Wavelength nm", "Color",
            "Gaussian σ", "Threshold method", "Manual value", "Auto scale", "Percentile",
        )
        for column, label in enumerate(headers):
            ttk.Label(parent, text=label).grid(row=0, column=column, sticky="w", padx=3, pady=4)
        for row, role in enumerate(self.active_roles, start=1):
            defaults = default_channel(role, row - 1)
            variables: dict[str, tk.Variable] = {
                "enabled": tk.BooleanVar(value=role != "dapi"),
                "index": tk.StringVar(value=str(row - 1)),
                "alias": tk.StringVar(value=defaults["alias"]),
                "wavelength_nm": tk.StringVar(),
                "color": tk.StringVar(value=defaults["color"]),
                "sigma": tk.StringVar(value="1.0"),
                "method": tk.StringVar(value="otsu"),
                "value": tk.StringVar(value="0"),
                "scale": tk.StringVar(value="1.0"),
                "percentile": tk.StringVar(value="95"),
            }
            self.channel_vars[role] = variables
            ttk.Label(parent, text=ROLE_LABELS[role]).grid(row=row, column=0, sticky="w", padx=3)
            check = ttk.Checkbutton(parent, variable=variables["enabled"])
            check.grid(row=row, column=1)
            variables["index_widget"] = ttk.Combobox(
                parent, textvariable=variables["index"], state="readonly", width=18
            )
            variables["index_widget"].grid(row=row, column=2, padx=3, pady=4)
            ttk.Entry(parent, textvariable=variables["alias"], width=13).grid(row=row, column=3, padx=3)
            ttk.Entry(parent, textvariable=variables["wavelength_nm"], width=10).grid(row=row, column=4, padx=3)
            color_frame = ttk.Frame(parent)
            color_frame.grid(row=row, column=5, padx=3)
            ttk.Entry(color_frame, textvariable=variables["color"], width=8).pack(side="left")
            ttk.Button(color_frame, text="Pick", width=5, command=lambda value=variables["color"]: self._pick_color(value)).pack(side="left", padx=(2, 0))
            ttk.Entry(parent, textvariable=variables["sigma"], width=8).grid(row=row, column=6, padx=3)
            ttk.Combobox(parent, textvariable=variables["method"], values=THRESHOLD_METHODS, state="readonly", width=11).grid(row=row, column=7, padx=3)
            ttk.Entry(parent, textvariable=variables["value"], width=10).grid(row=row, column=8, padx=3)
            ttk.Entry(parent, textvariable=variables["scale"], width=8).grid(row=row, column=9, padx=3)
            ttk.Entry(parent, textvariable=variables["percentile"], width=8).grid(row=row, column=10, padx=3)

        footer_row = len(self.active_roles) + 1
        ttk.Label(parent, text="These channel settings are applied to every selected file.").grid(row=footer_row, column=0, columnspan=11, sticky="w", padx=4, pady=(4, 0))
        filter_group = ttk.LabelFrame(parent, text="Object filter", padding=8)
        filter_group.grid(row=footer_row + 1, column=0, columnspan=11, sticky="nsew", padx=5, pady=12)
        selector = ttk.Frame(filter_group)
        selector.grid(row=0, column=0, sticky="w")
        ttk.Label(selector, text="Channel").pack(side="left")
        self.object_filter_channel = tk.StringVar(value=ROLE_LABELS[self.active_roles[0]])
        object_filter_selector = ttk.Combobox(
            selector,
            textvariable=self.object_filter_channel,
            values=tuple(ROLE_LABELS.values()),
            state="readonly",
            width=16,
        )
        object_filter_selector.pack(side="left", padx=6)
        object_filter_selector.bind("<<ComboboxSelected>>", self._show_object_filter_channel)
        container = ttk.Frame(filter_group)
        container.grid(row=1, column=0, sticky="nsew")
        self.object_filter_frames: dict[str, ttk.Frame] = {}
        for role in self.active_roles:
            defaults = default_channel(role, self.active_roles.index(role))["object_filter"]
            variables = {
                "enabled": tk.BooleanVar(value=defaults["enabled"]),
                "opening_radius_px": tk.StringVar(value=str(defaults["opening_radius_px"])),
                "closing_radius_px": tk.StringVar(value=str(defaults["closing_radius_px"])),
                "fill_holes": tk.BooleanVar(value=defaults["fill_holes"]),
                "max_hole_area_um2": tk.StringVar(value=str(defaults["max_hole_area_um2"])),
                "min_area_um2": tk.StringVar(value=str(defaults["min_area_um2"])),
                "max_area_um2": tk.StringVar(),
                "min_circularity": tk.StringVar(value=str(defaults["min_circularity"])),
                "min_solidity": tk.StringVar(value=str(defaults["min_solidity"])),
                "max_eccentricity": tk.StringVar(value=str(defaults["max_eccentricity"])),
            }
            self.marker_vars[role] = variables
            frame = ttk.Frame(container, padding=(0, 6))
            frame.grid(row=0, column=0, sticky="nsew")
            self.object_filter_frames[role] = frame
            self._entry_grid(
                frame,
                variables,
                [
                    ("Enable object filtering", "enabled", None),
                    ("Open px", "opening_radius_px", None),
                    ("Close px", "closing_radius_px", None),
                    ("Fill holes", "fill_holes", None),
                    ("Maximum filled-hole area µm²", "max_hole_area_um2", None),
                    ("Minimum area µm²", "min_area_um2", None),
                    ("Maximum area µm² (blank = unlimited)", "max_area_um2", None),
                    ("Minimum circularity", "min_circularity", None),
                    ("Minimum solidity", "min_solidity", None),
                    ("Maximum eccentricity", "max_eccentricity", None),
                ],
            )
        self._show_object_filter_channel()

    def _show_object_filter_channel(self, _event: Any = None) -> None:
        role = ROLE_BY_LABEL[self.object_filter_channel.get()]
        self.object_filter_frames[role].tkraise()

    def _set_channel_count(self, count: int) -> None:
        """Rebuild channel-dependent controls from the inspected image metadata."""

        roles = list(roles_for_count(count))
        if not roles:
            raise ValueError("The selected image has no readable channels.")
        channel_state = {
            role: {key: value.get() for key, value in variables.items() if isinstance(value, tk.Variable)}
            for role, variables in self.channel_vars.items()
        }
        marker_state = {
            role: {key: value.get() for key, value in variables.items()}
            for role, variables in self.marker_vars.items()
        }
        self.active_roles = roles
        ROLE_LABELS.clear()
        ROLE_LABELS.update({role: f"Channel {index}" for index, role in enumerate(roles, 1)})
        ROLE_BY_LABEL.clear()
        ROLE_BY_LABEL.update({label: role for role, label in ROLE_LABELS.items()})
        for child in self.channels_tab.winfo_children():
            child.destroy()
        self._build_channels_tab(self.channels_tab)
        for role in roles:
            for key, value in channel_state.get(role, {}).items():
                if key in self.channel_vars[role]:
                    self.channel_vars[role][key].set(value)
            for key, value in marker_state.get(role, {}).items():
                self.marker_vars[role][key].set(value)
        self._refresh_channel_dependent_controls()

    def _refresh_channel_dependent_controls(self) -> None:
        labels = tuple(ROLE_LABELS.values())
        for widget in getattr(self, "pair_selectors", []):
            widget.configure(values=labels)
        if hasattr(self, "radial_reference_widget"):
            self.radial_reference_widget.configure(values=labels)
        for widget in getattr(self, "plaque_channel_widgets", []):
            widget.configure(values=labels)
        for widget in getattr(self, "cell_channel_widgets", []):
            widget.configure(values=("", *labels))
        for variable in (
            getattr(self, "pair_a", None), getattr(self, "pair_b", None),
            getattr(self, "radial_reference_selector", None),
            getattr(self, "plaque_vars", {}).get("reference_channel"),
        ):
            if variable is not None and variable.get() not in labels:
                variable.set(labels[0])
        for key in ("nucleus_channel", "confirmation_channel"):
            variable = getattr(self, "microglia_vars", {}).get(key)
            if variable is not None and variable.get() not in ("", *labels):
                variable.set("")
        self.advanced_pairs = [
            pair for pair in getattr(self, "advanced_pairs", [])
            if len(pair) == 2 and all(role in self.active_roles for role in pair)
        ]
        if hasattr(self, "pair_list"):
            self._refresh_advanced_pairs()
            self._rebuild_advanced_channel_controls()

        if hasattr(self, "boundary_vars"):
            old = self.boundary_vars
            self.boundary_vars = {
                role: old.get(role, {
                    "color": tk.StringVar(value=self.channel_vars[role]["color"].get()),
                    "width_px": tk.StringVar(value="1"),
                })
                for role in self.active_roles
            }
            self.boundary_vars.update({
                name: variables for name, variables in old.items()
                if name not in ROLES and not name.startswith("channel_")
            })
            self.boundary_vars["ring"]["width_px"] = self.output_vars["ring_boundary_width_px"]

            old_processing = getattr(self, "processing_vars", {})
            self.processing_choices = processing_choices(len(self.active_roles))
            self.processing_vars = {
                key: old_processing.get(key, tk.BooleanVar(value=True))
                for key in self.processing_choices
            }
            old_qc = getattr(self, "qc_panel_vars", {})
            choices = qc_panel_choices(len(self.active_roles))
            self.qc_panel_labels = {
                key: QC_PANEL_LABELS.get(key, self.processing_choices.get(key, key.replace("_", " ").title()))
                for key in choices
            }
            self.qc_panel_vars = {
                key: old_qc.get(key, tk.BooleanVar(value=True)) for key in choices
            }
            self._update_qc_panel_status()

    def _build_advanced_tab(self, parent):
        self.advanced_vars = {
            **{key: (tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=str(value)))
               for key, value in FEATURE_DEFAULTS.items() if key not in {"object_channels", "radial_reference_channel"}},
            "neighbour_enabled": tk.BooleanVar(value=False),
            "colocalization_enabled": tk.BooleanVar(value=False),
            "object_distances_enabled": tk.BooleanVar(value=False),
            "scope": tk.StringVar(value="roi"), "proximity_um": tk.StringVar(value="10"),
        }
        self.background_vars = {role: tk.StringVar(value="0") for role in self.active_roles}
        self.object_channel_vars = {role: tk.BooleanVar(value=role in FEATURE_DEFAULTS["object_channels"]) for role in self.active_roles}
        self.radial_reference_selector = tk.StringVar(value="Channel 1")
        self.advanced_pairs = []
        switches = ttk.Frame(parent)
        switches.pack(fill="x", pady=(0, 8))
        for index, (label, key) in enumerate((("Neighbour analysis", "neighbour_enabled"),
                           ("Colocalization", "colocalization_enabled"),
                           ("Object distances (A -> B)", "object_distances_enabled"),
                           ("Per-cell measurements", "cell_measurements_enabled"),
                           ("Radial profiles", "radial_profiles_enabled"),
                           ("Spatial distribution", "spatial_distribution_enabled"),
                           ("Skeleton analysis", "skeleton_enabled"))):
            ttk.Checkbutton(
                switches, text=label, variable=self.advanced_vars[key],
                command=self._update_advanced_visibility,
            ).grid(row=index // 3, column=index % 3, sticky="w", padx=8, pady=3)

        self.advanced_common = ttk.LabelFrame(parent, text="Shared scope", padding=6)
        self._entry_grid(self.advanced_common, self.advanced_vars, [
            ("Analysis scope", "scope", ("roi", "cells", "reference_objects")),
        ])
        self.advanced_empty = ttk.Label(
            parent, text="Select an advanced analysis above to show its parameters."
        )
        self.advanced_notebook = ttk.Notebook(parent)
        neighbour = ttk.Frame(self.advanced_notebook, padding=8)
        relationships = ttk.Frame(self.advanced_notebook, padding=8)
        cell_measurements = ttk.Frame(self.advanced_notebook, padding=8)
        radial = ttk.Frame(self.advanced_notebook, padding=8)
        spatial = ttk.Frame(self.advanced_notebook, padding=8)
        skeleton = ttk.Frame(self.advanced_notebook, padding=8)
        self.advanced_sections = (
            (neighbour, "Neighbour settings", lambda: self.advanced_vars["neighbour_enabled"].get()),
            (relationships, "Channel relationships", lambda: self.advanced_vars["colocalization_enabled"].get() or self.advanced_vars["object_distances_enabled"].get()),
            (cell_measurements, "Per-cell measurements", lambda: self.advanced_vars["cell_measurements_enabled"].get()),
            (radial, "Radial profiles", lambda: self.advanced_vars["radial_profiles_enabled"].get()),
            (spatial, "Spatial distribution", lambda: self.advanced_vars["spatial_distribution_enabled"].get()),
            (skeleton, "Skeleton analysis", lambda: self.advanced_vars["skeleton_enabled"].get()),
        )
        self._build_plaque_tab(neighbour)
        pair_frame = ttk.LabelFrame(relationships, text="Channel pairs (shared by colocalization and distances)", padding=8)
        pair_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        self.pair_a = tk.StringVar(value=ROLE_LABELS[ROLES[0]])
        self.pair_b = tk.StringVar(value=ROLE_LABELS[ROLES[1]])
        self.pair_selectors = []
        for column, variable in enumerate((self.pair_a, self.pair_b)):
            widget = ttk.Combobox(pair_frame, textvariable=variable, values=tuple(ROLE_LABELS.values()), state="readonly", width=16)
            widget.grid(row=0, column=column, padx=4)
            self.pair_selectors.append(widget)
        ttk.Button(pair_frame, text="Add pair", command=self._add_advanced_pair).grid(row=0, column=2, padx=4)
        self.pair_list = tk.Listbox(pair_frame, height=9, selectmode="extended")
        self.pair_list.grid(row=1, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Button(pair_frame, text="Remove selected pairs", command=self._remove_advanced_pairs).grid(row=2, column=0, columnspan=3, sticky="w")
        settings = ttk.Frame(relationships)
        settings.grid(row=0, column=1, sticky="nsew", padx=5)
        self.distance_settings = ttk.LabelFrame(settings, text="Object-distance settings", padding=8)
        self._entry_grid(self.distance_settings, self.advanced_vars, [
            ("Proximity threshold um", "proximity_um", None),
        ])
        self.backgrounds_frame = ttk.LabelFrame(settings, text="Colocalization background subtraction (raw units)", padding=8)
        cell_settings = ttk.LabelFrame(cell_measurements, text="Per-cell measurement regions", padding=10)
        cell_settings.grid(row=0, column=0, sticky="nsew", padx=5)
        self._entry_grid(cell_settings, self.advanced_vars, [("Expansion from counted nucleus um", "cell_expansion_um", None)])
        ttk.Label(cell_settings, wraplength=380, text=(
            "Requires Cell Counting. Measures all enabled channels inside each confirmed nucleus plus its expansion. "
            "Regions are clipped to the selected scope and assigned to the nearest counted nucleus, so cells do not share pixels. "
            "These are nuclear / perinuclear measurement regions, not segmented cell bodies."
        )).grid(row=1, column=0, columnspan=2, sticky="w", pady=12)
        radial_settings = ttk.LabelFrame(radial, text="Radial profiles from object edges", padding=10)
        radial_settings.grid(row=0, column=0, sticky="nsew", padx=5)
        ttk.Label(radial_settings, text="Reference channel").grid(row=0, column=0, sticky="w")
        self.radial_reference_widget = ttk.Combobox(
            radial_settings, textvariable=self.radial_reference_selector,
            values=tuple(ROLE_LABELS.values()), state="readonly", width=20,
        )
        self.radial_reference_widget.grid(row=0, column=1, padx=5)
        radial_entries = ttk.Frame(radial_settings)
        radial_entries.grid(row=1, column=0, columnspan=2, sticky="ew", pady=8)
        self._entry_grid(radial_entries, self.advanced_vars, [
            ("Maximum external distance um", "radial_max_um", None),
            ("External bin width um", "radial_step_um", None),
        ])
        ttk.Label(radial_settings, wraplength=380, text=(
            "Uses this channel's filtered objects, independently of Neighbour Analysis. Bin 0 includes each object's interior. "
            "External bins extend from its edge; pixels are assigned to their nearest reference, preventing double counting. "
            "Exports all channel intensities and positive fractions per object / bin. Curves average object means, not pooled pixels."
        )).grid(row=2, column=0, columnspan=2, sticky="w", pady=12)
        self.spatial_channels_frame = ttk.LabelFrame(spatial, text="Channels", padding=10)
        self.spatial_channels_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        spatial_settings = ttk.LabelFrame(spatial, text="Spatial distribution", padding=10)
        spatial_settings.grid(row=0, column=1, sticky="nsew", padx=5)
        self._entry_grid(spatial_settings, self.advanced_vars, [("Centroid neighbour radius um", "spatial_radius_um", None)])
        ttk.Label(spatial_settings, wraplength=390, text=(
            "Measures same-channel nearest centroid distances, neighbours within the radius (excluding self), and object density. "
            "ROI-edge truncation is flagged; no edge correction, clustering significance or Ripley's K test is claimed. "
            "Fewer than two objects gives NaN nearest-neighbour distances."
        )).grid(row=1, column=0, columnspan=2, sticky="w", pady=12)
        self.skeleton_channels_frame = ttk.LabelFrame(skeleton, text="Channels", padding=10)
        self.skeleton_channels_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        ttk.Label(skeleton, wraplength=850, text=(
            "Skeleton Analysis uses each filtered object independently. It reports calibrated 8-neighbour pixel-graph length, "
            "endpoints, isolated pixels and clusters of junction pixels. Preview colors identify skeleton, endpoints and junction pixels. "
            "Touching cells can form one object; skeleton statistics are not automatic cell-type or activation classifications."
        )).grid(row=0, column=1, sticky="w", pady=16)
        ttk.Label(relationships, wraplength=1050, text=(
            "PCC and Manders M1/M2 use raw background-corrected intensities in the selected ROI. "
            "Overlap and distances use each channel's filtered masks, not neighbour-specific exclusions. "
            "Distances: calibrated centroid and minimum pixel-center mask distance (overlap = 0). "
            "All results are 2-D; projection overlap does not prove internalization. "
            "No Costes automatic threshold or significance test is performed."
        )).grid(row=1, column=0, columnspan=2, sticky="w", pady=12)
        self._rebuild_advanced_channel_controls()
        self._update_advanced_visibility()

    def _update_advanced_visibility(self):
        enabled = any(
            variable.get() for key, variable in self.advanced_vars.items()
            if key.endswith("_enabled")
        )
        if enabled:
            self.advanced_empty.pack_forget()
            self.advanced_common.pack(fill="x", pady=(0, 6))
            self.advanced_notebook.pack(fill="both", expand=True)
        else:
            self.advanced_common.pack_forget()
            self.advanced_notebook.pack_forget()
            self.advanced_empty.pack(anchor="w", padx=8, pady=12)
        existing = set(self.advanced_notebook.tabs())
        for frame, title, predicate in self.advanced_sections:
            if str(frame) in existing:
                self.advanced_notebook.forget(frame)
            if predicate():
                self.advanced_notebook.add(frame, text=title)
        if hasattr(self, "backgrounds_frame"):
            if self.advanced_vars["colocalization_enabled"].get():
                self.backgrounds_frame.pack(fill="x", pady=(0, 8))
            else:
                self.backgrounds_frame.pack_forget()
            if self.advanced_vars["object_distances_enabled"].get():
                self.distance_settings.pack(fill="x", pady=(0, 8))
            else:
                self.distance_settings.pack_forget()

    def _rebuild_advanced_channel_controls(self):
        if not hasattr(self, "backgrounds_frame"):
            return
        backgrounds = {role: variable.get() for role, variable in self.background_vars.items()}
        selected = {role: variable.get() for role, variable in self.object_channel_vars.items()}
        self.background_vars = {
            role: tk.StringVar(value=backgrounds.get(role, "0")) for role in self.active_roles
        }
        self.object_channel_vars = {
            role: tk.BooleanVar(value=selected.get(role, role in FEATURE_DEFAULTS["object_channels"]))
            for role in self.active_roles
        }
        for frame in (self.backgrounds_frame, self.spatial_channels_frame, self.skeleton_channels_frame):
            for child in frame.winfo_children():
                child.destroy()
        self._entry_grid(
            self.backgrounds_frame, self.background_vars,
            [(ROLE_LABELS[role], role, None) for role in self.active_roles],
        )
        for frame in (self.spatial_channels_frame, self.skeleton_channels_frame):
            for row, role in enumerate(self.active_roles):
                ttk.Checkbutton(frame, text=ROLE_LABELS[role], variable=self.object_channel_vars[role]).grid(
                    row=row, column=0, sticky="w", pady=4
                )

    def _refresh_advanced_pairs(self):
        self.pair_list.delete(0, tk.END)
        for a, b in self.advanced_pairs:
            self.pair_list.insert(tk.END, f"{ROLE_LABELS[a]} -> {ROLE_LABELS[b]}")

    def _add_advanced_pair(self):
        pair = [ROLE_BY_LABEL[self.pair_a.get()], ROLE_BY_LABEL[self.pair_b.get()]]
        if pair[0] == pair[1]:
            messagebox.showerror("Invalid pair", "Select two different channels.")
        elif pair not in self.advanced_pairs:
            self.advanced_pairs.append(pair)
            self._refresh_advanced_pairs()

    def _remove_advanced_pairs(self):
        for index in reversed(self.pair_list.curselection()):
            self.advanced_pairs.pop(index)
        self._refresh_advanced_pairs()

    def _advanced_config(self):
        return {
            **{key: variable.get() for key, variable in self.advanced_vars.items()},
            "proximity_um": float(self.advanced_vars["proximity_um"].get()),
            **{key: float(self.advanced_vars[key].get()) for key in
               ("cell_expansion_um", "radial_max_um", "radial_step_um", "spatial_radius_um")},
            "radial_reference_channel": ROLE_BY_LABEL[self.radial_reference_selector.get()],
            "object_channels": [role for role, variable in self.object_channel_vars.items() if variable.get()],
            "pairs": copy.deepcopy(self.advanced_pairs),
            "backgrounds": {role: float(value.get()) for role, value in self.background_vars.items()},
        }

    def _build_plaque_tab(self, parent: ttk.Frame) -> None:
        self.plaque_vars = {
            "reference_channel": tk.StringVar(value=ROLE_LABELS[ROLES[0]]),
            "opening_radius_px": tk.StringVar(value="0"),
            "closing_radius_px": tk.StringVar(value="1"),
            "fill_holes": tk.BooleanVar(value=False),
            "max_hole_area_um2": tk.StringVar(value="0"),
            "min_area_um2": tk.StringVar(value="10"),
            "max_area_um2": tk.StringVar(),
            "min_circularity": tk.StringVar(value="0"),
            "min_solidity": tk.StringVar(value="0"),
            "max_eccentricity": tk.StringVar(value="1"),
            "neuron_exclusion_mode": tk.StringVar(value="shape_and_dark_center"),
            "neuron_detection_threshold_scale": tk.StringVar(value="0.75"),
            "neuron_min_diameter_um": tk.StringVar(value="8"),
            "neuron_max_diameter_um": tk.StringVar(value="28"),
            "neuron_min_circularity": tk.StringVar(value="0.35"),
            "neuron_min_solidity": tk.StringVar(value="0.60"),
            "neuron_min_hole_fraction": tk.StringVar(value="0.03"),
            "neuron_max_center_shell_ratio": tk.StringVar(value="0.95"),
            "require_nearby_microglia": tk.BooleanVar(value=False),
            "nearby_microglia_radius_um": tk.StringVar(value="30"),
            "min_nearby_microglia_count": tk.StringVar(value="1"),
            "split_touching": tk.BooleanVar(value=False),
            "min_peak_distance_px": tk.StringVar(value="8"),
            "watershed_min_peak_height_px": tk.StringVar(value="0"),
            "watershed_compactness": tk.StringVar(value="0"),
            "exclude_boundary_plaques_from_table": tk.BooleanVar(value=True),
            "boundary_margin_um": tk.StringVar(value="0"),
            "ring_edges_um": tk.StringVar(),
        }
        frames = [
            ("Reference-object morphology", [
                ("Reference object channel", "reference_channel", tuple(ROLE_LABELS.values())),
                ("Open px", "opening_radius_px", None),
                ("Close px", "closing_radius_px", None),
                ("Fill holes", "fill_holes", None),
                ("Maximum filled-hole area µm²", "max_hole_area_um2", None),
                ("Minimum area µm²", "min_area_um2", None),
                ("Maximum area µm² (blank = unlimited)", "max_area_um2", None),
                ("Minimum circularity", "min_circularity", None),
                ("Minimum solidity", "min_solidity", None),
                ("Maximum eccentricity", "max_eccentricity", None),
            ]),
            ("Round/hollow-object exclusion", [
                ("Exclusion mode", "neuron_exclusion_mode", ("off", "shape", "shape_and_dark_center")),
                ("Detection threshold scale", "neuron_detection_threshold_scale", None),
                ("Minimum diameter µm", "neuron_min_diameter_um", None),
                ("Maximum diameter µm", "neuron_max_diameter_um", None),
                ("Minimum circularity", "neuron_min_circularity", None),
                ("Minimum solidity", "neuron_min_solidity", None),
                ("Minimum hole fraction", "neuron_min_hole_fraction", None),
                ("Maximum center/shell intensity ratio", "neuron_max_center_shell_ratio", None),
            ]),
            ("Cell gating, splitting, and distance rings", [
                ("Require nearby cells", "require_nearby_microglia", None),
                ("Nearby-cell radius µm", "nearby_microglia_radius_um", None),
                ("Minimum nearby-cell count", "min_nearby_microglia_count", None),
                ("Split touching reference objects", "split_touching", None),
                ("Minimum peak distance px", "min_peak_distance_px", None),
                ("Minimum peak height px", "watershed_min_peak_height_px", None),
                ("Watershed compactness", "watershed_compactness", None),
                ("Exclude boundary reference objects from table", "exclude_boundary_plaques_from_table", None),
                ("Boundary margin µm", "boundary_margin_um", None),
                ("Cumulative ranges µm (e.g. 0,15,30)", "ring_edges_um", None),
            ]),
        ]
        self.plaque_channel_widgets = []
        for column, (title, fields) in enumerate(frames):
            frame = ttk.LabelFrame(parent, text=title, padding=8)
            frame.grid(row=0, column=column, sticky="nsew", padx=5)
            widgets = self._entry_grid(frame, self.plaque_vars, fields)
            if "reference_channel" in widgets:
                self.plaque_channel_widgets.append(widgets["reference_channel"])
            parent.columnconfigure(column, weight=1)
        ttk.Label(
            parent,
            text=(
                "Neighbour analysis uses the selected reference-object channel for "
                "per-object measurements, distance rings, and nearby-cell counts."
            ),
            wraplength=1200,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 0))

    def _build_microglia_tab(self, parent: ttk.Frame) -> None:
        self.microglia_vars = {
            "enabled": tk.BooleanVar(value=False),
            "nucleus_channel": tk.StringVar(),
            "confirmation_channel": tk.StringVar(),
            "opening_radius_px": tk.StringVar(value="0"),
            "closing_radius_px": tk.StringVar(value="1"),
            "fill_holes": tk.BooleanVar(value=True),
            "min_nucleus_area_um2": tk.StringVar(value="10"),
            "max_nucleus_area_um2": tk.StringVar(value="150"),
            "min_circularity": tk.StringVar(value="0.2"),
            "min_solidity": tk.StringVar(value="0.7"),
            "max_eccentricity": tk.StringVar(value="0.98"),
            "split_touching": tk.BooleanVar(value=True),
            "min_peak_distance_px": tk.StringVar(value="3"),
            "perinuclear_radius_um": tk.StringVar(value="3"),
            "min_confirmation_positive_fraction": tk.StringVar(value="0.15"),
        }
        frame = ttk.LabelFrame(parent, text="Cell detection", padding=8)
        frame.grid(row=0, column=0, sticky="nsew")
        widgets = self._entry_grid(
            frame,
            self.microglia_vars,
            [
                ("Enable cell counting", "enabled", None),
                ("Nucleus channel", "nucleus_channel", ("", *ROLE_LABELS.values())),
                ("Confirmation channel (optional)", "confirmation_channel", ("", *ROLE_LABELS.values())),
                ("Nucleus open px", "opening_radius_px", None),
                ("Nucleus close px", "closing_radius_px", None),
                ("Fill nucleus holes", "fill_holes", None),
                ("Minimum nucleus area µm²", "min_nucleus_area_um2", None),
                ("Maximum nucleus area µm²", "max_nucleus_area_um2", None),
                ("Minimum nucleus circularity", "min_circularity", None),
                ("Minimum nucleus solidity", "min_solidity", None),
                ("Maximum nucleus eccentricity", "max_eccentricity", None),
                ("Split touching nuclei", "split_touching", None),
                ("Minimum nucleus peak distance px", "min_peak_distance_px", None),
                ("Perinuclear radius µm", "perinuclear_radius_um", None),
                ("Minimum confirmation-positive fraction", "min_confirmation_positive_fraction", None),
            ],
        )
        self.cell_channel_widgets = [widgets["nucleus_channel"], widgets["confirmation_channel"]]
        ttk.Label(
            parent,
            text="Choose any enabled nucleus channel. Leave confirmation blank to count by nucleus morphology only.",
            wraplength=700,
        ).grid(row=1, column=0, sticky="w", padx=5, pady=10)

    def _build_output_tab(self, parent: ttk.Frame) -> None:
        self.output_vars = {
            key: tk.BooleanVar(value=value)
            for key, value in OUTPUT_SELECTION_DEFAULTS.items()
        }
        self.output_vars.update(
            {
                "ring_boundary_width_px": tk.StringVar(value="1"),
                "preview_max_dimension_px": tk.StringVar(value="2200"),
                "continue_on_error": tk.BooleanVar(value=True),
                "open_output": tk.BooleanVar(value=True),
            }
        )
        self.qc_panel_labels = {
            key: QC_PANEL_LABELS.get(key, PROCESSING_CHOICES.get(key, key.replace("_", " ").title()))
            for key in qc_panel_choices(len(self.active_roles))
        }
        self.qc_panel_vars = {
            key: tk.BooleanVar(value=key in QC_PANEL_DEFAULTS) for key in self.qc_panel_labels
        }

        self.processing_choices = processing_choices(len(self.active_roles))
        self.processing_vars = {key: tk.BooleanVar(value=True) for key in self.processing_choices}
        self.drawing_vars = {
            key: (tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=str(value)))
            for key, value in DRAWING_DEFAULTS.items() if key != "boundaries"
        }
        self.boundary_vars = {
            key: {"color": tk.StringVar(value=value["color"]), "width_px": tk.StringVar(value=str(value["width_px"]))}
            for key, value in DRAWING_DEFAULTS["boundaries"].items()
        }
        self.boundary_vars["ring"]["width_px"] = self.output_vars["ring_boundary_width_px"]
        table_frame = ttk.LabelFrame(parent, text="Table columns", padding=8)
        table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        ttk.Label(
            table_frame,
            text="Run a preview to load the exact parameters produced by the current settings.",
            wraplength=250,
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.choose_columns_button = ttk.Button(
            table_frame,
            text="Choose output parameters...",
            command=self._choose_output_columns,
            state="disabled",
        )
        self.choose_columns_button.grid(row=1, column=0, sticky="w")
        self.output_column_status = tk.StringVar(
            value="All table parameters will be exported until a preview is run."
        )
        ttk.Label(
            table_frame,
            textvariable=self.output_column_status,
            wraplength=250,
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        image_frame = ttk.LabelFrame(parent, text="Images and masks", padding=8)
        image_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        image_outputs = (
            ("Overview QC image", "save_qc"),
            ("Raw channel preview images", "save_raw_channel_images"),
            ("Composite image", "save_composite_image"),
            ("Segmentation overlay images", "save_segmentation_images"),
            ("Mask preview images", "save_mask_images"),
            ("Intermediate pipeline images", "save_stage_images"),
            ("Advanced analysis images", "save_advanced_images"),
            ("Tissue ROI mask TIFF", "save_tissue_mask"),
            ("Neighbour-reference label TIFF", "save_primary_object_labels"),
            ("Excluded-object mask TIFFs", "save_excluded_object_masks"),
            ("Channel-object label TIFF", "save_channel_object_labels"),
            ("Nucleus / cell label TIFFs", "save_cell_labels"),
            ("Positive-mask TIFF", "save_positive_masks"),
        )
        for row, (label, key) in enumerate(image_outputs):
            ttk.Checkbutton(image_frame, text=label, variable=self.output_vars[key]).grid(
                row=row, column=0, sticky="w", pady=2
            )
        ttk.Label(
            image_frame,
            text="PNG image choices also control generated previews.",
            wraplength=230,
        ).grid(row=len(image_outputs), column=0, sticky="w", pady=(8, 0))
        qc_menu_button = ttk.Button(image_frame, text="Overview QC panels...", command=self._choose_qc_panels)
        qc_menu_button.grid(row=len(image_outputs) + 1, column=0, sticky="w", pady=(8, 0))
        self.qc_panel_status = tk.StringVar()
        ttk.Label(image_frame, textvariable=self.qc_panel_status).grid(
            row=len(image_outputs) + 2, column=0, sticky="w", pady=(4, 0)
        )
        self._update_qc_panel_status()
        ttk.Button(image_frame, text="Choose processing images...", command=self._choose_processing_images).grid(row=len(image_outputs) + 3, column=0, sticky="w", pady=6)

        preview_frame = ttk.LabelFrame(parent, text="Processed preview", padding=8)
        preview_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        self.preview_label = tk.Label(
            preview_frame,
            text="Select an image, then run Preview selected image.",
            background="#111111",
            foreground="white",
            anchor="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_label.bind("<Button-1>", lambda _event: self.preview_label.focus_set())
        preview_toolbar = ttk.Frame(preview_frame)
        preview_toolbar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(preview_toolbar, text="Displayed result").pack(side="left")
        self.preview_choice = tk.StringVar()
        self.preview_selector = ttk.Combobox(
            preview_toolbar,
            textvariable=self.preview_choice,
            state="disabled",
            width=32,
        )
        self.preview_selector.pack(side="left", padx=5)
        self.preview_selector.bind("<<ComboboxSelected>>", self._show_selected_preview)
        self.preview_selector.bind("<KeyPress>", self._preview_key, add="+")
        self.open_preview_button = ttk.Button(
            preview_toolbar,
            text="Open image",
            command=self._open_preview,
            state="disabled",
        )
        self.open_preview_button.pack(side="left")
        ttk.Label(preview_frame, text="A: previous preview | D: next preview").grid(
            row=3, column=0, sticky="w", pady=(6, 0)
        )
        display_settings = ttk.Frame(preview_frame)
        display_settings.grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(display_settings, text="Ring width px").pack(side="left")
        ttk.Entry(
            display_settings, textvariable=self.output_vars["ring_boundary_width_px"], width=7
        ).pack(side="left", padx=(3, 12))
        ttk.Label(display_settings, text="Max image dimension px").pack(side="left")
        ttk.Entry(
            display_settings, textvariable=self.output_vars["preview_max_dimension_px"], width=8
        ).pack(side="left", padx=3)
        ttk.Button(display_settings, text="Plot settings...", command=self._plot_settings).pack(side="left", padx=8)
        ttk.Button(preview_frame, text="Apply display settings to preview", command=self._redraw_preview).grid(row=4, column=0, sticky="w", pady=6)
        ttk.Label(preview_frame, text="Display only; rerun Preview after changing segmentation or analysis.").grid(row=5, column=0, sticky="w")
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        run_frame = ttk.LabelFrame(parent, text="Run", padding=8)
        run_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.preview_button = ttk.Button(
            run_frame, text="Preview selected image", command=self._preview
        )
        self.preview_button.grid(row=0, column=0, padx=(0, 5))
        self.run_button = ttk.Button(run_frame, text="Run all images", command=self._run)
        self.run_button.grid(row=0, column=1, padx=5)
        ttk.Checkbutton(
            run_frame,
            text="Continue after a file fails",
            variable=self.output_vars["continue_on_error"],
        ).grid(row=0, column=2, padx=12)
        ttk.Checkbutton(
            run_frame,
            text="Open output folder when complete",
            variable=self.output_vars["open_output"],
        ).grid(row=0, column=3, padx=12)
        ttk.Label(
            run_frame,
            text="Run parameters and selected file paths are always saved.",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(7, 0))

        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(2, weight=1)

    def _drawing_config(self):
        result = {key: variable.get() for key, variable in self.drawing_vars.items()}
        for key in ("low_percentile", "high_percentile", "gamma", "gain", "opacity"):
            result[key] = float(result[key])
        for key in ("dpi", "font_size", "histogram_bins"):
            result[key] = int(result[key])
        result["boundaries"] = {
            name: {"color": variables["color"].get(), "width_px": int(variables["width_px"].get())}
            for name, variables in self.boundary_vars.items()
        }
        return result

    def _choose_processing_images(self):
        self._choose_image_selection(self.processing_vars, self.processing_choices, "Processing images to save and preview")

    def _choose_qc_panels(self):
        self._choose_image_selection(self.qc_panel_vars, self.qc_panel_labels, "Overview QC panels")

    def _choose_image_selection(self, variables, choices, title):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("650x620")
        window.transient(self.root)
        window.grab_set()
        ttk.Label(window, text="Select the images to include. Image-category switches still apply.").pack(pady=8)
        frame = ttk.Frame(window)
        frame.pack(fill="both", expand=True, padx=12)
        listing = tk.Listbox(frame, selectmode="multiple", exportselection=False)
        listing.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(frame, command=listing.yview)
        scroll.pack(side="right", fill="y")
        listing.configure(yscrollcommand=scroll.set)
        keys = list(choices)
        for index, key in enumerate(keys):
            listing.insert(tk.END, choices[key])
            if variables[key].get():
                listing.selection_set(index)
        buttons = ttk.Frame(window, padding=10)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="All", command=lambda: listing.selection_set(0, tk.END)).pack(side="left", padx=4)
        ttk.Button(buttons, text="None", command=lambda: listing.selection_clear(0, tk.END)).pack(side="left", padx=4)
        def apply():
            selection = set(listing.curselection())
            for index, key in enumerate(keys):
                variables[key].set(index in selection)
            self._update_qc_panel_status()
            window.destroy()
        ttk.Button(buttons, text="Apply", command=apply).pack(side="right", padx=4)

    def _plot_settings(self):
        window = tk.Toplevel(self.root)
        window.title("Plot settings (display only)")
        window.geometry("960x650")
        window.transient(self.root)
        left = ttk.LabelFrame(window, text="Image and figure display", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self._entry_grid(left, self.drawing_vars, [
            ("Low display percentile", "low_percentile", None),
            ("High display percentile", "high_percentile", None),
            ("Gamma", "gamma", None), ("Brightness gain", "gain", None),
            ("Boundary / overlap opacity", "opacity", None),
            ("Show object IDs", "show_object_ids", None),
            ("Label font size px", "font_size", None), ("Figure DPI", "dpi", None),
            ("Figure background #RRGGBB", "background", None),
            ("Heatmap color map", "heatmap_cmap", ("viridis", "magma", "inferno", "plasma", "cividis", "gray", "turbo")),
            ("Histogram bins", "histogram_bins", None),
            ("Log-scale density colors", "density_log_scale", None),
        ])
        right = ttk.LabelFrame(window, text="Boundary style", padding=8)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        labels = {name: ROLE_LABELS.get(name, name.replace("_", " ").title()) for name in self.boundary_vars}
        selected = tk.StringVar(value=labels["ring"])
        selector = ttk.Combobox(right, textvariable=selected, values=tuple(labels.values()), state="readonly", width=24)
        selector.grid(row=0, column=0, columnspan=2, pady=8)
        frames = {}
        for name, variables in self.boundary_vars.items():
            frame = ttk.Frame(right)
            frame.grid(row=1, column=0, columnspan=2)
            self._entry_grid(frame, variables, [("Line color #RRGGBB", "color", None), ("Line width (rendered px)", "width_px", None)])
            ttk.Button(frame, text="Pick color", command=lambda v=variables["color"]: self._pick_color(v)).grid(row=2, column=1, pady=8)
            frames[labels[name]] = frame
        selector.bind("<<ComboboxSelected>>", lambda _: frames[selected.get()].tkraise())
        frames[selected.get()].tkraise()
        ttk.Label(right, text="Channel image colors are set in Channels.\nLine widths are measured in the saved PNG,\nnot in the full-resolution source image.", wraplength=300).grid(row=2, column=0, columnspan=2, sticky="w", pady=14)
        ttk.Button(window, text="Apply display settings to preview", command=self._redraw_preview).grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Button(window, text="Close", command=window.destroy).grid(row=1, column=1, padx=8, pady=8, sticky="e")
        ttk.Label(window, text="Display settings do not alter thresholds, masks or measurement values.").grid(row=2, column=0, columnspan=2, padx=8, sticky="w")

    def _redraw_preview(self):
        if self.run_button.instate(["disabled"]):
            return
        if not self.preview_context:
            messagebox.showinfo("Preview required", "Run Preview selected image first.")
            return
        try:
            context = self.preview_context
            config = copy.deepcopy(context["config"])
            config["output"]["drawing"] = self._drawing_config()
            config["output"]["preview_max_dimension_px"] = int(self.output_vars["preview_max_dimension_px"].get())
            config["output"]["processing_steps"] = [key for key, value in self.processing_vars.items() if value.get()]
            config["output"]["qc_panels"] = [key for key, value in self.qc_panel_vars.items() if value.get()]
            for key in PREVIEW_IMAGE_OUTPUT_KEYS:
                config["output"][key] = bool(self.output_vars[key].get())
            # Read colors/names only; cached segmentation and analysis stay unchanged.
            for role in self.active_roles:
                config["channels"][role]["color"] = self.channel_vars[role]["color"].get()
            config = normalize_config(config, context["info"])
            self._set_busy(True)
            self.status.set("Rendering cached preview...")
        except Exception as error:
            messagebox.showerror("Invalid plot settings", str(error))
            return
        def worker():
            try:
                output = config["output"]
                directory = Path(config["input"]["output_dir"])
                files = save_qc(
                    directory / "cell_analysis_qc.png" if output["save_qc"] else None,
                    context["images"], context["products"], output["preview_max_dimension_px"],
                    _channel_colors(context["info"], config, set(context["images"])),
                    {role: config["channels"][role]["alias"] for role in context["images"]},
                    config["microglia_count"], config["plaque"]["reference_channel"] if context["products"].neighbour_enabled else next(iter(context["images"])), output["qc_panels"], directory / "processing_images",
                    save_raw_channels=output["save_raw_channel_images"], save_composite=output["save_composite_image"],
                    save_segmentation=output["save_segmentation_images"], save_mask_images=output["save_mask_images"],
                    save_stages=output["save_stage_images"], save_advanced=output["save_advanced_images"],
                    drawing=output["drawing"], processing_steps=output["processing_steps"],
                )
                self.messages.put(("redraw_complete", files))
            except Exception as error:
                self.messages.put(("error", error))
        threading.Thread(target=worker, daemon=True).start()

    def _update_qc_panel_status(self) -> None:
        selected = sum(variable.get() for variable in self.qc_panel_vars.values())
        self.qc_panel_status.set(f"{selected} of {len(self.qc_panel_vars)} panels selected")

    def _update_output_column_status(self) -> None:
        if not self.available_output_columns:
            self.output_column_status.set(
                "All table parameters will be exported until a preview is run."
            )
            return
        total = sum(len(columns) for columns in self.available_output_columns.values())
        selected = sum(
            len(self.selected_output_columns.get(name, columns))
            for name, columns in self.available_output_columns.items()
        )
        self.output_column_status.set(
            f"{selected} of {total} parameters selected across "
            f"{len(self.available_output_columns)} tables."
        )

    def _set_available_output_columns(self, tables: dict[str, Any]) -> None:
        available = {
            name: [str(column) for column in table.columns]
            for name, table in tables.items()
            if name in OUTPUT_TABLE_LABELS and hasattr(table, "columns")
        }
        available["batch_summary"] = list(BATCH_SUMMARY_COLUMNS)
        image_columns = available.get("image_summary", [])
        animal_columns = [
            "genotype",
            "mouse_id",
            "region",
            "sex",
            "image_count",
            "roi_area_um2",
            "roi_area_mm2",
            "reference_object_count_all",
            "reference_object_count_boundary",
            "reference_object_count_interior",
            "reference_object_density_all_per_mm2",
        ]
        animal_columns.extend(
            column
            for column in image_columns
            if column.endswith("_area_um2")
            or "fraction" in column
            or column.endswith("_per_mm2")
        )
        animal_columns.extend(
            (
                "median_interior_reference_object_area_um2",
                "mean_interior_reference_object_area_um2",
            )
        )
        available["animal_summary"] = list(dict.fromkeys(animal_columns))
        self.available_output_columns = {
            name: available[name]
            for name in OUTPUT_TABLE_LABELS
            if name in available
        }
        self.choose_columns_button.configure(state="normal")
        self._update_output_column_status()

    def _choose_output_columns(self) -> None:
        if not self.available_output_columns:
            messagebox.showinfo(
                "Output parameters",
                "Run Preview selected image first to load the available parameters.",
            )
            return

        window = tk.Toplevel(self.root)
        window.title("Choose output parameters")
        window.geometry("840x560")
        window.transient(self.root)
        window.grab_set()
        body = ttk.Frame(window, padding=10)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Table").grid(row=0, column=0, sticky="w")
        ttk.Label(body, text="Parameters to include").grid(row=0, column=1, sticky="w")
        table_list = tk.Listbox(body, exportselection=False, width=30)
        table_list.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        column_list = tk.Listbox(body, selectmode="extended", exportselection=False)
        column_list.grid(row=1, column=1, sticky="nsew")
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=column_list.yview)
        scrollbar.grid(row=1, column=2, sticky="ns")
        column_list.configure(yscrollcommand=scrollbar.set)
        table_names = list(self.available_output_columns)
        for name in table_names:
            table_list.insert(tk.END, OUTPUT_TABLE_LABELS[name])
        draft = {
            name: list(self.selected_output_columns.get(name, columns))
            for name, columns in self.available_output_columns.items()
        }
        current: dict[str, str | None] = {"name": None}

        def store_current() -> None:
            name = current["name"]
            if name is not None:
                draft[name] = [column_list.get(index) for index in column_list.curselection()]

        def show_table(_event: Any = None) -> None:
            store_current()
            selection = table_list.curselection()
            if not selection:
                return
            name = table_names[selection[0]]
            current["name"] = name
            column_list.delete(0, tk.END)
            selected = set(draft[name])
            for index, column in enumerate(self.available_output_columns[name]):
                column_list.insert(tk.END, column)
                if column in selected:
                    column_list.selection_set(index)

        def select_all() -> None:
            column_list.selection_set(0, tk.END)

        def clear_current() -> None:
            column_list.selection_clear(0, tk.END)

        def apply_selection() -> None:
            store_current()
            empty = [OUTPUT_TABLE_LABELS[name] for name in table_names if not draft[name]]
            if empty:
                messagebox.showerror(
                    "Output parameters",
                    "Select at least one parameter for: " + ", ".join(empty),
                    parent=window,
                )
                return
            self.selected_output_columns = {
                name: columns
                for name, columns in draft.items()
                if columns != self.available_output_columns[name]
            }
            self._update_output_column_status()
            window.destroy()

        table_list.bind("<<ListboxSelect>>", show_table)
        buttons = ttk.Frame(body)
        buttons.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Button(buttons, text="Select all in current table", command=select_all).pack(side="left")
        ttk.Button(buttons, text="Clear current table", command=clear_current).pack(side="left", padx=6)
        ttk.Button(buttons, text="Apply", command=apply_selection).pack(side="right")
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right", padx=6)
        body.rowconfigure(1, weight=1)
        body.columnconfigure(1, weight=1)
        table_list.selection_set(0)
        show_table()

    def _selected_path(self) -> Path:
        if not self.image_paths:
            raise ValueError("Add at least one image first.")
        selected = self.file_list.curselection()
        return self.image_paths[selected[0] if selected else 0]

    def _add_image_paths(self, values: tuple[str, ...] | list[str]) -> None:
        existing = {str(path).casefold() for path in self.image_paths}
        for value in values:
            path = Path(value).expanduser().resolve()
            key = str(path).casefold()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES and key not in existing:
                self.image_paths.append(path)
                existing.add(key)
        self._refresh_files()
        if self.image_paths and self.info is None:
            self.file_list.selection_clear(0, tk.END)
            self.file_list.selection_set(0)

    def _drop_images(self, event: Any) -> str:
        self._add_image_paths(list(self.root.tk.splitlist(event.data)))
        return "break"

    def _add_images(self) -> None:
        values = filedialog.askopenfilenames(
            title="Select microscopy images",
            filetypes=[
                ("All supported microscopy images", MICROSCOPY_FILE_PATTERN),
                ("Zeiss CZI", "*.czi"),
                ("Leica", "*.lif *.lei *.scn *.lof *.xlef"),
                ("Olympus", "*.oir *.vsi *.oib *.oif"),
                ("OME/TIFF", "*.ome.tif *.ome.tiff *.tif *.tiff"),
                ("Nikon ND2", "*.nd2"),
                ("All files", "*.*"),
            ],
        )
        self._add_image_paths(list(values))

    def _remove_images(self) -> None:
        selected = set(self.file_list.curselection())
        self.image_paths = [path for index, path in enumerate(self.image_paths) if index not in selected]
        self.info = None
        self._refresh_files()

    def _clear_images(self) -> None:
        self.image_paths.clear()
        self.info = None
        self._refresh_files()

    def _refresh_files(self) -> None:
        self.file_list.delete(0, tk.END)
        for path in self.image_paths:
            self.file_list.insert(tk.END, str(path))
        self.status.set(f"{len(self.image_paths)} image(s) selected")

    def _inspect_selected(self) -> None:
        try:
            source = self._selected_path()
            first_inspection = self.info is None
            cache_value = self.cache_root.get().strip()
            if cache_value:
                configure_cache_directory(cache_value)
            self.info = inspect_image(
                source,
                pixel_size_um_x=_optional_float(self.input_vars["pixel_size_um_x"]),
                pixel_size_um_y=_optional_float(self.input_vars["pixel_size_um_y"]),
                image_width_um=_optional_float(self.input_vars["image_width_um"]),
                image_height_um=_optional_float(self.input_vars["image_height_um"]),
            )
            self._set_channel_count(len(self.info.channels))
            choices = [f"{channel.index}: {channel.name}" for channel in self.info.channels]
            available = {channel.index for channel in self.info.channels}
            for position, role in enumerate(self.active_roles):
                variables = self.channel_vars[role]
                if first_inspection:
                    variables["enabled"].set(True)
                variables["index_widget"].configure(values=choices)
                try:
                    current = int(str(variables["index"].get()).split(":", 1)[0])
                except ValueError:
                    current = -1
                index = current if current in available else self.info.channels[position].index
                match = next((choice for choice in choices if choice.startswith(f"{index}:")), choices[0])
                variables["index"].set(match)
            if not self.input_vars["pixel_size_um_x"].get() and self.info.pixel_size_um_x:
                self.input_vars["pixel_size_um_x"].set(str(self.info.pixel_size_um_x))
            if not self.input_vars["pixel_size_um_y"].get() and self.info.pixel_size_um_y:
                self.input_vars["pixel_size_um_y"].set(str(self.info.pixel_size_um_y))
            dims = self.info.dimensions
            self.metadata_label.configure(
                text=(
                    f"{source.name} | {dims.get('X', (0, 0))[1]} × {dims.get('Y', (0, 0))[1]} px | "
                    f"Channels: {', '.join(choices)} | Pixel size: {self.info.pixel_size_um_x} × "
                    f"{self.info.pixel_size_um_y} µm"
                )
            )
            self.status.set("Image metadata loaded")
        except Exception as error:
            messagebox.showerror("Could not read image", str(error))

    def _browse_output(self) -> None:
        value = filedialog.askdirectory(title="Select output folder")
        if value:
            self.output_root.set(value)

    def _browse_cache(self) -> None:
        value = filedialog.askdirectory(title="Select application cache folder")
        if value:
            self.cache_root.set(value)

    def _browse_metadata(self) -> None:
        value = filedialog.askopenfilename(title="Select sample metadata CSV", filetypes=[("CSV", "*.csv")])
        if value:
            self.metadata_csv.set(value)

    def _browse_mask_directory(self) -> None:
        value = filedialog.askdirectory(title="Select tissue-mask directory")
        if value:
            self.roi_vars["mask_directory"].set(value)

    def _build_config(
        self,
        source: Path | None = None,
        *,
        require_output: bool = False,
        require_cache: bool = False,
    ) -> dict[str, Any]:
        source = source or self._selected_path()
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["channels"] = {
            role: default_channel(role, position)
            for position, role in enumerate(self.active_roles)
        }
        config["advanced"] = self._advanced_config()
        output_value = self.output_root.get().strip()
        cache_value = self.cache_root.get().strip()
        if require_output and not output_value:
            raise ValueError("Select an output folder.")
        if require_cache and not cache_value:
            raise ValueError("Select a cache folder.")
        output_root = (
            Path(output_value).expanduser().resolve()
            if output_value
            else source.with_name(f"{source.stem}_cell_analysis")
        )
        config["application"]["cache_directory"] = (
            str(configure_cache_directory(cache_value)) if cache_value else None
        )
        config["input"].update(
            {
                "image_path": str(source),
                "output_dir": str(output_root / _safe_name(source)),
                "scene": int(self.input_vars["scene"].get()),
                "time_index": int(self.input_vars["time_index"].get()),
                "z_projection": self.input_vars["z_projection"].get(),
                "z_index": int(self.input_vars["z_index"].get()),
                "zoom": float(self.input_vars["zoom"].get()),
                "pixel_size_um_x": _optional_float(self.input_vars["pixel_size_um_x"]),
                "pixel_size_um_y": _optional_float(self.input_vars["pixel_size_um_y"]),
                "image_width_um": _optional_float(self.input_vars["image_width_um"]),
                "image_height_um": _optional_float(self.input_vars["image_height_um"]),
            }
        )
        for role in self.active_roles:
            variables = self.channel_vars[role]
            config["channels"][role].update(
                {
                    "enabled": bool(variables["enabled"].get()),
                    "index": int(str(variables["index"].get()).split(":", 1)[0]),
                    "alias": variables["alias"].get().strip(),
                    "wavelength_nm": _optional_float(variables["wavelength_nm"]),
                    "color": variables["color"].get().strip(),
                    "gaussian_sigma_px": float(variables["sigma"].get()),
                    "threshold": {
                        "method": variables["method"].get(),
                        "value": float(variables["value"].get()),
                        "scale": float(variables["scale"].get()),
                        "percentile": float(variables["percentile"].get()),
                    },
                }
            )
        for role in self.active_roles:
            variables = self.marker_vars[role]
            config["channels"][role]["object_filter"].update(
                {
                    "enabled": bool(variables["enabled"].get()),
                    "opening_radius_px": int(variables["opening_radius_px"].get()),
                    "closing_radius_px": int(variables["closing_radius_px"].get()),
                    "fill_holes": bool(variables["fill_holes"].get()),
                    "max_hole_area_um2": float(variables["max_hole_area_um2"].get()),
                    "min_area_um2": float(variables["min_area_um2"].get()),
                    "max_area_um2": _optional_float(variables["max_area_um2"]),
                    "min_circularity": float(variables["min_circularity"].get()),
                    "min_solidity": float(variables["min_solidity"].get()),
                    "max_eccentricity": float(variables["max_eccentricity"].get()),
                }
            )
        config["tissue_roi"].update(
            {
                "mode": self.roi_vars["mode"].get(),
                "mask_directory": self.roi_vars["mask_directory"].get().strip() or None,
                "mask_suffix": self.roi_vars["mask_suffix"].get(),
                "invert_mask": bool(self.roi_vars["invert_mask"].get()),
            }
        )
        plaque = self.plaque_vars
        config["plaque"].update(
            {
                "reference_channel": ROLE_BY_LABEL.get(plaque["reference_channel"].get()),
                "opening_radius_px": int(plaque["opening_radius_px"].get()),
                "closing_radius_px": int(plaque["closing_radius_px"].get()),
                "fill_holes": bool(plaque["fill_holes"].get()),
                "max_hole_area_um2": float(plaque["max_hole_area_um2"].get()),
                "min_area_um2": float(plaque["min_area_um2"].get()),
                "max_area_um2": _optional_float(plaque["max_area_um2"]),
                "min_circularity": float(plaque["min_circularity"].get()),
                "min_solidity": float(plaque["min_solidity"].get()),
                "max_eccentricity": float(plaque["max_eccentricity"].get()),
                "neuron_exclusion_mode": plaque["neuron_exclusion_mode"].get(),
                "neuron_detection_threshold_scale": float(plaque["neuron_detection_threshold_scale"].get()),
                "neuron_min_diameter_um": float(plaque["neuron_min_diameter_um"].get()),
                "neuron_max_diameter_um": float(plaque["neuron_max_diameter_um"].get()),
                "neuron_min_circularity": float(plaque["neuron_min_circularity"].get()),
                "neuron_min_solidity": float(plaque["neuron_min_solidity"].get()),
                "neuron_min_hole_fraction": float(plaque["neuron_min_hole_fraction"].get()),
                "neuron_max_center_shell_ratio": float(plaque["neuron_max_center_shell_ratio"].get()),
                "require_nearby_microglia": bool(plaque["require_nearby_microglia"].get()),
                "nearby_microglia_radius_um": float(plaque["nearby_microglia_radius_um"].get()),
                "min_nearby_microglia_count": int(plaque["min_nearby_microglia_count"].get()),
                "split_touching": bool(plaque["split_touching"].get()),
                "min_peak_distance_px": int(plaque["min_peak_distance_px"].get()),
                "watershed_min_peak_height_px": float(plaque["watershed_min_peak_height_px"].get()),
                "watershed_compactness": float(plaque["watershed_compactness"].get()),
                "exclude_boundary_plaques_from_table": bool(plaque["exclude_boundary_plaques_from_table"].get()),
                "boundary_margin_um": float(plaque["boundary_margin_um"].get()),
            }
        )
        config["spatial"]["ring_edges_um"] = [
            float(value.strip()) for value in plaque["ring_edges_um"].get().split(",") if value.strip()
        ]
        microglia = self.microglia_vars
        config["microglia_count"].update(
            {
                "enabled": bool(microglia["enabled"].get()),
                "nucleus_channel": ROLE_BY_LABEL.get(microglia["nucleus_channel"].get()),
                "confirmation_channel": ROLE_BY_LABEL.get(microglia["confirmation_channel"].get()),
                "opening_radius_px": int(microglia["opening_radius_px"].get()),
                "closing_radius_px": int(microglia["closing_radius_px"].get()),
                "fill_holes": bool(microglia["fill_holes"].get()),
                "min_nucleus_area_um2": float(microglia["min_nucleus_area_um2"].get()),
                "max_nucleus_area_um2": _optional_float(microglia["max_nucleus_area_um2"]),
                "min_circularity": float(microglia["min_circularity"].get()),
                "min_solidity": float(microglia["min_solidity"].get()),
                "max_eccentricity": float(microglia["max_eccentricity"].get()),
                "split_touching": bool(microglia["split_touching"].get()),
                "min_peak_distance_px": int(microglia["min_peak_distance_px"].get()),
                "perinuclear_radius_um": float(microglia["perinuclear_radius_um"].get()),
                "min_confirmation_positive_fraction": float(
                    microglia["min_confirmation_positive_fraction"].get()
                ),
            }
        )
        config["batch"].update(
            {
                "metadata_csv": self.metadata_csv.get().strip() or None,
                "continue_on_error": bool(self.output_vars["continue_on_error"].get()),
            }
        )
        config["output"].update(
            {
                **{
                    key: (
                        True
                        if key in TABLE_FILE_OUTPUT_KEYS
                        else bool(self.output_vars[key].get())
                    )
                    for key in OUTPUT_SELECTION_DEFAULTS
                },
                "table_columns": copy.deepcopy(self.selected_output_columns),
                "drawing": self._drawing_config(),
                "processing_steps": (None if all(v.get() for v in self.processing_vars.values()) else [key for key, value in self.processing_vars.items() if value.get()]),
                "qc_panels": [
                    key for key, variable in self.qc_panel_vars.items() if variable.get()
                ],
                "ring_boundary_width_px": int(
                    self.output_vars["ring_boundary_width_px"].get()
                ),
                "preview_max_dimension_px": int(
                    self.output_vars["preview_max_dimension_px"].get()
                ),
            }
        )
        info = inspect_image(
            source,
            pixel_size_um_x=config["input"]["pixel_size_um_x"],
            pixel_size_um_y=config["input"]["pixel_size_um_y"],
            image_width_um=config["input"]["image_width_um"],
            image_height_um=config["input"]["image_height_um"],
        )
        return normalize_config(config, info)

    def _apply_config(self, config: dict[str, Any]) -> None:
        channel_count = len(config.get("channels", {}))
        if channel_count and channel_count != len(self.active_roles):
            self._set_channel_count(channel_count)
        advanced = config.get("advanced", {})
        for key, variable in self.advanced_vars.items():
            default = DEFAULT_CONFIG["advanced"][key]
            if key == "neighbour_enabled" and "advanced" not in config and "plaque" in config:
                default = True
            variable.set(advanced.get(key, default))
        self.advanced_pairs = copy.deepcopy(advanced.get("pairs", []))
        radial_role = advanced.get("radial_reference_channel", self.active_roles[0])
        self.radial_reference_selector.set(ROLE_LABELS.get(radial_role, ROLE_LABELS[self.active_roles[0]]))
        for role, variable in self.object_channel_vars.items():
            variable.set(role in advanced.get("object_channels", FEATURE_DEFAULTS["object_channels"]))
        self._refresh_advanced_pairs()
        for role, variable in self.background_vars.items():
            variable.set(advanced.get("backgrounds", {}).get(role, 0))
        application = config.get("application", {})
        if application.get("cache_directory"):
            self.cache_root.set(str(application["cache_directory"]))
        input_config = config.get("input", {})
        for key, default in (
            ("scene", 0), ("time_index", 0), ("z_projection", "max"), ("z_index", 0),
            ("zoom", 1.0), ("pixel_size_um_x", None), ("pixel_size_um_y", None),
            ("image_width_um", None), ("image_height_um", None),
        ):
            value = input_config.get(key, default)
            self.input_vars[key].set("" if value is None else str(value))
        for position, role in enumerate(self.active_roles):
            item = config.get("channels", {}).get(role, {})
            variables = self.channel_vars[role]
            variables["enabled"].set(bool(item.get("enabled", True)))
            variables["index"].set(str(item.get("index", position)))
            variables["alias"].set(str(item.get("alias", ROLE_LABELS[role])))
            wavelength = item.get("wavelength_nm")
            variables["wavelength_nm"].set("" if wavelength is None else str(wavelength))
            variables["color"].set(str(item.get("color", default_channel(role, position)["color"])))
            variables["sigma"].set(str(item.get("gaussian_sigma_px", 1.0)))
            threshold = item.get("threshold", {})
            for key, default in (("method", "otsu"), ("value", 0), ("scale", 1), ("percentile", 95)):
                variables[key].set(str(threshold.get(key, default)))
        for position, role in enumerate(self.active_roles):
            item = config.get("channels", {}).get(role, {}).get("object_filter", {})
            for key, variable in self.marker_vars[role].items():
                value = item.get(key, default_channel(role, position)["object_filter"].get(key))
                variable.set("" if value is None else value)
        roi = config.get("tissue_roi", {})
        for key, variable in self.roi_vars.items():
            value = roi.get(key, DEFAULT_CONFIG["tissue_roi"].get(key))
            variable.set("" if value is None else value)
        plaque = config.get("plaque", {})
        for key, variable in self.plaque_vars.items():
            if key == "ring_edges_um":
                value = config.get("spatial", {}).get("ring_edges_um", [])
                variable.set(",".join(str(item) for item in value))
            elif key == "reference_channel":
                variable.set(ROLE_LABELS.get(plaque.get(key, "abeta"), ROLE_LABELS["abeta"]))
            else:
                value = plaque.get(key, DEFAULT_CONFIG["plaque"].get(key))
                variable.set("" if value is None else value)
        microglia = config.get("microglia_count", {})
        for key, variable in self.microglia_vars.items():
            value = microglia.get(key, DEFAULT_CONFIG["microglia_count"].get(key))
            if key in {"nucleus_channel", "confirmation_channel"}:
                variable.set(ROLE_LABELS.get(value, ""))
            else:
                variable.set("" if value is None else value)
        output = config.get("output", {})
        for key, default in OUTPUT_SELECTION_DEFAULTS.items():
            self.output_vars[key].set(output.get(key, default))
        for key in ("ring_boundary_width_px", "preview_max_dimension_px"):
            self.output_vars[key].set(output.get(key, DEFAULT_CONFIG["output"][key]))
        drawing = output.get("drawing", {})
        for key, variable in self.drawing_vars.items():
            variable.set(drawing.get(key, DRAWING_DEFAULTS[key]))
        for name, variables in self.boundary_vars.items():
            for key, variable in variables.items():
                default = DRAWING_DEFAULTS["boundaries"][name][key]
                if name == "ring" and key == "width_px":
                    default = output.get("ring_boundary_width_px", 1)
                variable.set(drawing.get("boundaries", {}).get(name, {}).get(key, default))
        steps = output.get("processing_steps")
        for key, variable in self.processing_vars.items():
            variable.set(steps is None or key in steps)
        selected_qc_panels = set(output.get("qc_panels", QC_PANEL_DEFAULTS))
        for key, variable in self.qc_panel_vars.items():
            variable.set(key in selected_qc_panels)
        self._update_qc_panel_status()
        self.selected_output_columns = copy.deepcopy(output.get("table_columns", {}))
        self._update_output_column_status()
        self.output_vars["continue_on_error"].set(bool(config.get("batch", {}).get("continue_on_error", True)))
        self.metadata_csv.set(str(config.get("batch", {}).get("metadata_csv") or ""))
        self._update_advanced_visibility()

    def _load_session(self) -> None:
        value = filedialog.askopenfilename(
            title="Load parameters or session", filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if not value:
            return
        try:
            document = load_config(value)
            config = copy.deepcopy(document.get("template_config", document))
            raw_paths = document.get("selected_image_files") or [config.get("input", {}).get("image_path")]
            paths = []
            for raw in raw_paths:
                if not raw:
                    continue
                path = Path(raw).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Image from session does not exist: {path}")
                paths.append(path)
            if paths:
                self.image_paths = paths
                self._refresh_files()
                self.file_list.selection_set(0)
            if document.get("output_root"):
                self.output_root.set(str(document["output_root"]))
            self._apply_config(config)
            if paths:
                self._inspect_selected()
            self.status.set(f"Loaded: {value}")
        except Exception as error:
            messagebox.showerror("Load failed", str(error))

    def _save_parameters(self) -> None:
        try:
            config = self._build_config()
            value = filedialog.asksaveasfilename(
                title="Save parameters", initialfile="cell_analysis_config.yaml", defaultextension=".yaml",
                filetypes=[("YAML", "*.yaml")],
            )
            if value:
                save_config(config, value)
                self.status.set(f"Parameters saved: {value}")
        except Exception as error:
            messagebox.showerror("Save failed", str(error))

    def _save_session(self) -> None:
        try:
            config = self._build_config()
            output_value = self.output_root.get().strip()
            value = filedialog.asksaveasfilename(
                title="Save full session", initialfile="cell_analysis_session.yaml", defaultextension=".yaml",
                filetypes=[("YAML", "*.yaml")],
            )
            if not value:
                return
            document = {
                "format_version": 2,
                "selected_image_files": [str(path) for path in self.image_paths],
                "output_root": (
                    str(Path(output_value).expanduser().resolve()) if output_value else None
                ),
                "template_config": config,
                "per_file_overrides": {},
            }
            save_config(document, value)
            files_path = Path(value).with_suffix(".files.txt")
            files_path.write_text("\n".join(document["selected_image_files"]) + "\n", encoding="utf-8")
            self.status.set(f"Session and file list saved: {value}")
        except Exception as error:
            messagebox.showerror("Save failed", str(error))

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.preview_button.configure(state=state)
        self.run_button.configure(state=state)

    def _preview(self) -> None:
        try:
            source = self._selected_path()
            config = self._build_config(source, require_cache=True)
            config["input"]["zoom"] = float(self.input_vars["preview_zoom"].get())
            config["input"]["output_dir"] = str(Path(self.cache_root.get()).expanduser().resolve() / "preview" / _safe_name(source))
            for key in OUTPUT_SELECTION_DEFAULTS:
                config["output"][key] = False
            for key in PREVIEW_IMAGE_OUTPUT_KEYS:
                config["output"][key] = bool(self.output_vars[key].get())
            if not any(config["output"][key] for key in PREVIEW_IMAGE_OUTPUT_KEYS):
                raise ValueError("Select at least one preview image under Images and masks.")
            if config["tissue_roi"]["mode"] == "mask_directory":
                _resolve_mask(config, source)
            self._set_busy(True)
            self.preview_context = None
            self.status.set("Generating preview...")
        except Exception as error:
            messagebox.showerror("Invalid preview parameters", str(error))
            return

        def worker() -> None:
            try:
                result = run_analysis(
                    config,
                    progress=lambda message: self.messages.put(("progress", message)),
                    include_tables=True,
                )
                self.messages.put(("preview_complete", result))
            except Exception as error:
                self.messages.put(("error", error))

        threading.Thread(target=worker, daemon=True).start()

    def _run(self) -> None:
        try:
            config = self._build_config(require_output=True, require_cache=True)
            output_root = self.output_root.get().strip()
            self._set_busy(True)
            self.status.set("Analyzing all images...")
        except Exception as error:
            messagebox.showerror("Invalid parameters", str(error))
            return

        def worker() -> None:
            try:
                result = run_batch_analysis(
                    self.image_paths,
                    config,
                    output_root,
                    progress=lambda message: self.messages.put(("progress", message)),
                )
                self.messages.put(("complete", result))
            except Exception as error:
                self.messages.put(("error", error))

        threading.Thread(target=worker, daemon=True).start()

    def _show_preview(self, path: str | Path) -> None:
        preview_path = Path(path)
        with Image.open(preview_path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((680, 500), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_path = preview_path
        self.preview_label.configure(image=self.preview_photo, text="")
        self.open_preview_button.configure(state="normal")
        self.notebook.select(self.output_tab)

    def _set_preview_files(self, files: dict[str, Any]) -> None:
        self.preview_files = _preview_image_choices(files)
        choices = tuple(self.preview_files)
        self.preview_selector.configure(
            values=choices, state="readonly" if choices else "disabled"
        )
        if choices:
            self.preview_choice.set(self.preview_choice.get() if self.preview_choice.get() in choices else choices[0])
            self._show_selected_preview()
        else:
            self.preview_choice.set("")
            self.preview_label.configure(image="", text="No preview image was generated.")
            self.open_preview_button.configure(state="disabled")

    def _show_selected_preview(self, _event: Any = None) -> None:
        path = self.preview_files.get(self.preview_choice.get())
        if path is not None:
            self._show_preview(path)

    def _preview_key(self, event: Any) -> str | None:
        key = event.keysym.lower()
        if key not in {"a", "d"} or event.state & (0x4 | 0x8 | 0x20000):
            return None
        if self.notebook.select() != str(self.output_tab):
            return None
        if event.widget is not self.preview_selector and event.widget.winfo_class() in {
            "Entry", "TEntry", "Text", "Spinbox", "TSpinbox", "TCombobox",
        }:
            return None
        choices = tuple(self.preview_files)
        if not choices:
            return None
        current = self.preview_choice.get()
        index = choices.index(current) if current in choices else 0
        self.preview_choice.set(choices[(index + (1 if key == "d" else -1)) % len(choices)])
        self._show_selected_preview()
        return "break"

    def _open_preview(self) -> None:
        if self.preview_path and self.preview_path.is_file():
            os.startfile(self.preview_path)

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "progress":
                    self.status.set(str(value))
                    self._append_log(str(value))
                elif kind == "preview_complete":
                    self._set_busy(False)
                    self.status.set("Preview complete")
                    self.preview_context = value.pop("_render_context", None)
                    self._set_available_output_columns(value.pop("_tables", {}))
                    try:
                        self._set_preview_files(value.get("files", {}))
                    except Exception as error:
                        messagebox.showerror("Could not display preview", str(error))
                elif kind == "redraw_complete":
                    self._set_busy(False)
                    self.status.set("Display settings applied (measurements unchanged)")
                    self._set_preview_files(value)
                elif kind == "complete":
                    self._set_busy(False)
                    self.status.set("Batch analysis complete")
                    messagebox.showinfo(
                        "Complete",
                        f"Completed: {value['completed']}\nFailed: {value['failed']}\n\nResults: {value['output_root']}",
                    )
                    if self.output_vars["open_output"].get():
                        os.startfile(value["output_root"])
                elif kind == "error":
                    self._set_busy(False)
                    self.status.set("Analysis failed")
                    messagebox.showerror("Analysis failed", str(value))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)


def main() -> None:
    root = TkinterDnD.Tk()
    BrainSectionGui(root)
    root.mainloop()
