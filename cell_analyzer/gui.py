"""Tkinter desktop interface for channel-specific CZI analysis settings."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .config import create_default_config, save_yaml
from .czi_io import inspect_czi
from .pipeline import run_analysis


def _optional_float(value: str) -> float | None:
    stripped = value.strip()
    return None if not stripped else float(stripped)


class AnalyzerGui:
    """Small desktop editor for input, segmentation, and per-channel settings."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CZI Cell Analyzer")
        self.root.geometry("1180x760")
        self.info = None
        self.channel_variables: dict[int, dict[str, tk.Variable]] = {}
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._build_layout()
        self.root.after(100, self._poll_messages)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        input_frame = ttk.LabelFrame(outer, text="Input and image plane", padding=8)
        input_frame.pack(fill="x")
        self.czi_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.scene = tk.StringVar(value="0")
        self.time_index = tk.StringVar(value="0")
        self.z_projection = tk.StringVar(value="max")
        self.z_index = tk.StringVar(value="0")
        self.zoom = tk.StringVar(value="1.0")
        self.segmentation_channel = tk.StringVar(value="0")

        ttk.Label(input_frame, text="CZI file").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.czi_path).grid(
            row=0, column=1, columnspan=7, sticky="ew", padx=4
        )
        ttk.Button(input_frame, text="Browse", command=self._browse_czi).grid(row=0, column=8)
        ttk.Button(input_frame, text="Inspect", command=self._inspect).grid(
            row=0, column=9, padx=(4, 0)
        )
        ttk.Label(input_frame, text="Output directory").grid(row=1, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.output_dir).grid(
            row=1, column=1, columnspan=7, sticky="ew", padx=4
        )
        ttk.Button(input_frame, text="Browse", command=self._browse_output).grid(row=1, column=8)

        compact_fields = [
            ("Scene", self.scene),
            ("Time", self.time_index),
            ("Z projection", self.z_projection),
            ("Z index", self.z_index),
            ("Zoom", self.zoom),
        ]
        for column, (label, variable) in enumerate(compact_fields):
            ttk.Label(input_frame, text=label).grid(row=2, column=column * 2, sticky="w")
            if label == "Z projection":
                widget = ttk.Combobox(
                    input_frame,
                    textvariable=variable,
                    values=("single", "max", "mean"),
                    state="readonly",
                    width=10,
                )
            else:
                widget = ttk.Entry(input_frame, textvariable=variable, width=10)
            widget.grid(row=2, column=column * 2 + 1, sticky="w", padx=(3, 12))
        input_frame.columnconfigure(1, weight=1)

        self.metadata_label = ttk.Label(outer, text="Load a CZI file to discover its channels.")
        self.metadata_label.pack(fill="x", pady=(6, 4))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)
        channel_tab = ttk.Frame(notebook, padding=6)
        segmentation_tab = ttk.Frame(notebook, padding=10)
        notebook.add(channel_tab, text="Channel parameters")
        notebook.add(segmentation_tab, text="Segmentation parameters")

        self.channel_canvas = tk.Canvas(channel_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            channel_tab, orient="vertical", command=self.channel_canvas.yview
        )
        self.channel_table = ttk.Frame(self.channel_canvas)
        self.channel_table.bind(
            "<Configure>",
            lambda event: self.channel_canvas.configure(
                scrollregion=self.channel_canvas.bbox("all")
            ),
        )
        self.channel_canvas.create_window((0, 0), window=self.channel_table, anchor="nw")
        self.channel_canvas.configure(yscrollcommand=scrollbar.set)
        self.channel_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.segmentation_variables: dict[str, tk.Variable] = {
            "threshold_method": tk.StringVar(value="otsu"),
            "threshold_scale": tk.StringVar(value="0.8"),
            "threshold_percentile": tk.StringVar(value="90"),
            "adaptive_block_size_px": tk.StringVar(value="51"),
            "adaptive_offset": tk.StringVar(value="0"),
            "opening_radius_px": tk.StringVar(value="1"),
            "closing_radius_px": tk.StringVar(value="2"),
            "fill_all_holes": tk.BooleanVar(value=True),
            "min_hole_area_px": tk.StringVar(value="32"),
            "min_area_px": tk.StringVar(value="50"),
            "max_area_px": tk.StringVar(value=""),
            "min_circularity": tk.StringVar(value="0.05"),
            "min_local_contrast_ratio": tk.StringVar(value="1.0"),
            "local_contrast_ring_px": tk.StringVar(value="4"),
            "clear_border": tk.BooleanVar(value=True),
            "border_exclusion_margin_px": tk.StringVar(value="10"),
            "split_touching": tk.BooleanVar(value=False),
            "min_peak_distance_px": tk.StringVar(value="8"),
            "watershed_min_peak_height_px": tk.StringVar(value="0"),
            "watershed_min_peak_prominence_px": tk.StringVar(value="0"),
            "watershed_compactness": tk.StringVar(value="0"),
            "invert": tk.BooleanVar(value=False),
        }
        self._build_segmentation_tab(segmentation_tab)

        action_frame = ttk.Frame(outer)
        action_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(action_frame, text="Segmentation channel").pack(side="left")
        self.segmentation_channel_box = ttk.Combobox(
            action_frame,
            textvariable=self.segmentation_channel,
            state="readonly",
            width=24,
        )
        self.segmentation_channel_box.pack(side="left", padx=5)
        ttk.Button(action_frame, text="Save configuration", command=self._save_config).pack(
            side="left", padx=4
        )
        self.run_button = ttk.Button(action_frame, text="Run analysis", command=self._run)
        self.run_button.pack(side="left", padx=4)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(action_frame, textvariable=self.status).pack(side="left", padx=12)

    def _build_segmentation_tab(self, parent: ttk.Frame) -> None:
        descriptions = [
            ("Threshold method", "threshold_method", ("otsu", "yen", "triangle", "percentile", "adaptive")),
            ("Automatic threshold multiplier", "threshold_scale", None),
            ("Threshold percentile", "threshold_percentile", None),
            ("Adaptive block size", "adaptive_block_size_px", None),
            ("Adaptive offset", "adaptive_offset", None),
            ("Opening radius (px)", "opening_radius_px", None),
            ("Closing radius (px)", "closing_radius_px", None),
            ("Maximum filled hole (px)", "min_hole_area_px", None),
            ("Minimum ROI area (px)", "min_area_px", None),
            ("Maximum ROI area (blank = none)", "max_area_px", None),
            ("Minimum circularity", "min_circularity", None),
            ("Minimum local contrast ratio", "min_local_contrast_ratio", None),
            ("Local contrast ring (px)", "local_contrast_ring_px", None),
            ("Minimum watershed peak distance (px)", "min_peak_distance_px", None),
            ("Minimum watershed peak height (px)", "watershed_min_peak_height_px", None),
            ("Watershed peak prominence (px)", "watershed_min_peak_prominence_px", None),
            ("Watershed compactness", "watershed_compactness", None),
            ("Border exclusion margin (px)", "border_exclusion_margin_px", None),
        ]
        for row, (label, key, choices) in enumerate(descriptions):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
            if choices:
                widget = ttk.Combobox(
                    parent,
                    textvariable=self.segmentation_variables[key],
                    values=choices,
                    state="readonly",
                    width=18,
                )
            else:
                widget = ttk.Entry(
                    parent, textvariable=self.segmentation_variables[key], width=20
                )
            widget.grid(row=row, column=1, sticky="w", padx=8)
        checkbox_row = len(descriptions)
        ttk.Checkbutton(
            parent,
            text="Fill all enclosed holes",
            variable=self.segmentation_variables["fill_all_holes"],
        ).grid(row=checkbox_row, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(
            parent,
            text="Split touching cells with watershed",
            variable=self.segmentation_variables["split_touching"],
        ).grid(row=checkbox_row + 1, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(
            parent,
            text="Discard ROIs touching the image border",
            variable=self.segmentation_variables["clear_border"],
        ).grid(row=checkbox_row + 2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(
            parent,
            text="Invert foreground and background",
            variable=self.segmentation_variables["invert"],
        ).grid(row=checkbox_row + 3, column=0, columnspan=2, sticky="w", pady=4)

    def _browse_czi(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a CZI file", filetypes=[("CZI image", "*.czi"), ("All files", "*.*")]
        )
        if path:
            self.czi_path.set(path)
            self._inspect()

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir.set(path)

    def _inspect(self) -> None:
        try:
            self.info = inspect_czi(self.czi_path.get())
            if not self.output_dir.get():
                source = Path(self.info.path)
                self.output_dir.set(str(source.with_name(f"{source.stem}_cell_analysis")))
            self._populate_channels()
            channel_text = ", ".join(
                f"C{channel.index}: {channel.name}" for channel in self.info.channels
            )
            dimensions = self.info.dimensions
            self.metadata_label.configure(
                text=(
                    f"Size: {dimensions.get('X', (0, 0))[1]} x {dimensions.get('Y', (0, 0))[1]} | "
                    f"Channels: {channel_text} | Scenes: {len(self.info.scenes)}"
                )
            )
            self.status.set("CZI metadata loaded")
        except Exception as error:
            messagebox.showerror("CZI inspection failed", str(error))

    def _populate_channels(self) -> None:
        for child in self.channel_table.winfo_children():
            child.destroy()
        self.channel_variables.clear()
        headers = [
            "Index",
            "CZI name",
            "Output alias",
            "Gaussian sigma",
            "Area threshold",
            "Threshold percentile",
        ]
        for column, text in enumerate(headers):
            ttk.Label(self.channel_table, text=text).grid(
                row=0, column=column, sticky="w", padx=3, pady=3
            )
        channel_choices: list[str] = []
        for row, channel in enumerate(self.info.channels, start=1):
            variables: dict[str, tk.Variable] = {
                "alias": tk.StringVar(value=channel.name),
                "gaussian_sigma_px": tk.StringVar(value="1"),
                "threshold_method": tk.StringVar(value="otsu"),
                "threshold_percentile": tk.StringVar(value="95"),
            }
            self.channel_variables[channel.index] = variables
            ttk.Label(self.channel_table, text=str(channel.index)).grid(row=row, column=0, padx=3)
            ttk.Label(self.channel_table, text=channel.name).grid(row=row, column=1, padx=3)
            entry_keys = [
                "alias",
                "gaussian_sigma_px",
            ]
            for column, key in enumerate(entry_keys, start=2):
                ttk.Entry(self.channel_table, textvariable=variables[key], width=13).grid(
                    row=row, column=column, padx=3, pady=2
                )
            ttk.Combobox(
                self.channel_table,
                textvariable=variables["threshold_method"],
                values=("none", "otsu", "yen", "triangle", "percentile"),
                state="readonly",
                width=12,
            ).grid(row=row, column=4, padx=3)
            ttk.Entry(
                self.channel_table, textvariable=variables["threshold_percentile"], width=13
            ).grid(row=row, column=5, padx=3)
            channel_choices.append(f"{channel.index}: {channel.name}")
        self.segmentation_channel_box.configure(values=channel_choices)
        if channel_choices:
            self.segmentation_channel.set(channel_choices[0])

    def _build_config(self) -> dict[str, Any]:
        current_path = Path(self.czi_path.get()).resolve()
        if self.info is None or Path(self.info.path) != current_path:
            self._inspect()
        if self.info is None or Path(self.info.path) != current_path:
            raise ValueError("Load a valid CZI file first.")
        config = create_default_config(self.info, self.output_dir.get(), zoom=float(self.zoom.get()))
        selected_channel = int(self.segmentation_channel.get().split(":", 1)[0])
        config["input"].update(
            {
                "scene": int(self.scene.get()),
                "time_index": int(self.time_index.get()),
                "z_projection": self.z_projection.get(),
                "z_index": int(self.z_index.get()),
                "zoom": float(self.zoom.get()),
                "segmentation_channel": selected_channel,
            }
        )
        for channel_index, variables in self.channel_variables.items():
            item = config["channels"][str(channel_index)]
            item.update(
                {
                    "alias": str(variables["alias"].get()),
                    "gaussian_sigma_px": float(variables["gaussian_sigma_px"].get()),
                    "measurement_threshold": {
                        "method": str(variables["threshold_method"].get()),
                        "percentile": float(variables["threshold_percentile"].get()),
                    },
                }
            )
        string_float_keys = {
            "threshold_percentile",
            "adaptive_offset",
            "min_circularity",
            "min_local_contrast_ratio",
            "watershed_compactness",
            "watershed_min_peak_height_px",
            "watershed_min_peak_prominence_px",
            "threshold_scale",
        }
        string_int_keys = {
            "adaptive_block_size_px",
            "opening_radius_px",
            "closing_radius_px",
            "min_hole_area_px",
            "min_area_px",
            "local_contrast_ring_px",
            "min_peak_distance_px",
            "border_exclusion_margin_px",
        }
        for key, variable in self.segmentation_variables.items():
            value = variable.get()
            if key == "max_area_px":
                config["segmentation"][key] = _optional_float(str(value))
            elif key in string_float_keys:
                config["segmentation"][key] = float(value)
            elif key in string_int_keys:
                config["segmentation"][key] = int(value)
            else:
                config["segmentation"][key] = value
        return config

    def _save_config(self) -> None:
        try:
            config = self._build_config()
            path = filedialog.asksaveasfilename(
                title="Save analysis configuration",
                defaultextension=".yaml",
                filetypes=[("YAML configuration", "*.yaml"), ("All files", "*.*")],
            )
            if path:
                save_yaml(config, path)
                self.status.set(f"Configuration saved: {path}")
        except Exception as error:
            messagebox.showerror("Configuration error", str(error))

    def _run(self) -> None:
        try:
            config = self._build_config()
        except Exception as error:
            messagebox.showerror("Configuration error", str(error))
            return
        self.run_button.configure(state="disabled")
        self.status.set("Starting analysis...")

        def worker() -> None:
            try:
                result = run_analysis(
                    config,
                    progress=lambda message: self.messages.put(("progress", message)),
                )
                self.messages.put(("complete", result))
            except Exception as error:
                self.messages.put(("error", error))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "progress":
                    self.status.set(str(value))
                elif kind == "complete":
                    self.run_button.configure(state="normal")
                    self.status.set(f"Complete: {value['roi_count']} ROIs")
                    messagebox.showinfo(
                        "Analysis complete",
                        f"Detected {value['roi_count']} ROIs.\n\nResults: {value['output_dir']}",
                    )
                elif kind == "error":
                    self.run_button.configure(state="normal")
                    self.status.set("Analysis failed")
                    messagebox.showerror("Analysis failed", str(value))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)


def main() -> None:
    root = tk.Tk()
    AnalyzerGui(root)
    root.mainloop()
