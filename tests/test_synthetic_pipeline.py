"""End-to-end tests using a generated three-channel CZI image."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import tifffile
from pylibCZIrw import czi as pyczi

from cell_analyzer.batch import run_batch_analysis
from cell_analyzer.config import create_default_config
from cell_analyzer.czi_io import inspect_czi, read_czi_channels
from cell_analyzer.interactive import launch_batch_tuning_widget
from cell_analyzer.models import ChannelInfo, CziInfo, SceneInfo
from cell_analyzer.pipeline import run_analysis
from cell_analyzer.preprocessing import preprocess_channel_steps
from cell_analyzer.segmentation import segment_cells, threshold_image


def _disk(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    return (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius**2


def create_three_channel_czi(path: Path) -> None:
    shape = (160, 180)
    rng = np.random.default_rng(7)
    channel_0 = rng.poisson(30, shape).astype(np.uint16)
    channel_1 = rng.poisson(100, shape).astype(np.uint16)
    channel_2 = rng.poisson(60, shape).astype(np.uint16)
    objects = [
        _disk(shape, (45, 45), 14),
        _disk(shape, (110, 65), 18),
        _disk(shape, (82, 135), 16),
    ]
    for index, mask in enumerate(objects):
        channel_0[mask] += np.uint16(2200 + index * 200)
        channel_1[mask] += np.uint16(500 + index * 350)
        channel_2[mask] += np.uint16(1300 - index * 200)

    with pyczi.create_czi(str(path), exist_ok=True) as writer:
        for channel_index, image in enumerate((channel_0, channel_1, channel_2)):
            writer.write(
                image[..., None],
                plane={"C": channel_index, "Z": 0, "T": 0},
                scene=0,
            )


class SyntheticPipelineTest(unittest.TestCase):
    def test_default_config_has_no_intensity_transformation_settings(self) -> None:
        info = CziInfo(
            path="default_background.czi",
            dimensions={"C": (0, 2), "T": (0, 1), "Z": (0, 1), "X": (0, 8), "Y": (0, 8)},
            channels=[
                ChannelInfo(index=0, name="Channel_0"),
                ChannelInfo(index=1, name="Channel_1"),
            ],
            scenes=[SceneInfo(index=0, x=0, y=0, width=8, height=8)],
        )
        config = create_default_config(info)
        for channel in config["channels"].values():
            self.assertEqual(
                set(channel),
                {"alias", "gaussian_sigma_px", "measurement_threshold"},
            )
            self.assertNotIn("background_radius_px", channel)
            self.assertNotIn("normalize_low_percentile", channel)
            self.assertNotIn("normalize_high_percentile", channel)
            self.assertNotIn("clahe_clip_limit", channel)
            self.assertNotIn("value", channel["measurement_threshold"])

    def test_preprocessing_preserves_raw_intensity_scale(self) -> None:
        image = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        processed, steps = preprocess_channel_steps(
            image,
            {"gaussian_sigma_px": 0},
        )
        np.testing.assert_array_equal(processed.raw, image)
        np.testing.assert_array_equal(processed.analysis_image, image.astype(np.float32))
        self.assertEqual(set(steps), {"raw", "gaussian_smoothed"})

    def test_batch_widget_remembers_independent_file_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            first_path = temporary / "first.czi"
            second_path = temporary / "second.czi"
            create_three_channel_czi(first_path)
            create_three_channel_czi(second_path)
            config = create_default_config(
                inspect_czi(first_path), output_dir=temporary / "batch_results"
            )
            panel = launch_batch_tuning_widget(
                config,
                czi_paths=[first_path, second_path],
                output_root=temporary / "batch_results",
            )
            self.assertEqual(panel.current_panel.accordion.selected_index, 1)
            self.assertEqual(
                panel.current_panel.accordion.get_title(0),
                "Per-channel smoothing and signal area",
            )
            self.assertFalse(hasattr(panel.current_panel, "background"))
            self.assertFalse(hasattr(panel.current_panel, "low"))
            self.assertFalse(hasattr(panel.current_panel, "high"))
            self.assertFalse(hasattr(panel.current_panel, "clahe"))
            self.assertFalse(hasattr(panel.current_panel, "cell_value"))
            self.assertFalse(hasattr(panel.current_panel, "signal_value"))
            self.assertEqual(
                panel.current_panel.min_area.description,
                "Minimum cell area (px^2)",
            )
            self.assertIn("Maximum cell area", panel.current_panel.max_area.description)

            panel.current_panel.threshold_scale.value = 0.6
            panel.file_selector.value = str(second_path.resolve())
            first_key = str(first_path.resolve()).casefold()
            self.assertAlmostEqual(
                panel.file_configs[first_key]["segmentation"]["threshold_scale"], 0.6
            )
            self.assertAlmostEqual(
                panel.current_panel.config["segmentation"]["threshold_scale"], 0.8
            )

            panel.current_panel.threshold_scale.value = 0.9
            panel._snapshot_current()
            second_key = str(second_path.resolve()).casefold()
            self.assertAlmostEqual(
                panel.file_configs[second_key]["segmentation"]["threshold_scale"], 0.9
            )
            panel._apply_current_to_all(None)
            self.assertAlmostEqual(panel.config["segmentation"]["threshold_scale"], 0.9)
            self.assertEqual(set(panel.file_configs), {second_key})

    def test_batch_pipeline_writes_per_file_and_combined_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            first_path = temporary / "first.czi"
            second_path = temporary / "second.czi"
            output_root = temporary / "batch_results"
            create_three_channel_czi(first_path)
            create_three_channel_czi(second_path)

            info = inspect_czi(first_path)
            config = create_default_config(info, output_dir=temporary / "unused")
            config["segmentation"].update(
                {
                    "min_area_px": 100,
                    "max_area_px": 2000,
                    "split_touching": False,
                }
            )
            for channel_config in config["channels"].values():
                channel_config.update(
                    {
                        "gaussian_sigma_px": 0.8,
                    }
                )

            result = run_batch_analysis(
                [first_path, second_path],
                config,
                output_root,
                progress=None,
            )
            self.assertEqual(result["file_count"], 2)
            self.assertEqual(result["completed"], 2)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["total_roi_count"], 6)
            self.assertTrue((output_root / "first" / "roi_measurements.csv").is_file())
            self.assertTrue((output_root / "second" / "roi_measurements.csv").is_file())
            summary = pd.read_csv(result["files"]["batch_summary_csv"])
            combined = pd.read_csv(result["files"]["combined_measurements_csv"])
            self.assertEqual(len(summary), 2)
            self.assertEqual(len(combined), 6)
            self.assertIn("source_file", combined.columns)
            self.assertEqual(set(combined["source_name"]), {"first.czi", "second.czi"})

    def test_lower_automatic_threshold_scale_expands_mask(self) -> None:
        image = np.linspace(0.0, 1.0, 10_000, dtype=np.float32).reshape(100, 100)
        strict_mask, strict_threshold = threshold_image(
            image,
            method="otsu",
            threshold_scale=1.0,
        )
        expanded_mask, expanded_threshold = threshold_image(
            image,
            method="otsu",
            threshold_scale=0.7,
        )
        self.assertLess(expanded_threshold, strict_threshold)
        self.assertGreater(int(expanded_mask.sum()), int(strict_mask.sum()))

    def test_enclosed_holes_are_filled(self) -> None:
        shape = (80, 80)
        outer = _disk(shape, (40, 40), 18)
        inner = _disk(shape, (40, 40), 6)
        image = (outer & ~inner).astype(np.float32)
        segmentation_config = {
            "threshold_method": "otsu",
            "threshold_scale": 1.0,
            "opening_radius_px": 0,
            "closing_radius_px": 0,
            "fill_all_holes": True,
            "min_hole_area_px": 0,
            "min_area_px": 1,
            "max_area_px": None,
            "min_circularity": 0,
            "min_local_contrast_ratio": 0,
            "local_contrast_ring_px": 2,
            "clear_border": False,
            "split_touching": False,
        }
        labels, cleaned, diagnostics = segment_cells(image, segmentation_config)
        self.assertTrue(cleaned[40, 40])
        self.assertEqual(int(labels.max()), 1)
        self.assertEqual(diagnostics["connected_component_count"], 1)

    def test_watershed_peak_height_controls_splitting(self) -> None:
        shape = (90, 130)
        mask = _disk(shape, (45, 42), 20) | _disk(shape, (45, 88), 20)
        mask[41:50, 42:89] = True
        image = mask.astype(np.float32)
        segmentation_config = {
            "threshold_method": "otsu",
            "threshold_scale": 1.0,
            "opening_radius_px": 0,
            "closing_radius_px": 0,
            "fill_all_holes": True,
            "min_hole_area_px": 0,
            "min_area_px": 1,
            "max_area_px": None,
            "min_circularity": 0,
            "min_local_contrast_ratio": 0,
            "local_contrast_ring_px": 2,
            "clear_border": False,
            "split_touching": True,
            "min_peak_distance_px": 5,
            "watershed_min_peak_height_px": 0,
            "watershed_min_peak_prominence_px": 0,
            "watershed_compactness": 0,
        }
        split_labels, _, _ = segment_cells(image, segmentation_config)
        self.assertGreaterEqual(int(split_labels.max()), 2)
        segmentation_config["watershed_min_peak_height_px"] = 100
        merged_labels, _, _ = segment_cells(image, segmentation_config)
        self.assertEqual(int(merged_labels.max()), 1)

    def test_area_and_border_filters_keep_only_valid_cells(self) -> None:
        shape = (100, 120)
        image = np.zeros(shape, dtype=np.float32)
        image[_disk(shape, (25, 25), 2)] = 1
        image[_disk(shape, (50, 55), 6)] = 1
        image[_disk(shape, (70, 95), 13)] = 1
        # This object is complete and does not touch the image edge, but it lies
        # inside the configured exclusion margin and must still be rejected.
        image[_disk(shape, (8, 55), 6)] = 1
        segmentation_config = {
            "threshold_method": "otsu",
            "threshold_scale": 1.0,
            "opening_radius_px": 0,
            "closing_radius_px": 0,
            "fill_all_holes": True,
            "min_hole_area_px": 0,
            "min_area_px": 50,
            "max_area_px": 200,
            "min_circularity": 0,
            "min_local_contrast_ratio": 0,
            "local_contrast_ring_px": 2,
            "clear_border": True,
            "border_exclusion_margin_px": 10,
            "split_touching": False,
        }
        labels, _, diagnostics = segment_cells(image, segmentation_config)
        self.assertEqual(int(labels.max()), 1)
        self.assertEqual(diagnostics["accepted"], 1)
        # The very small object is removed during morphology cleanup; the
        # oversized candidate reaches the region filter and is counted here.
        self.assertEqual(diagnostics["rejected_area"], 1)
        self.assertEqual(diagnostics["rejected_border"], 1)

    def test_scene_less_czi_omits_scene_argument(self) -> None:
        info = CziInfo(
            path="scene_less.czi",
            dimensions={"C": (0, 1), "T": (0, 1), "Z": (0, 1), "X": (0, 8), "Y": (0, 8)},
            channels=[ChannelInfo(index=0, name="Channel_0")],
            scenes=[SceneInfo(index=0, x=0, y=0, width=8, height=8)],
        )
        reader = MagicMock()
        reader.__enter__.return_value = reader
        reader.__exit__.return_value = None
        reader.scenes_bounding_rectangle = {}

        def fake_read(**arguments):
            self.assertNotIn("scene", arguments)
            return np.ones((8, 8, 1), dtype=np.uint16)

        reader.read.side_effect = fake_read
        with patch("cell_analyzer.czi_io.pyczi.open_czi", return_value=reader):
            images = read_czi_channels(
                "scene_less.czi",
                info,
                scene=0,
                time_index=0,
                z_projection="max",
                z_index=0,
                zoom=1.0,
            )
        self.assertEqual(images[0].shape, (8, 8))

    def test_three_channel_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            czi_path = temporary / "synthetic_three_channel.czi"
            output_dir = temporary / "results"
            create_three_channel_czi(czi_path)

            info = inspect_czi(czi_path)
            self.assertEqual(len(info.channels), 3)
            config = create_default_config(info, output_dir=output_dir)
            config["input"]["segmentation_channel"] = 0
            config["segmentation"].update(
                {
                    "threshold_method": "otsu",
                    "opening_radius_px": 1,
                    "closing_radius_px": 1,
                    "min_area_px": 100,
                    "max_area_px": 2000,
                    "split_touching": False,
                }
            )
            for channel_config in config["channels"].values():
                channel_config.update(
                    {
                        "gaussian_sigma_px": 0.8,
                    }
                )

            result = run_analysis(config, progress=None)
            self.assertEqual(result["roi_count"], 3)
            measurements = pd.read_csv(result["files"]["measurements_csv"])
            self.assertEqual(len(measurements), 3)
            for excluded_column in (
                "centroid_x_px",
                "centroid_y_px",
                "major_axis_length_analysis_px",
                "minor_axis_length_analysis_px",
                "eccentricity",
            ):
                self.assertNotIn(excluded_column, measurements.columns)
            self.assertFalse(
                any(
                    "background_corrected" in column
                    for column in measurements.columns
                )
            )
            for channel_index in range(3):
                self.assertIn(
                    f"channel_{channel_index}_mean_intensity", measurements.columns
                )
                self.assertIn(
                    f"channel_{channel_index}_positive_area_analysis_px",
                    measurements.columns,
                )
            labels = tifffile.imread(result["files"]["label_image"])
            self.assertEqual(int(labels.max()), 3)
            self.assertTrue(Path(result["files"]["imagej_rois"]).is_file())
            self.assertTrue(Path(result["files"]["geojson"]).is_file())


if __name__ == "__main__":
    unittest.main()
