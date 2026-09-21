from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tkinterdnd2 import TkinterDnD
from brain_section_analyzer.config import DEFAULT_CONFIG, OUTPUT_SELECTION_DEFAULTS, normalize_config
from brain_section_analyzer.analysis import analyze_arrays, _overlay, _write_outputs
from brain_section_analyzer.gui import BrainSectionGui, _preview_image_choices
from brain_section_analyzer.batch import run_batch_analysis
from cell_analyzer.models import ChannelInfo, CziInfo


def config_for_arrays():
    config = copy.deepcopy(DEFAULT_CONFIG)
    for role in ("abeta", "iba1", "cd68"):
        config["channels"][role]["gaussian_sigma_px"] = 0
        config["channels"][role]["threshold"]["method"] = "manual"
        config["channels"][role]["threshold"]["value"] = 1
    config["advanced"].update(colocalization_enabled=True, object_distances_enabled=True, pairs=[["abeta", "iba1"]])
    return config


def image_info():
    return CziInfo(path="test.ome.tif", dimensions={"C": (0, 3), "X": (0, 16), "Y": (0, 16)},
                   channels=[ChannelInfo(index=i, name=f"Channel {i + 1}") for i in range(3)], scenes=[])


class AdvancedTests(unittest.TestCase):
    def test_known_colocalization_and_background_corrected_intensities(self):
        config = config_for_arrays()
        a = np.array([[2., 4.], [6., 8.]])
        b = np.array([[0., 3.], [0., 9.]])
        images = {"abeta": a, "iba1": b, "cd68": np.zeros_like(a)}
        result = analyze_arrays(images, config, pixel_size_um_x=2, pixel_size_um_y=3)
        row = result.advanced_metrics.iloc[0]
        self.assertAlmostEqual(row.pearson_r, np.corrcoef(a.ravel(), b.ravel())[0, 1])
        self.assertAlmostEqual(row.manders_m1, 0.6)
        self.assertAlmostEqual(row.manders_m2, 1)
        self.assertAlmostEqual(row.jaccard, 0.5)
        self.assertAlmostEqual(row.dice, 2 / 3)
        self.assertEqual(row.overlap_area_um2, 12)
        self.assertEqual(row.threshold_a, 1)
        self.assertFalse(result.neighbour_enabled)
        self.assertEqual(result.plaque_labels.max(), 0)
        np.testing.assert_array_equal(result.positive_masks["abeta"], a > 1)
        config["advanced"]["backgrounds"]["abeta"] = 1
        row = analyze_arrays(images, config, pixel_size_um_x=2, pixel_size_um_y=3).advanced_metrics.iloc[0]
        self.assertAlmostEqual(row.manders_m1, 10 / 16)

    def test_distance_anisotropic_calibration_and_missing_target_are_not_zero(self):
        config = config_for_arrays()
        images = {role: np.zeros((9, 9)) for role in ("abeta", "iba1", "cd68")}
        images["abeta"][2, 1] = images["abeta"][6, 1] = 4
        images["iba1"][2, 4] = 4
        products = analyze_arrays(images, config, pixel_size_um_x=2, pixel_size_um_y=3)
        np.testing.assert_allclose(products.object_distances.minimum_mask_distance_um, [6, np.sqrt(180)])
        np.testing.assert_allclose(products.object_distances.nearest_centroid_distance_um, [6, np.sqrt(180)])
        images["iba1"][:] = 0
        products = analyze_arrays(images, config, pixel_size_um_x=2, pixel_size_um_y=3)
        self.assertTrue(products.object_distances.minimum_mask_distance_um.isna().all())
        self.assertFalse(products.object_distances.target_available.any())
        self.assertTrue(np.isnan(products.advanced_metrics.iloc[0].fraction_a_within_proximity))
        self.assertTrue(np.isnan(products.advanced_metrics.iloc[0].pearson_r))

    def test_selected_stages_and_style_do_not_change_measurements(self):
        config = config_for_arrays()
        images = {role: np.zeros((16, 16)) for role in ("abeta", "iba1", "cd68")}
        images["abeta"][5:9, 5:9] = 10
        images["iba1"][7:11, 7:11] = 20
        products = analyze_arrays(images, config, pixel_size_um_x=1, pixel_size_um_y=1)
        before = products.advanced_metrics.copy(deep=True)
        for key in OUTPUT_SELECTION_DEFAULTS:
            config["output"][key] = False
        config["output"].update(save_stage_images=True, save_advanced_images=True,
            processing_steps=["stage_channel_1_threshold", "advanced_overlap"])
        config["output"]["drawing"]["boundaries"]["overlap"] = {"color": "#123456", "width_px": 3}
        with tempfile.TemporaryDirectory() as directory:
            files, _ = _write_outputs(Path(directory), config, image_info(), products, images)
            previews = _preview_image_choices(files)
            self.assertEqual(len(previews), 2)
            self.assertTrue(any("Threshold" in name for name in previews))
            overlay_path = next(path for name, path in previews.items() if "Overlap" in name)
            pixels = np.asarray(Image.open(overlay_path).convert("RGB"))
            self.assertTrue(np.any(np.all(pixels == [18, 52, 86], axis=-1)))
        pd.testing.assert_frame_equal(before, products.advanced_metrics)
        overlay = _overlay(np.zeros((5, 5, 3)), np.eye(5, dtype=bool), (1, 0, 0), opacity=0.5)
        self.assertEqual(overlay[0, 0, 0], 0.5)
        self.assertEqual(overlay[0, 1, 0], 0)

    def test_old_sessions_migrate_and_invalid_pairs_and_scopes_fail(self):
        old = copy.deepcopy(DEFAULT_CONFIG)
        old.pop("advanced")
        old["output"].pop("drawing")
        old["output"]["ring_boundary_width_px"] = 4
        normalized = normalize_config(old, image_info())
        self.assertTrue(normalized["advanced"]["neighbour_enabled"])
        self.assertEqual(normalized["output"]["drawing"]["boundaries"]["ring"]["width_px"], 4)
        config = config_for_arrays()
        for update in ({"pairs": []}, {"pairs": [["dapi", "iba1"]]}, {"scope": "cells"}, {"scope": "reference_objects"}):
            with self.subTest(update=update):
                invalid = copy.deepcopy(config)
                invalid["advanced"].update(update)
                with self.assertRaises(ValueError):
                    normalize_config(invalid, image_info())

    def test_gui_advanced_and_drawing_configuration_roundtrip(self):
        root = TkinterDnD.Tk()
        root.withdraw()
        try:
            gui = BrainSectionGui(root)
            self.assertFalse(gui.advanced_vars["neighbour_enabled"].get())
            config = config_for_arrays()
            config["output"]["drawing"]["boundaries"]["iba1"] = {"color": "#123456", "width_px": 4}
            config["output"]["processing_steps"] = ["stage_channel_2_candidates"]
            gui._apply_config(config)
            self.assertEqual(gui._advanced_config(), config["advanced"])
            self.assertEqual(gui._drawing_config(), config["output"]["drawing"])
            self.assertEqual([key for key, variable in gui.processing_vars.items() if variable.get()], config["output"]["processing_steps"])
            gui.pair_a.set("Channel 2")
            gui.pair_b.set("Channel 3")
            gui._add_advanced_pair()
            self.assertEqual(gui.advanced_pairs[-1], ["iba1", "cd68"])
            gui._plot_settings()
            root.update_idletasks()
        finally:
            root.destroy()

    def test_batch_exports_selected_advanced_columns_to_csv_and_excel(self):
        config = config_for_arrays()
        config["output"]["table_columns"]["advanced_metrics"] = ["source_name", "channel_a", "pearson_r", "manders_m1"]
        for key in OUTPUT_SELECTION_DEFAULTS:
            config["output"][key] = False
        config["output"].update(save_excel=True, save_advanced_metrics_csv=True, save_object_distances_csv=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for index in range(2):
                image = np.zeros((3, 16, 16), dtype=np.uint16)
                image[0, 5:9, 5:9] = 20
                image[1, 7:11, 7:11] = 30
                path = root / f"test_{index}.ome.tif"
                tifffile.imwrite(path, image, photometric="minisblack", ome=True, metadata={"axes": "CYX", "PhysicalSizeX": 1., "PhysicalSizeY": 1.})
                files.append(path)
            result = run_batch_analysis(files, config, root / "out")
            self.assertEqual(result["completed"], 2, result["records"])
            table = pd.read_csv(result["files"]["advanced_metrics"])
            self.assertEqual(list(table.columns), ["source_name", "channel_a", "pearson_r", "manders_m1"])
            self.assertEqual(len(table), 2)
            book = pd.ExcelFile(result["files"]["excel"])
            self.assertIn("Advanced Metrics", book.sheet_names)
            self.assertIn("Object Distances", book.sheet_names)
            book.close()


if __name__ == "__main__":
    unittest.main()
