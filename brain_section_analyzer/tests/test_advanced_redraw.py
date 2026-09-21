import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from tkinterdnd2 import TkinterDnD
from brain_section_analyzer.analysis import analyze_arrays, _write_outputs
from brain_section_analyzer.gui import BrainSectionGui
from brain_section_analyzer.tests.test_advanced import config_for_arrays, image_info


class RedrawTests(unittest.TestCase):
    def test_cached_preview_redraw_does_not_rerun_segmentation_or_read_input(self):
        root = TkinterDnD.Tk()
        root.withdraw()
        try:
            gui = BrainSectionGui(root)
            config = config_for_arrays()
            images = {role: np.zeros((16, 16)) for role in ("abeta", "iba1", "cd68")}
            images["abeta"][4:8, 4:8] = 20
            images["iba1"][6:10, 6:10] = 30
            products = analyze_arrays(images, config, pixel_size_um_x=1, pixel_size_um_y=1)
            before = products.advanced_metrics.copy(deep=True)
            with tempfile.TemporaryDirectory() as directory:
                config["input"]["output_dir"] = directory
                config["output"]["processing_steps"] = ["stage_channel_1_threshold"]
                config["output"]["save_qc"] = False
                gui._apply_config(config)
                gui.preview_context = {"config": config, "images": images, "products": products, "info": image_info()}
                gui.drawing_vars["gamma"].set("2")
                with patch("brain_section_analyzer.gui.threading.Thread", side_effect=lambda target, daemon: SimpleNamespace(start=target)), \
                     patch("brain_section_analyzer.gui.inspect_image", side_effect=AssertionError("must not read input")), \
                     patch("brain_section_analyzer.gui.run_analysis", side_effect=AssertionError("must not rerun analysis")):
                    gui._redraw_preview()
                kind, files = gui.messages.get_nowait()
                self.assertEqual(kind, "redraw_complete", files)
                self.assertTrue(Path(files["processing_stage_channel_1_threshold"]).is_file())
                pd.testing.assert_frame_equal(products.advanced_metrics, before)
        finally:
            root.destroy()

    def test_disabled_neighbour_can_retain_an_inactive_reference_channel(self):
        config = config_for_arrays()
        config["advanced"].update(colocalization_enabled=False, object_distances_enabled=False, pairs=[["dapi", "iba1"]])
        config["plaque"]["reference_channel"] = "dapi"
        images = {role: np.zeros((16, 16)) for role in ("abeta", "iba1", "cd68")}
        images["abeta"][4:8, 4:8] = 20
        products = analyze_arrays(images, config, pixel_size_um_x=1, pixel_size_um_y=1)
        self.assertFalse(products.neighbour_enabled)
        self.assertEqual(products.marker_labels["abeta"].max(), 1)
        with tempfile.TemporaryDirectory() as directory:
            files, _ = _write_outputs(Path(directory), config, image_info(), products, images)
            self.assertIn("qc", files)
            self.assertNotIn("plaque_labels", files)

    def test_constant_intensities_have_nan_pcc_and_a_valid_density_plot(self):
        config = config_for_arrays()
        images = {"abeta": np.full((16, 16), 3.), "iba1": np.full((16, 16), 4.), "cd68": np.zeros((16, 16))}
        products = analyze_arrays(images, config, pixel_size_um_x=1, pixel_size_um_y=1)
        self.assertTrue(np.isnan(products.advanced_metrics.iloc[0].pearson_r))
        with tempfile.TemporaryDirectory() as directory:
            files, _ = _write_outputs(Path(directory), config, image_info(), products, images)
            self.assertTrue(Path(files["processing_advanced_pair_1_2_scatter"]).is_file())


if __name__ == "__main__":
    unittest.main()
