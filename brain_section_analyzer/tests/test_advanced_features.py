from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from tkinterdnd2 import TkinterDnD
from brain_section_analyzer.advanced_features import analyze_features, _skeleton_graph
from brain_section_analyzer.analysis import analyze_arrays, _write_outputs
from brain_section_analyzer.batch import run_batch_analysis
from brain_section_analyzer.config import DEFAULT_CONFIG, OUTPUT_SELECTION_DEFAULTS, normalize_config
from brain_section_analyzer.gui import BrainSectionGui, _preview_image_choices
from brain_section_analyzer.tests.test_advanced import config_for_arrays, image_info


def feature_inputs(shape=(15, 15)):
    config = config_for_arrays()
    config["advanced"].update(colocalization_enabled=False, object_distances_enabled=False, pairs=[])
    images = {role: np.zeros(shape) for role in ("abeta", "iba1", "cd68")}
    labels = {role: np.zeros(shape, np.int32) for role in images}
    return config, images, labels, np.zeros(shape, np.int32), np.ones(shape, bool)


def run_features(inputs, pixel_x=1, pixel_y=1):
    config, images, labels, cells, domain = inputs
    return analyze_features(images, labels, cells, domain, config, pixel_x, pixel_y, "synthetic.tif")


class FeatureTests(unittest.TestCase):
    def test_cell_regions_anisotropic_radius_background_and_finite_pixels(self):
        inputs = feature_inputs((9, 9))
        config, images, labels, cells, domain = inputs
        config["advanced"].update(cell_measurements_enabled=True, cell_expansion_um=2)
        config["advanced"]["backgrounds"]["iba1"] = 2
        config["microglia_count"].update(enabled=True, nucleus_channel="abeta")
        cells[4, 4] = 7
        images["iba1"][:] = 6
        images["iba1"][4, 4] = np.nan
        labels["iba1"][4, 3] = 1
        tables, maps, _ = run_features(inputs, pixel_x=2, pixel_y=3)
        regions = maps["advanced_cell_regions"][1]
        self.assertEqual(np.count_nonzero(regions), 3)
        row = tables["cell_measurements"].query("channel == 'Channel 2'").iloc[0]
        self.assertEqual(row.cell_id, 7)
        self.assertEqual(row.region_area_um2, 18)
        self.assertEqual(row.valid_area_um2, 12)
        self.assertEqual(row.mean_intensity, 4)
        self.assertEqual(row.integrated_intensity, 8)
        self.assertEqual(row.intensity_area_integral, 48)
        self.assertEqual(row.positive_fraction, 0.5)
        self.assertEqual(row.positive_mean_intensity, 4)

    def test_cell_expansion_does_not_share_pixels_and_clips_roi(self):
        inputs = feature_inputs((9, 9))
        config, _, _, cells, domain = inputs
        config["advanced"].update(cell_measurements_enabled=True, cell_expansion_um=10)
        config["microglia_count"].update(enabled=True, nucleus_channel="abeta")
        cells[4, 2], cells[4, 6] = 1, 2
        domain[0] = False
        tables, maps, _ = run_features(inputs)
        regions = maps["advanced_cell_regions"][1]
        self.assertTrue(np.all(regions[~domain] == 0))
        measured = tables["cell_measurements"].query("channel == 'Channel 1'")
        self.assertEqual(measured.region_area_um2.sum(), domain.sum())
        self.assertEqual(set(measured.cell_id), {1, 2})

    def test_radial_interior_and_edge_bins_anisotropic_and_no_double_count(self):
        inputs = feature_inputs((9, 9))
        config, images, labels, _, domain = inputs
        config["advanced"].update(radial_profiles_enabled=True, radial_reference_channel="abeta", radial_max_um=3, radial_step_um=2)
        labels["abeta"][4, 2], labels["abeta"][4, 6] = 5, 9
        images["iba1"][:] = 10
        labels["iba1"][4, 2] = 3
        tables, maps, _ = run_features(inputs, pixel_x=2, pixel_y=3)
        rows = tables["radial_profiles"].query("channel == 'Channel 2'")
        interior = rows.loc[rows.is_interior]
        self.assertEqual(set(interior.reference_object_id), {5, 9})
        self.assertTrue((interior.bin_outer_um == 0).all())
        self.assertTrue((rows.loc[rows.bin_index == 2].bin_outer_um == 3).all())
        self.assertEqual(rows.region_area_um2.sum(), np.count_nonzero(maps["advanced_radial_regions"][1]) * 6)
        self.assertEqual(interior.loc[interior.reference_object_id == 5, "positive_fraction"].iloc[0], 1)

    def test_spatial_centroids_radius_excludes_self_and_flags_border(self):
        inputs = feature_inputs()
        config, _, labels, _, _ = inputs
        config["advanced"].update(spatial_distribution_enabled=True, object_channels=["iba1"], spatial_radius_um=7)
        labels["iba1"][2, 2], labels["iba1"][2, 5], labels["iba1"][10, 10] = 1, 3, 4
        tables, maps, summaries = run_features(inputs, pixel_x=2, pixel_y=3)
        rows = tables["spatial_objects"]
        np.testing.assert_allclose(rows.nearest_neighbour_distance_um.iloc[:2], [6, 6])
        self.assertEqual(rows.neighbours_within_radius.tolist(), [1, 1, 0])
        self.assertTrue(rows.radius_roi_truncated.iloc[0])
        self.assertEqual(summaries.iloc[0].density_per_mm2, 3 / (225 * 6 / 1e6))
        labels["iba1"][2, 5] = labels["iba1"][10, 10] = 0
        tables, _, _ = run_features(inputs)
        self.assertTrue(tables["spatial_objects"].nearest_neighbour_distance_um.isna().all())
        self.assertEqual(tables["spatial_objects"].neighbours_within_radius.iloc[0], 0)

    def test_skeleton_horizontal_diagonal_and_junction_clusters(self):
        horizontal = np.ones((1, 7), bool)
        _, _, _, row = _skeleton_graph(horizontal, 2, 3)
        self.assertEqual(row["skeleton_graph_length_um"], 12)
        self.assertEqual(row["endpoint_count"], 2)
        _, _, _, row = _skeleton_graph(np.eye(7, dtype=bool), 2, 3)
        self.assertAlmostEqual(row["skeleton_graph_length_um"], 6 * np.sqrt(13))
        tee = np.zeros((9, 9), bool)
        tee[4, 1:8] = True
        tee[1:5, 4] = True
        _, _, _, row = _skeleton_graph(tee, 1, 1)
        self.assertEqual(row["endpoint_count"], 3)
        self.assertEqual(row["junction_cluster_count"], 1)
        _, _, _, row = _skeleton_graph(np.ones((1, 1), bool), 1, 1)
        self.assertEqual(row["skeleton_graph_length_um"], 0)
        self.assertEqual(row["isolated_pixel_count"], 1)

    def test_empty_modules_keep_export_schemas_and_do_not_modify_labels(self):
        inputs = feature_inputs()
        config, _, labels, _, _ = inputs
        config["advanced"].update(cell_measurements_enabled=True, radial_profiles_enabled=True,
                                  skeleton_enabled=True, spatial_distribution_enabled=True)
        config["microglia_count"].update(enabled=True, nucleus_channel="abeta")
        before = {role: array.copy() for role, array in labels.items()}
        tables, maps, _ = run_features(inputs)
        self.assertEqual(set(tables), {"cell_measurements", "radial_profiles", "spatial_objects", "skeleton_objects"})
        for table in tables.values():
            self.assertTrue(table.empty)
            self.assertIn("source_file", table.columns)
        for role in labels:
            np.testing.assert_array_equal(labels[role], before[role])

    def test_parameter_validation_and_missing_new_options(self):
        config, *_ = feature_inputs()
        for update in ({"cell_measurements_enabled": True}, {"radial_step_um": 0}, {"radial_max_um": np.inf},
                       {"radial_max_um": 1000, "radial_step_um": 1}, {"skeleton_enabled": True, "object_channels": []},
                       {"spatial_distribution_enabled": True, "object_channels": ["dapi"]},
                       {"radial_profiles_enabled": True, "radial_reference_channel": "dapi"},
                       {"skeleton_enabled": True, "scope": "reference_objects"}):
            bad = copy.deepcopy(config)
            bad["advanced"].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                normalize_config(bad, image_info())
        config["advanced"] = {"neighbour_enabled": False}
        restored = normalize_config(config, image_info())
        self.assertFalse(restored["advanced"]["skeleton_enabled"])
        self.assertEqual(restored["advanced"]["radial_step_um"], 5)

    def test_gui_tab_order_and_all_new_parameters_roundtrip(self):
        root = TkinterDnD.Tk()
        root.withdraw()
        try:
            gui = BrainSectionGui(root)
            labels = [gui.notebook.tab(tab, "text") for tab in gui.notebook.tabs()]
            self.assertLess(labels.index("Cell Counting"), labels.index("Advanced Analysis"))
            config, *_ = feature_inputs()
            config["advanced"].update(cell_measurements_enabled=True, radial_profiles_enabled=True,
                spatial_distribution_enabled=True, skeleton_enabled=True, cell_expansion_um=4,
                radial_reference_channel="cd68", radial_max_um=21, radial_step_um=3, spatial_radius_um=17, object_channels=["iba1"])
            gui._apply_config(config)
            self.assertEqual(gui._advanced_config(), config["advanced"])
            for tab in gui.advanced_notebook.tabs():
                gui.advanced_notebook.select(tab)
                root.update_idletasks()
        finally:
            root.destroy()

    def test_end_to_end_new_tables_images_column_selection_and_batch(self):
        config, images, _, _, _ = feature_inputs((16, 16))
        images["abeta"][5:8, 5:8] = 10
        images["iba1"][6:9, 7:10] = 20
        config["advanced"].update(cell_measurements_enabled=True, radial_profiles_enabled=True,
            spatial_distribution_enabled=True, skeleton_enabled=True, radial_reference_channel="iba1", object_channels=["iba1"])
        config["microglia_count"].update(enabled=True, nucleus_channel="abeta", confirmation_channel=None,
            min_nucleus_area_um2=1, max_nucleus_area_um2=None, min_circularity=0, min_solidity=0,
            max_eccentricity=1, split_touching=False, closing_radius_px=0)
        config["output"]["table_columns"]["cell_measurements"] = ["cell_id", "channel", "mean_intensity"]
        for key in OUTPUT_SELECTION_DEFAULTS:
            config["output"][key] = False
        config["output"].update(save_excel=True, save_cell_measurements_csv=True, save_radial_profiles_csv=True,
            save_spatial_objects_csv=True, save_skeleton_objects_csv=True, save_advanced_images=True,
            processing_steps=["advanced_cell_regions", "advanced_radial_plot", "advanced_skeleton"],
            qc_panels=["advanced_skeleton"])
        products = analyze_arrays(images, config, pixel_size_um_x=1, pixel_size_um_y=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, tables = _write_outputs(root / "single", config, image_info(), products, images)
            self.assertEqual(len(_preview_image_choices(files)), 3)
            table = pd.read_csv(files["cell_measurements_csv"])
            self.assertEqual(list(table.columns), ["cell_id", "channel", "mean_intensity"])
            self.assertEqual(len(table), 3)
            path = root / "fixture.ome.tif"
            tifffile.imwrite(path, np.stack(list(images.values())).astype(np.uint16), photometric="minisblack", ome=True,
                metadata={"axes": "CYX", "PhysicalSizeX": 1., "PhysicalSizeY": 1.})
            result = run_batch_analysis([path], config, root / "batch")
            self.assertEqual(result["completed"], 1, result["records"])
            table = pd.read_csv(result["files"]["cell_measurements"])
            self.assertEqual(list(table.columns), ["cell_id", "channel", "mean_intensity"])
            with pd.ExcelFile(result["files"]["excel"]) as book:
                for sheet in ("Cell Measurements", "Radial Profiles", "Spatial Objects", "Skeleton Objects"):
                    self.assertIn(sheet, book.sheet_names)


if __name__ == "__main__":
    unittest.main()
