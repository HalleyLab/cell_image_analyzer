from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from skimage.draw import disk

from brain_section_analyzer.analysis import (
    _channel_colors,
    _overlay,
    _clean_and_label_plaques,
    _plaque_ring_metrics_table,
    _select_output_columns,
    _write_outputs,
    analyze_arrays,
)
from brain_section_analyzer.config import (
    DEFAULT_CONFIG,
    OUTPUT_SELECTION_DEFAULTS,
    _guess_channel_indices,
)
from cell_analyzer.models import ChannelInfo, CziInfo


def manual_config() -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    for role in ("abeta", "iba1", "cd68"):
        config["channels"][role]["gaussian_sigma_px"] = 0
        config["channels"][role]["threshold"] = {
            "method": "manual",
            "value": 5,
            "scale": 1,
            "percentile": 95,
        }
    config["plaque"].update(
        {
            "opening_radius_px": 0,
            "closing_radius_px": 0,
            "min_area_um2": 5,
            "neuron_exclusion_mode": "off",
            "exclude_boundary_plaques_from_table": True,
        }
    )
    config["spatial"]["ring_edges_um"] = [0.0, 10.0]
    return config


class BrainSectionAnalysisTests(unittest.TestCase):
    def test_overlay_line_width_expands_boundary_without_changing_background(self) -> None:
        image = np.zeros((7, 7, 3), dtype=float)
        boundary = np.zeros((7, 7), dtype=bool)
        boundary[3, 3] = True

        overlay = _overlay(image, boundary, (1.0, 1.0, 0.0), width_px=3)

        self.assertEqual(int(np.count_nonzero(overlay[..., 0])), 9)
        self.assertTrue(np.all(overlay[2:5, 2:5] == (1.0, 1.0, 0.0)))
        self.assertTrue(np.all(overlay[0, 0] == 0))

    def test_filename_wavelengths_map_real_channel_order(self) -> None:
        info = CziInfo(
            path="AD3 ab-488+cd68-555+iba1-647 20x.czi",
            dimensions={"C": (0, 4), "X": (0, 100), "Y": (0, 100)},
            channels=[
                ChannelInfo(index=0, name="AF647-T1"),
                ChannelInfo(index=1, name="AF546-T2"),
                ChannelInfo(index=2, name="AF488-T3"),
                ChannelInfo(index=3, name="DAPI-T4"),
            ],
            scenes=[],
        )
        self.assertEqual(
            _guess_channel_indices(info),
            {"abeta": 2, "cd68": 1, "iba1": 0, "dapi": 3},
        )

    def test_configured_channel_color_overrides_microscope_metadata(self) -> None:
        info = CziInfo(
            path="test.czi",
            dimensions={"C": (0, 1), "X": (0, 10), "Y": (0, 10)},
            channels=[ChannelInfo(index=0, name="AF647-T1", color="#FFFF0000")],
            scenes=[],
        )
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["channels"]["iba1"].update({"index": 0, "color": "#123456"})
        self.assertEqual(
            _channel_colors(info, config, {"iba1"})["iba1"],
            (18 / 255.0, 52 / 255.0, 86 / 255.0),
        )

    def test_plaque_and_ring_metrics(self) -> None:
        shape = (100, 100)
        abeta = np.zeros(shape, dtype=np.uint16)
        iba1 = np.zeros(shape, dtype=np.uint16)
        cd68 = np.zeros(shape, dtype=np.uint16)
        plaque_rr, plaque_cc = disk((50, 50), 5, shape=shape)
        abeta[plaque_rr, plaque_cc] = 100
        yy, xx = np.indices(shape)
        radial = np.sqrt((yy - 50) ** 2 + (xx - 50) ** 2)
        ring = (radial >= 5) & (radial <= 14)
        iba1[ring] = 50
        cd68[ring & (xx >= 50)] = 40
        iba1[plaque_rr, plaque_cc] = 50
        cd68[plaque_rr, plaque_cc] = 40

        config = manual_config()
        config["spatial"]["ring_edges_um"] = [0.0, 10.0, 20.0]
        products = analyze_arrays(
            {"abeta": abeta, "iba1": iba1, "cd68": cd68},
            config,
            pixel_size_um_x=1,
            pixel_size_um_y=1,
        )
        summary = products.image_summary.iloc[0]
        self.assertEqual(int(summary["plaque_count_all"]), 1)
        self.assertEqual(len(products.plaque_measurements), 1)
        self.assertTrue(
            products.ring_masks["ring_0_10um"][products.plaque_labels > 0].all()
        )
        self.assertTrue(
            products.ring_masks["ring_0_20um"][products.plaque_labels > 0].all()
        )
        self.assertNotIn("ring_10_20um", products.ring_masks)
        self.assertGreater(summary["ring_0_10um_iba1_positive_fraction"], 0.8)
        self.assertGreater(summary["ring_0_10um_cd68_in_iba1_fraction_of_iba1"], 0.4)
        self.assertLess(summary["ring_0_10um_cd68_in_iba1_fraction_of_iba1"], 0.6)

    def test_boundary_plaque_retained_in_burden_but_excluded_from_table(self) -> None:
        shape = (80, 80)
        abeta = np.zeros(shape, dtype=np.uint16)
        iba1 = np.zeros(shape, dtype=np.uint16)
        cd68 = np.zeros(shape, dtype=np.uint16)
        for center in ((0, 20), (40, 40)):
            rr, cc = disk(center, 5, shape=shape)
            abeta[rr, cc] = 100
        products = analyze_arrays(
            {"abeta": abeta, "iba1": iba1, "cd68": cd68},
            manual_config(),
            pixel_size_um_x=1,
            pixel_size_um_y=1,
        )
        summary = products.image_summary.iloc[0]
        self.assertEqual(int(summary["plaque_count_all"]), 2)
        self.assertEqual(int(summary["plaque_count_boundary"]), 1)
        self.assertEqual(int(summary["plaque_count_interior"]), 1)
        self.assertEqual(len(products.plaque_measurements), 1)
        self.assertGreater(summary["abeta_positive_area_um2"], products.plaque_measurements.iloc[0]["plaque_area_um2"])

    def test_touching_plaques_can_be_split_with_watershed(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        for center in ((50, 44), (50, 56)):
            rr, cc = disk(center, 12, shape=mask.shape)
            mask[rr, cc] = True

        config = manual_config()
        config["plaque"]["split_touching"] = False
        _, unsplit = _clean_and_label_plaques(mask, config, pixel_area_um2=1.0)

        config["plaque"].update(
            {
                "split_touching": True,
                "min_peak_distance_px": 8,
                "watershed_min_peak_height_px": 2.0,
            }
        )
        _, split = _clean_and_label_plaques(mask, config, pixel_area_um2=1.0)
        self.assertEqual(int(unsplit.max()), 1)
        self.assertEqual(int(split.max()), 2)

    def test_eccentricity_filter_removes_elongated_objects(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        rr, cc = disk((30, 30), 10, shape=mask.shape)
        mask[rr, cc] = True
        mask[65:70, 45:85] = True

        config = manual_config()
        config["plaque"]["max_eccentricity"] = 0.8
        _, labels = _clean_and_label_plaques(mask, config, pixel_area_um2=1.0)
        self.assertEqual(int(labels.max()), 1)

    def test_dark_center_soma_exclusion_preserves_filled_round_plaque(self) -> None:
        shape = (100, 100)
        abeta = np.zeros(shape, dtype=np.uint16)
        iba1 = np.zeros(shape, dtype=np.uint16)
        cd68 = np.zeros(shape, dtype=np.uint16)

        soma_rr, soma_cc = disk((30, 30), 8, shape=shape)
        nucleus_rr, nucleus_cc = disk((30, 30), 3, shape=shape)
        abeta[soma_rr, soma_cc] = 100
        abeta[nucleus_rr, nucleus_cc] = 0
        plaque_rr, plaque_cc = disk((70, 70), 8, shape=shape)
        abeta[plaque_rr, plaque_cc] = 100

        config = manual_config()
        config["plaque"].update(
            {
                "neuron_exclusion_mode": "shape_and_dark_center",
                "neuron_min_diameter_um": 8.0,
                "neuron_max_diameter_um": 28.0,
                "neuron_min_circularity": 0.60,
                "neuron_min_solidity": 0.70,
                "neuron_min_hole_fraction": 0.05,
                "neuron_max_center_shell_ratio": 0.90,
            }
        )
        products = analyze_arrays(
            {"abeta": abeta, "iba1": iba1, "cd68": cd68},
            config,
            pixel_size_um_x=1,
            pixel_size_um_y=1,
        )
        summary = products.image_summary.iloc[0]
        self.assertEqual(int(summary["abeta_neuron_like_excluded_count"]), 1)
        self.assertEqual(int(summary["plaque_count_all"]), 1)
        self.assertTrue(products.neuron_like_mask[30, 34])
        self.assertGreater(products.plaque_labels[70, 70], 0)

    def test_plaque_selection_does_not_depend_on_cd68(self) -> None:
        shape = (80, 80)
        abeta = np.zeros(shape, dtype=np.uint16)
        rr, cc = disk((40, 40), 9, shape=shape)
        abeta[rr, cc] = 100
        config = manual_config()
        config["plaque"]["neuron_exclusion_mode"] = "off"

        low_cd68 = analyze_arrays(
            {
                "abeta": abeta,
                "iba1": np.zeros(shape, dtype=np.uint16),
                "cd68": np.zeros(shape, dtype=np.uint16),
            },
            config,
            pixel_size_um_x=1,
            pixel_size_um_y=1,
        )
        high_cd68 = analyze_arrays(
            {
                "abeta": abeta,
                "iba1": np.full(shape, 100, dtype=np.uint16),
                "cd68": np.full(shape, 100, dtype=np.uint16),
            },
            config,
            pixel_size_um_x=1,
            pixel_size_um_y=1,
        )
        np.testing.assert_array_equal(low_cd68.plaque_labels, high_cd68.plaque_labels)

    def test_iba1_and_cd68_object_filters_are_independent_and_auditable(self) -> None:
        shape = (120, 120)
        abeta = np.zeros(shape, dtype=np.uint16)
        iba1 = np.zeros(shape, dtype=np.uint16)
        cd68 = np.zeros(shape, dtype=np.uint16)

        rr, cc = disk((60, 60), 7, shape=shape)
        abeta[rr, cc] = 100

        # Iba1: one accepted round object, one tiny object, and one elongated object.
        rr, cc = disk((25, 25), 5, shape=shape)
        iba1[rr, cc] = 100
        iba1[10, 90] = 100
        iba1[85:88, 15:55] = 100

        # CD68: one accepted round object and one object below the area threshold.
        rr, cc = disk((35, 85), 4, shape=shape)
        cd68[rr, cc] = 100
        cd68[95:97, 95:97] = 100

        config = manual_config()
        config["channels"]["iba1"]["object_filter"].update(
            {
                "min_area_um2": 10.0,
                "min_circularity": 0.60,
                "max_eccentricity": 0.85,
            }
        )
        config["channels"]["cd68"]["object_filter"].update(
            {
                "min_area_um2": 10.0,
                "min_circularity": 0.50,
                "max_eccentricity": 0.90,
            }
        )
        products = analyze_arrays(
            {"abeta": abeta, "iba1": iba1, "cd68": cd68},
            config,
            pixel_size_um_x=1,
            pixel_size_um_y=1,
        )
        summary = products.image_summary.iloc[0]
        self.assertEqual(int(summary["iba1_component_candidate_count"]), 3)
        self.assertEqual(int(summary["iba1_component_accepted_count"]), 1)
        self.assertEqual(int(summary["iba1_component_excluded_count"]), 2)
        self.assertEqual(int(summary["cd68_component_candidate_count"]), 2)
        self.assertEqual(int(summary["cd68_component_accepted_count"]), 1)
        self.assertEqual(int(summary["cd68_component_excluded_count"]), 1)
        self.assertTrue(products.positive_masks["iba1"][25, 25])
        self.assertFalse(products.positive_masks["iba1"][10, 90])
        self.assertFalse(products.positive_masks["iba1"][86, 30])
        self.assertTrue(products.positive_masks["cd68"][35, 85])
        self.assertFalse(products.positive_masks["cd68"][95, 95])
        self.assertEqual(int(products.plaque_labels.max()), 1)

        iba1_qc = products.marker_component_qc.query("marker == 'iba1'")
        self.assertEqual(int(iba1_qc["accepted"].sum()), 1)
        rejected_reasons = ";".join(
            iba1_qc.loc[~iba1_qc["accepted"], "exclusion_reason"].astype(str)
        )
        self.assertIn("below_min_area", rejected_reasons)
        self.assertTrue(
            "below_min_circularity" in rejected_reasons
            or "above_max_eccentricity" in rejected_reasons
        )

    def test_dapi_iba1_microglia_are_counted_by_plaque_ring(self) -> None:
        shape = (100, 100)
        abeta = np.zeros(shape, dtype=np.uint16)
        iba1 = np.zeros(shape, dtype=np.uint16)
        cd68 = np.zeros(shape, dtype=np.uint16)
        dapi = np.zeros(shape, dtype=np.uint16)
        rr, cc = disk((50, 50), 5, shape=shape)
        abeta[rr, cc] = 100
        rr, cc = disk((80, 20), 5, shape=shape)
        abeta[rr, cc] = 100
        for center, has_iba1 in (((50, 60), True), ((20, 20), True), ((80, 80), False)):
            rr, cc = disk(center, 3, shape=shape)
            dapi[rr, cc] = 100
            if has_iba1:
                rr, cc = disk(center, 6, shape=shape)
                iba1[rr, cc] = 100

        config = manual_config()
        config["channels"]["dapi"].update(
            {
                "enabled": True,
                "gaussian_sigma_px": 0,
                "threshold": {
                    "method": "manual",
                    "value": 5,
                    "scale": 1,
                    "percentile": 95,
                },
            }
        )
        config["microglia_count"].update(
            {
                "enabled": True,
                "nucleus_channel": "dapi",
                "confirmation_channel": "iba1",
                "closing_radius_px": 0,
                "min_nucleus_area_um2": 10,
                "max_nucleus_area_um2": 100,
                "min_circularity": 0.1,
                "min_solidity": 0.5,
                "max_eccentricity": 1.0,
                "split_touching": False,
                "perinuclear_radius_um": 2.0,
                "min_confirmation_positive_fraction": 0.1,
            }
        )
        config["plaque"].update(
            {
                "require_nearby_microglia": True,
                "nearby_microglia_radius_um": 15.0,
                "min_nearby_microglia_count": 1,
            }
        )
        products = analyze_arrays(
            {"abeta": abeta, "iba1": iba1, "cd68": cd68, "dapi": dapi},
            config,
            pixel_size_um_x=1,
            pixel_size_um_y=1,
        )
        summary = products.image_summary.iloc[0]
        self.assertEqual(int(summary["microglia_count_roi"]), 2)
        self.assertEqual(
            int(summary["plaque_count_before_nearby_microglia_filter"]), 2
        )
        self.assertEqual(int(summary["plaque_count_all"]), 1)
        self.assertEqual(
            int(summary["plaque_without_nearby_microglia_excluded_count"]), 1
        )
        self.assertEqual(int(summary["ring_0_10um_microglia_count"]), 1)
        self.assertEqual(int(products.microglia_labels.max()), 2)
        self.assertEqual(
            int(products.microglia_cells["accepted_microglia"].sum()), 2
        )
        self.assertEqual(
            int(products.plaque_measurements.iloc[0]["ring_0_10um_microglia_count"]),
            1,
        )
        self.assertEqual(
            int(products.plaque_measurements.iloc[0]["nearby_microglia_count"]),
            1,
        )
        self.assertTrue(products.microglia_absent_plaque_mask.any())

        ring_table = _plaque_ring_metrics_table(
            products.plaque_measurements, ["ring_0_10um"]
        )
        self.assertEqual(len(ring_table), 1)
        self.assertIn("ring_iba1_positive_area_um2", ring_table)
        self.assertIn("ring_cd68_in_iba1_fraction_of_iba1", ring_table)


    def test_cell_count_can_use_any_selected_channel_without_confirmation(self) -> None:
        shape = (60, 60)
        images = {
            "abeta": np.zeros(shape, dtype=np.uint16),
            "iba1": np.zeros(shape, dtype=np.uint16),
            "cd68": np.zeros(shape, dtype=np.uint16),
        }
        rr, cc = disk((30, 30), 5, shape=shape)
        images["abeta"][rr, cc] = 100
        for center in ((15, 15), (45, 45)):
            rr, cc = disk(center, 3, shape=shape)
            images["cd68"][rr, cc] = 100
        config = manual_config()
        config["microglia_count"].update(
            {
                "enabled": True,
                "nucleus_channel": "cd68",
                "confirmation_channel": None,
                "closing_radius_px": 0,
                "min_nucleus_area_um2": 5,
                "max_nucleus_area_um2": 100,
                "min_circularity": 0,
                "min_solidity": 0,
                "max_eccentricity": 1,
                "split_touching": False,
            }
        )

        products = analyze_arrays(images, config, pixel_size_um_x=1, pixel_size_um_y=1)

        self.assertEqual(int(products.image_summary.iloc[0]["microglia_count_roi"]), 2)
        self.assertEqual(set(products.microglia_cells["nucleus_channel"]), {"cd68"})
        self.assertTrue(products.microglia_cells["confirmation_channel"].eq("").all())

    def test_selected_output_columns_are_exact_and_ordered(self) -> None:
        frame = pd.DataFrame({"first": [1], "second": [2], "third": [3]})
        config = {"output": {"table_columns": {"example": ["third", "first"]}}}

        selected = _select_output_columns(frame, config, "example")

        self.assertEqual(list(selected.columns), ["third", "first"])

    def test_output_selection_writes_only_requested_result(self) -> None:
        shape = (40, 40)
        images = {
            "abeta": np.zeros(shape, dtype=np.uint16),
            "iba1": np.zeros(shape, dtype=np.uint16),
            "cd68": np.zeros(shape, dtype=np.uint16),
        }
        rr, cc = disk((20, 20), 5, shape=shape)
        images["abeta"][rr, cc] = 100
        config = manual_config()
        for key in OUTPUT_SELECTION_DEFAULTS:
            config["output"][key] = False
        config["output"]["save_primary_objects_csv"] = True
        config["output"]["table_columns"] = {
            "primary_objects": ["primary_object_id", "primary_object_area_um2"]
        }
        products = analyze_arrays(
            images,
            config,
            pixel_size_um_x=1,
            pixel_size_um_y=1,
        )
        info = SimpleNamespace(to_dict=lambda: {"source": "synthetic"})
        with tempfile.TemporaryDirectory() as directory:
            paths, tables = _write_outputs(
                Path(directory), config, info, products, images
            )
            written = {path.name for path in Path(directory).iterdir()}
            saved_columns = list(
                pd.read_csv(paths["plaque_measurements_csv"]).columns
            )

        self.assertEqual(
            written,
            {
                "primary_object_measurements.csv",
                "config_used.yaml",
                "image_metadata.json",
            },
        )
        self.assertEqual(
            set(paths), {"plaque_measurements_csv", "config_used", "metadata"}
        )
        self.assertEqual(len(tables["primary_objects"]), 1)
        self.assertEqual(
            saved_columns,
            ["primary_object_id", "primary_object_area_um2"],
        )

if __name__ == "__main__":
    unittest.main()
