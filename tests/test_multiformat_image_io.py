from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from cell_analyzer.image_io import inspect_image, read_image_channels


class MultiFormatImageIoTest(unittest.TestCase):
    def test_runtime_cache_and_temp_stay_with_project(self) -> None:
        expected = Path(__file__).resolve().parent.parent / ".cache"
        self.assertEqual(Path(tempfile.gettempdir()), expected / "tmp")
        for name in ("CELL_ANALYZER_CACHE_DIR", "CJDK_CACHE_DIR", "MPLCONFIGDIR"):
            configured = Path(os.environ[name]).resolve()
            self.assertTrue(configured.is_relative_to(expected), (name, configured))

    def test_multichannel_ome_tiff_metadata_and_projection(self) -> None:
        data = np.zeros((1, 2, 3, 8, 10), dtype=np.uint16)
        data[0, 0, 0] = 10
        data[0, 0, 1] = 30
        data[0, 0, 2] = 20
        data[0, 1, 0] = 7
        data[0, 1, 1] = 11
        data[0, 1, 2] = 15

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "two_channel.ome.tif"
            tifffile.imwrite(
                path,
                data,
                ome=True,
                metadata={
                    "axes": "TCZYX",
                    "PhysicalSizeX": 0.5,
                    "PhysicalSizeXUnit": "µm",
                    "PhysicalSizeY": 0.6,
                    "PhysicalSizeYUnit": "µm",
                    "Channel": {"Name": ["Abeta", "Iba1"]},
                },
            )

            info = inspect_image(path)
            self.assertEqual(info.dimensions["C"], (0, 2))
            self.assertEqual(info.dimensions["Z"], (0, 3))
            self.assertEqual([channel.name for channel in info.channels], ["Abeta", "Iba1"])
            self.assertAlmostEqual(info.pixel_size_um_x, 0.5)
            self.assertAlmostEqual(info.pixel_size_um_y, 0.6)

            images = read_image_channels(
                path,
                info,
                scene=0,
                time_index=0,
                z_projection="max",
                z_index=0,
                zoom=1.0,
                channel_indices=[0, 1],
            )
            np.testing.assert_array_equal(images[0], np.full((8, 10), 30, dtype=np.uint16))
            np.testing.assert_array_equal(images[1], np.full((8, 10), 15, dtype=np.uint16))


if __name__ == "__main__":
    unittest.main()
