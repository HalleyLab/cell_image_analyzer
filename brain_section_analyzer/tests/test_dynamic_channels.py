from __future__ import annotations

import copy
import unittest

import numpy as np
from tkinterdnd2 import TkinterDnD

from brain_section_analyzer.advanced import roles_for_count
from brain_section_analyzer.analysis import analyze_arrays
from brain_section_analyzer.config import DEFAULT_CONFIG, default_channel, normalize_config
from brain_section_analyzer.gui import BrainSectionGui
from cell_analyzer.models import ChannelInfo, CziInfo


class DynamicChannelTests(unittest.TestCase):
    def test_five_channels_and_conditional_advanced_panels(self):
        for count in (1, 3, 5):
            info = CziInfo(
                path=f"{count}-channel.ome.tif",
                dimensions={"C": (0, count), "X": (0, 12), "Y": (0, 12)},
                channels=[ChannelInfo(index=index, name=f"Channel {index + 1}") for index in range(count)],
                scenes=[],
            )
            normalized = normalize_config(copy.deepcopy(DEFAULT_CONFIG), info)
            self.assertEqual(tuple(normalized["channels"]), roles_for_count(count))

        roles = roles_for_count(5)
        info = CziInfo(
            path="five-channel.ome.tif",
            dimensions={"C": (0, 5), "X": (0, 12), "Y": (0, 12)},
            channels=[ChannelInfo(index=index, name=f"Channel {index + 1}") for index in range(5)],
            scenes=[],
        )
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["channels"] = {
            role: default_channel(role, index) for index, role in enumerate(roles)
        }
        for item in config["channels"].values():
            item["gaussian_sigma_px"] = 0
            item["threshold"].update(method="manual", value=1)
        config = normalize_config(config, info)
        products = analyze_arrays(
            {role: np.zeros((12, 12), dtype=float) for role in roles},
            config,
            pixel_size_um_x=1,
            pixel_size_um_y=1,
        )
        self.assertEqual(list(products.positive_masks), list(roles))

        root = TkinterDnD.Tk()
        root.withdraw()
        try:
            gui = BrainSectionGui(root)
            gui._set_channel_count(5)
            self.assertEqual(len(gui.channel_vars), 5)
            self.assertEqual(gui.advanced_notebook.tabs(), ())
            gui.advanced_vars["radial_profiles_enabled"].set(True)
            gui._update_advanced_visibility()
            labels = [gui.advanced_notebook.tab(tab, "text") for tab in gui.advanced_notebook.tabs()]
            self.assertEqual(labels, ["Radial profiles"])
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
