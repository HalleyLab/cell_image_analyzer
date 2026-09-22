"""Run with python -m unittest brain_section_analyzer.tests.test_gui."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from tkinter import ttk

from tkinterdnd2 import TkinterDnD

from brain_section_analyzer.gui import BrainSectionGui, _bind_drag_toggle


class SessionButtonTests(unittest.TestCase):
    def test_selection_list_drag_paints_select_and_clear(self):
        class Listing:
            def __init__(self):
                self.selected = set()
                self.bindings = {}

            def bind(self, event, callback):
                self.bindings[event] = callback

            def nearest(self, y):
                return int(y)

            def selection_includes(self, index):
                return index in self.selected

            def selection_set(self, first, last):
                self.selected.update(range(first, last + 1))

            def selection_clear(self, first, last):
                self.selected.difference_update(range(first, last + 1))

        listing = Listing()
        start, move = _bind_drag_toggle(listing)
        start(SimpleNamespace(y=0))
        move(SimpleNamespace(y=2))
        self.assertEqual(listing.selected, {0, 1, 2})

        start(SimpleNamespace(y=1))
        move(SimpleNamespace(y=3))
        self.assertEqual(listing.selected, {0})

    def test_compact_sections_expand_without_losing_controls(self):
        root = TkinterDnD.Tk()
        root.withdraw()
        try:
            gui = BrainSectionGui(root)
            self.assertEqual(int(gui.file_list.cget("height")), 4)
            self.assertFalse(gui.calibration_visible)
            self.assertFalse(gui.microglia_vars["enabled"].get())
            self.assertFalse(gui.log_visible)
            self.assertEqual(gui.log.winfo_manager(), "")
            self.assertEqual(
                [gui.output_settings.tab(tab, "text") for tab in gui.output_settings.tabs()],
                ["Tables", "Images"],
            )
            gui._toggle_calibration()
            gui.microglia_vars["enabled"].set(True)
            gui._update_cell_count_visibility()
            gui._toggle_log()
            self.assertTrue(gui.calibration_visible)
            self.assertTrue(gui.log_visible)
            self.assertEqual(gui.log.winfo_manager(), "pack")
        finally:
            root.destroy()

    def test_session_buttons_keep_symmetric_centered_layout_at_different_scales(self):
        for scaling in (1.0, 1.333333333, 2.0):
            with self.subTest(scaling=scaling):
                root = TkinterDnD.Tk()
                root.withdraw()
                try:
                    root.tk.call("tk", "scaling", scaling)
                    gui = BrainSectionGui(root)
                    root.update_idletasks()
                    style = ttk.Style(root)
                    self.assertEqual(len(gui.session_buttons), 3)
                    self.assertEqual(style.lookup("Session.TButton", "anchor"), "center")
                    self.assertEqual(str(style.lookup("Session.TButton", "justify")), "center")
                    padding = root.tk.splitlist(style.lookup("Session.TButton", "padding"))
                    self.assertEqual(tuple(int(str(value)) for value in padding), (12, 8, 12, 8))
                    label = style.layout("Session.TButton")[0][1]["children"][0][1]["children"][0][1]["children"][0]
                    self.assertEqual(label, ("Button.label", {"sticky": ""}))
                    self.assertEqual(len({b.winfo_reqwidth() for b in gui.session_buttons}), 1)
                finally:
                    root.destroy()


    def test_preview_ad_navigation_wraps_and_does_not_intercept_parameter_typing(self):
        root = TkinterDnD.Tk()
        root.withdraw()
        try:
            gui = BrainSectionGui(root)
            gui.notebook.select(gui.output_tab)
            gui.preview_files = {"First": None, "Second": None, "Third": None}
            gui.preview_choice.set("First")
            gui._show_selected_preview = Mock()
            event = SimpleNamespace(keysym="d", state=0, widget=gui.preview_label)
            self.assertEqual(gui._preview_key(event), "break")
            self.assertEqual(gui.preview_choice.get(), "Second")
            event.keysym = "A"
            gui._preview_key(event)
            self.assertEqual(gui.preview_choice.get(), "First")
            gui._preview_key(event)
            self.assertEqual(gui.preview_choice.get(), "Third")
            event.keysym = "d"
            event.widget = gui.preview_selector
            gui._preview_key(event)
            self.assertEqual(gui.preview_choice.get(), "First")
            entry = ttk.Entry(root)
            event.widget = entry
            self.assertIsNone(gui._preview_key(event))
            self.assertEqual(gui.preview_choice.get(), "First")
            event.widget = gui.preview_label
            event.state = 0x4
            self.assertIsNone(gui._preview_key(event))
            event.state = 0
            gui.notebook.select(0)
            self.assertIsNone(gui._preview_key(event))
            gui.notebook.select(gui.output_tab)
            gui.preview_files = {}
            self.assertIsNone(gui._preview_key(event))
            self.assertEqual(gui._show_selected_preview.call_count, 4)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
