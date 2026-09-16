"""Run with python -m unittest brain_section_analyzer.tests.test_gui."""

import unittest
from tkinter import ttk

from tkinterdnd2 import TkinterDnD

from brain_section_analyzer.gui import BrainSectionGui


class SessionButtonTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
