from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain_section_analyzer.cli import run_saved_session
from brain_section_analyzer.config import save_config


class DesktopSessionTest(unittest.TestCase):
    def test_saved_session_reuses_paths_parameters_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.czi"
            image.touch()
            session = root / "session.yaml"
            config = {"input": {"image_path": str(image)}, "plaque": {"min_area_um2": 12}}
            save_config(
                {
                    "selected_image_files": [str(image)],
                    "output_root": str(root / "results"),
                    "template_config": config,
                },
                session,
            )
            with patch("brain_section_analyzer.cli.run_batch_analysis", return_value={"completed": 1}) as run:
                result = run_saved_session(session)
            self.assertEqual(result, {"completed": 1})
            self.assertEqual(run.call_args.args[0], [str(image)])
            self.assertEqual(run.call_args.args[1]["plaque"]["min_area_um2"], 12)
            self.assertEqual(run.call_args.args[2], str(root / "results"))


if __name__ == "__main__":
    unittest.main()
