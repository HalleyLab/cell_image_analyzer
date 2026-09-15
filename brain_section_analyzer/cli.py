"""Command-line entry point for the cell fluorescence analyzer."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence

from cell_analyzer.image_io import inspect_image

from .analysis import run_analysis
from .batch import run_batch_analysis
from .config import create_default_config, load_config, save_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cell-fluorescence-analyzer",
        description="Analyze cellular Aβ and associated Iba1/CD68 fluorescence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Show image metadata and channels.")
    inspect_parser.add_argument("image")

    init_parser = subparsers.add_parser("init-config", help="Create an editable YAML config.")
    init_parser.add_argument("image")
    init_parser.add_argument("--output", required=True)
    init_parser.add_argument("--result-dir")

    run_parser = subparsers.add_parser("run", help="Analyze one image from a YAML config.")
    run_parser.add_argument("--config", required=True)

    batch_parser = subparsers.add_parser("batch", help="Analyze multiple images.")
    batch_parser.add_argument("--config", required=True)
    batch_parser.add_argument("--output-root", required=True)
    batch_parser.add_argument("images", nargs="+")

    session_parser = subparsers.add_parser(
        "session", help="Run a GUI-saved session, including its image paths and parameters."
    )
    session_parser.add_argument("--session", required=True)

    subparsers.add_parser("gui", help="Open the standalone graphical interface.")
    return parser


def run_saved_session(session_path: str | Path) -> dict:
    document = load_config(session_path)
    config = copy.deepcopy(document.get("template_config", document))
    paths = document.get("selected_image_files") or [config.get("input", {}).get("image_path")]
    paths = [path for path in paths if path]
    if not paths:
        raise ValueError("The session contains no image paths.")
    output_root = document.get("output_root")
    if not output_root:
        configured = config.get("input", {}).get("output_dir")
        output_root = str(Path(configured).expanduser().resolve().parent) if configured else "cell_analysis_results"
    per_file = document.get("effective_file_configs") or document.get("per_file_overrides") or None
    return run_batch_analysis(paths, config, output_root, per_file_configs=per_file)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(inspect_image(args.image).to_dict(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "init-config":
        config = create_default_config(args.image, output_dir=args.result_dir)
        save_config(config, args.output)
        print(Path(args.output).expanduser().resolve())
        return 0
    if args.command == "run":
        result = run_analysis(args.config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "batch":
        result = run_batch_analysis(args.images, args.config, args.output_root)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "session":
        result = run_saved_session(args.session)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "gui":
        from .gui import main as gui_main

        gui_main()
        return 0
    return 1
