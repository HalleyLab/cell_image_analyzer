"""Command-line interface for microscopy image inspection, configuration, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import create_default_config, save_yaml
from .image_io import inspect_image
from .pipeline import run_analysis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cell-analyzer",
        description="Segment cell ROIs in microscopy images and measure all channels.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print image metadata as JSON.")
    inspect_parser.add_argument("czi_path", type=Path)

    config_parser = subparsers.add_parser(
        "init-config", help="Create an editable YAML configuration from an image file."
    )
    config_parser.add_argument("czi_path", type=Path)
    config_parser.add_argument("--output", type=Path, default=Path("config.yaml"))
    config_parser.add_argument("--result-dir", type=Path, default=None)
    config_parser.add_argument("--zoom", type=float, default=1.0)

    run_parser = subparsers.add_parser("run", help="Run an analysis from YAML.")
    run_parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(inspect_image(args.czi_path).to_dict(), indent=2))
        return 0
    if args.command == "init-config":
        info = inspect_image(args.czi_path)
        config = create_default_config(info, args.result_dir, zoom=args.zoom)
        save_yaml(config, args.output)
        print(f"Configuration written to {args.output.resolve()}")
        return 0
    if args.command == "run":
        result = run_analysis(args.config)
        print(json.dumps(result, indent=2))
        return 0
    return 2
