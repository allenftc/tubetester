from __future__ import annotations

import argparse
from pathlib import Path

from controller.config.settings import load_settings
from controller.web.server import serve_control_server
from controller.workflow.state_machine import TubeScanWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test tube scanner controller")
    default_config_dir = Path(__file__).resolve().parents[1] / "calibration"
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=default_config_dir,
        help="Directory containing calibration JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned workflow and exit.",
    )
    parser.add_argument(
        "--serve-web",
        action="store_true",
        help="Start the built-in web control surface.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_settings(args.config_dir)
    workflow = TubeScanWorkflow(settings)

    if args.dry_run:
        for line in workflow.describe():
            print(line)
        return 0

    if args.serve_web:
        serve_control_server(settings)
        return 0

    print("Controller scaffold is loaded.")
    print("Use --dry-run to inspect the planned scan sequence.")
    print("Use --serve-web to start the browser control surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())