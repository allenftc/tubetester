from __future__ import annotations

import argparse
import os
import sys
import threading
import time
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
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart automatically when Python or calibration files change.",
    )
    return parser


def _snapshot_watch_files(root: Path) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".json"}:
            continue
        try:
            snapshot[str(path)] = path.stat().st_mtime
        except OSError:
            continue
    return snapshot


def _start_reload_watcher(root: Path, poll_interval: float = 1.0) -> None:
    initial_snapshot = _snapshot_watch_files(root)
    while True:
        time.sleep(poll_interval)
        current_snapshot = _snapshot_watch_files(root)
        if current_snapshot != initial_snapshot:
            print("Detected source or calibration change. Reloading...")
            os.execvp(sys.executable, [sys.executable, "-m", "controller.main", *sys.argv[1:]])


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
        if args.reload:
            repo_root = Path(__file__).resolve().parents[1]
            watcher = threading.Thread(
                target=_start_reload_watcher,
                args=(repo_root,),
                daemon=True,
            )
            watcher.start()
        serve_control_server(settings)
        return 0

    print("Controller scaffold is loaded.")
    print("Use --dry-run to inspect the planned scan sequence.")
    print("Use --serve-web to start the browser control surface.")
    print("Use --reload with --serve-web to restart automatically on source or config changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())