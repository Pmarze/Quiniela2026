from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


STAGE_COMMANDS: dict[str, list[list[str]]] = {
    "download": [["scripts/download_data.py"]],
    "history": [["scripts/build_history.py"]],
    "state": [["scripts/run_daily.py", "--skip-download", "--skip-dashboard"]],
    "backtest": [["scripts/run_backtest.py"]],
    "predictions": [["scripts/run_model.py"]],
    "dashboard": [["scripts/generate_dashboard.py"], ["scripts/generate_validation_dashboard.py"]],
}

PRESETS = {
    "base": ["download", "history", "state"],
    "modeling": ["download", "history", "state", "backtest", "predictions"],
    "all": ["download", "history", "state", "backtest", "predictions", "dashboard"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruye artefactos locales descargables/generados que no se versionan en Git."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="base",
        help="Conjunto de etapas a ejecutar.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=sorted(STAGE_COMMANDS),
        help="Etapa especifica. Si se usa, ignora --preset. Puede repetirse.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime comandos.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = args.stage or PRESETS[args.preset]
    commands = [command for stage in stages for command in STAGE_COMMANDS[stage]]

    for command in commands:
        full_command = [PYTHON, *command]
        print("$ " + " ".join(full_command), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(full_command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
