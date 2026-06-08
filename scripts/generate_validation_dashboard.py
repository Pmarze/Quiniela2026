from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.backtest.dashboard import generate_validation_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera dashboard local de validacion historica de modelos."
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "quiniela.db"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "validation_dashboard" / "index.html"),
        help="Ruta del HTML generado.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = generate_validation_dashboard(
        db_path=Path(args.db),
        project_root=PROJECT_ROOT,
        output_path=Path(args.output),
    )
    print(f"validation_dashboard: {result.output_path}")
    print(f"backtest_run_id: {result.backtest_run_id}")
    print(f"matches: {result.matches}")
    print(f"predictions: {result.predictions}")
    print(f"models: {result.models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
