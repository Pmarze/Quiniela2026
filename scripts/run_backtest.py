from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.backtest import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta backtest historico walk-forward para validar modelos de quiniela."
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "quiniela.db"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument(
        "--backtest-config",
        default=str(PROJECT_ROOT / "configs" / "backtest.yaml"),
        help="Configuracion del backtest.",
    )
    parser.add_argument(
        "--models-config",
        default=str(PROJECT_ROOT / "configs" / "models.yaml"),
        help="Configuracion de modelos.",
    )
    parser.add_argument(
        "--scoring-config",
        default=str(PROJECT_ROOT / "configs" / "scoring.yaml"),
        help="Reglas de puntaje de quiniela.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "backtests"),
        help="Carpeta raiz para artefactos de backtest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_backtest(
        db_path=Path(args.db),
        project_root=PROJECT_ROOT,
        backtest_config_path=Path(args.backtest_config),
        models_config_path=Path(args.models_config),
        scoring_config_path=Path(args.scoring_config),
        output_root=Path(args.output_root),
    )
    print(f"backtest_run_id: {result.backtest_run_id}")
    print(f"years: {', '.join(str(year) for year in result.years)}")
    print(f"models: {', '.join(result.models)}")
    print(f"matches: {result.matches}")
    print(f"predictions: {result.predictions}")
    print(f"json: {result.output_json_path}")
    print(f"csv: {result.output_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
