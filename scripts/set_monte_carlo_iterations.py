from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "bayesian_monte_carlo_scoreline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Actualiza las iteraciones del modelo bayesian_monte_carlo_scoreline."
    )
    parser.add_argument(
        "--models-config",
        default=str(PROJECT_ROOT / "configs" / "models.yaml"),
        help="Ruta a configs/models.yaml.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=None,
        help="Iteraciones para prediccion diaria.",
    )
    parser.add_argument(
        "--backtest-num-simulations",
        type=int,
        default=None,
        help="Iteraciones por partido durante backtest.",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=None,
        help="Multiplica los valores actuales por este factor.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.models_config)
    config = json.loads(path.read_text(encoding="utf-8"))
    model = _find_model(config)

    if args.multiplier is not None:
        model["num_simulations"] = _positive_int(round(int(model.get("num_simulations", 20000)) * args.multiplier))
        model["backtest_num_simulations"] = _positive_int(
            round(int(model.get("backtest_num_simulations", 5000)) * args.multiplier)
        )
    if args.num_simulations is not None:
        model["num_simulations"] = _positive_int(args.num_simulations)
    if args.backtest_num_simulations is not None:
        model["backtest_num_simulations"] = _positive_int(args.backtest_num_simulations)

    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{MODEL_ID}: num_simulations={model['num_simulations']}")
    print(f"{MODEL_ID}: backtest_num_simulations={model['backtest_num_simulations']}")
    return 0


def _find_model(config: dict[str, Any]) -> dict[str, Any]:
    for model in config.get("models", []):
        if model.get("model_id") == MODEL_ID:
            return model
    raise RuntimeError(f"No existe {MODEL_ID} en la configuracion.")


def _positive_int(value: int | float) -> int:
    value = int(value)
    if value < 500:
        raise RuntimeError("Usa al menos 500 simulaciones.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

