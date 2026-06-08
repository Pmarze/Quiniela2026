from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.training.neural_hybrid_trainer import train_neural_hybrid_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo neural_hybrid_v2.")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "quiniela.db"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "neural_hybrid_v2.yaml"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "models" / "neural_hybrid_v2"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--folds-only", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = train_neural_hybrid_v2(
        db_path=Path(args.db),
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        device_name=args.device,
        folds_only=args.folds_only,
        final_only=args.final_only,
        resume=not args.fresh,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
