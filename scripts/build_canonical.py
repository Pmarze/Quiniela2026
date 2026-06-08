from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.canonical import build_canonical_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye equipos/partidos canonicos, normaliza horarios y reconcilia fuentes."
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "quiniela.db"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument(
        "--timezone-config",
        default=str(PROJECT_ROOT / "configs" / "stadium_timezones.json"),
        help="Mapa de zonas horarias por estadio.",
    )
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="run_id de ingesta a usar. Si se omite, usa la ultima corrida completada.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_canonical_dataset(
        db_path=Path(args.db),
        project_root=PROJECT_ROOT,
        timezone_config_path=Path(args.timezone_config),
        source_run_id=args.source_run_id,
    )
    print(f"canonical_run_id: {result.canonical_run_id}")
    print(f"reconciliation_run_id: {result.reconciliation_run_id}")
    print(f"source_run_id: {result.source_run_id}")
    print(f"as_of_utc: {result.as_of_utc}")
    print(f"teams: {result.teams}")
    print(f"matches: {result.matches}")
    print(f"sources_checked: {result.sources_checked}")
    print(f"reconciliation_issues: {result.reconciliation_issues}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

