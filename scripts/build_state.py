from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.state import build_tournament_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye el estado vivo del torneo desde la base SQLite."
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "quiniela.db"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="run_id de ingesta a usar. Si se omite, usa la ultima corrida completada.",
    )
    parser.add_argument(
        "--as-of-utc",
        default=None,
        help="Corte temporal ISO-8601. Si se omite, usa el as_of_utc de la corrida de ingesta.",
    )
    parser.add_argument(
        "--source-name",
        default="worldcup26_ir",
        help="Fuente operativa para estado. Default: worldcup26_ir.",
    )
    parser.add_argument(
        "--state-id",
        default=None,
        help="Identificador del estado. Si se omite, se genera automaticamente.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_tournament_state(
        db_path=Path(args.db),
        project_root=PROJECT_ROOT,
        source_run_id=args.source_run_id,
        as_of_utc=args.as_of_utc,
        source_name=args.source_name,
        state_id=args.state_id,
    )

    print(f"state_id: {result.state_id}")
    print(f"source_run_id: {result.source_run_id}")
    print(f"as_of_utc: {result.as_of_utc}")
    print(f"source_name: {result.source_name}")
    print(f"total_matches: {result.total_matches}")
    print(f"completed_matches: {result.completed_matches}")
    print(f"pending_matches: {result.pending_matches}")
    print(f"group_matches_completed: {result.group_matches_completed}")
    print(f"teams: {result.teams}")
    print(f"groups: {result.groups}")
    print(f"output_dir: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

