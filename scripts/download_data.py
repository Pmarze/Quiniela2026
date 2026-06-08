from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.data.pipeline import run_download_pipeline
from quiniela.storage.sqlite_store import print_database_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga snapshots de datos del Mundial 2026 y los normaliza en SQLite."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "sources.json"),
        help="Ruta al archivo de fuentes JSON.",
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "quiniela.db"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument(
        "--as-of-utc",
        default=None,
        help="Corte temporal ISO-8601. Si se omite, usa la hora UTC actual.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Fuente especifica a ejecutar. Puede repetirse.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="No descarga datos; solo imprime resumen de la base SQLite.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)

    if args.summary_only:
        print_database_summary(db_path)
        return 0

    result = run_download_pipeline(
        config_path=Path(args.config),
        db_path=db_path,
        project_root=PROJECT_ROOT,
        as_of_utc=args.as_of_utc,
        source_filter=set(args.source) if args.source else None,
    )

    print(f"run_id: {result.run_id}")
    print(f"as_of_utc: {result.as_of_utc}")
    print(f"snapshots: {result.snapshots_written}")
    print(f"errors: {len(result.errors)}")
    for error in result.errors:
        print(f"- {error}")
    print_database_summary(db_path)
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

