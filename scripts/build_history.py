from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.history import build_history_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga fuentes historicas e importa partidos para entrenamiento de modelos."
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "quiniela.db"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument(
        "--sources-config",
        default=str(PROJECT_ROOT / "configs" / "history_sources.json"),
        help="Configuracion de fuentes historicas.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "history"),
        help="Carpeta para guardar CSVs historicos descargados.",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        help="source_id a importar. Puede repetirse. Si se omite, usa ingest_enabled=true.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_history_layer(
        db_path=Path(args.db),
        project_root=PROJECT_ROOT,
        sources_config_path=Path(args.sources_config),
        raw_dir=Path(args.raw_dir),
        source_ids=args.sources,
    )
    print(f"history_run_id: {result.history_run_id}")
    print(f"as_of_utc: {result.as_of_utc}")
    print(f"sources_checked: {result.sources_checked}")
    print(f"files_downloaded: {result.files_downloaded}")
    print(f"matches_imported: {result.matches_imported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
