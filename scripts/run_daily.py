from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.canonical import build_canonical_dataset
from quiniela.data.pipeline import run_download_pipeline
from quiniela.state import build_tournament_state
from quiniela.ui import generate_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el flujo diario: descarga, canon, estado y dashboard."
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "quiniela.db"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument(
        "--sources-config",
        default=str(PROJECT_ROOT / "configs" / "sources.json"),
        help="Config de fuentes de datos.",
    )
    parser.add_argument(
        "--timezone-config",
        default=str(PROJECT_ROOT / "configs" / "stadium_timezones.json"),
        help="Config de zonas horarias por estadio.",
    )
    parser.add_argument(
        "--as-of-utc",
        default=None,
        help="Corte temporal ISO-8601 para descarga. Si se omite, usa UTC actual.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Fuente especifica a descargar. Puede repetirse.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Omite descarga y usa la ultima ingesta completada.",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Omite generacion del dashboard HTML.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)

    print("== Quiniela2026 daily run ==")
    source_run_id = None

    if args.skip_download:
        print("download: skipped")
    else:
        download = run_download_pipeline(
            config_path=Path(args.sources_config),
            db_path=db_path,
            project_root=PROJECT_ROOT,
            as_of_utc=args.as_of_utc,
            source_filter=set(args.source) if args.source else None,
        )
        print(f"download.run_id: {download.run_id}")
        print(f"download.as_of_utc: {download.as_of_utc}")
        print(f"download.snapshots: {download.snapshots_written}")
        if download.errors:
            print(f"download.errors: {len(download.errors)}")
            for error in download.errors:
                print(f"- {error}")
            return 1
        source_run_id = download.run_id

    canonical = build_canonical_dataset(
        db_path=db_path,
        project_root=PROJECT_ROOT,
        timezone_config_path=Path(args.timezone_config),
        source_run_id=source_run_id,
    )
    print(f"canonical.run_id: {canonical.canonical_run_id}")
    print(f"canonical.teams: {canonical.teams}")
    print(f"canonical.matches: {canonical.matches}")
    print(f"reconciliation.run_id: {canonical.reconciliation_run_id}")
    print(f"reconciliation.issues: {canonical.reconciliation_issues}")

    state = build_tournament_state(
        db_path=db_path,
        project_root=PROJECT_ROOT,
        source_run_id=canonical.source_run_id,
        as_of_utc=canonical.as_of_utc,
    )
    print(f"state.state_id: {state.state_id}")
    print(f"state.completed: {state.completed_matches}")
    print(f"state.pending: {state.pending_matches}")

    if args.skip_dashboard:
        print("dashboard: skipped")
    else:
        dashboard = generate_dashboard(db_path=db_path, project_root=PROJECT_ROOT)
        print(f"dashboard.output: {dashboard.output_path}")
        print(f"dashboard.state_id: {dashboard.state_id}")

    print("daily run: completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

