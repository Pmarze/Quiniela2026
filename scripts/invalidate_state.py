from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marca un estado del torneo como invalidado.")
    parser.add_argument("state_id", help="Identificador del estado a invalidar.")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "quiniela.db"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument(
        "--notes",
        default="invalidated manually",
        help="Nota de auditoria para explicar la invalidacion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "UPDATE tournament_state_runs SET status = ?, notes = ? WHERE state_id = ?",
            ("invalidated", args.notes, args.state_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
            print(f"No existe state_id: {args.state_id}")
            return 1
        print(f"Estado invalidado: {args.state_id}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

