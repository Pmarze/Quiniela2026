from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.storage.sqlite_store import SQLiteStore


def main() -> int:
    db_path = PROJECT_ROOT / "data" / "quiniela.db"
    store = SQLiteStore(db_path)
    try:
        store.initialize()
    finally:
        store.close()
    print(f"Base inicializada: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

