"""
Lee la quiniela de amigos desde Google Sheets y genera data/ui/friends_quinielas.json.

Estructura esperada en la hoja:
  - Una fila de cabecera con "#" en la primera columna.
  - Columnas 1-3: Partido, Grupo/Fase, Fecha  (se ignoran)
  - Columnas 4+: una columna por participante; el nombre de la columna es el nombre que se muestra.
  - Filas de datos: columna 0 = número de partido (entero), columnas 4+ = pronóstico "G-G".
  - Filas separadoras (fase, título, instrucciones): se ignoran automáticamente.

El nombre de cada participante se lee del encabezado en cada ejecución,
por lo que renombrar una columna en la hoja se refleja en el dashboard sin tocar el código.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH  = PROJECT_ROOT / "data" / "ui" / "friends_quinielas.json"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "friends_sheet.local.json"

PICK_RE = re.compile(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$")


def _slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "participante"


def _load_sheet_config() -> dict[str, str]:
    local_config: dict[str, str] = {}
    if LOCAL_CONFIG_PATH.exists():
        local_config = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))

    sheet_id = (
        os.environ.get("QUINIELA_FRIENDS_SHEET_ID")
        or local_config.get("sheet_id")
        or ""
    ).strip()
    if not sheet_id:
        raise RuntimeError(
            "Falta configurar QUINIELA_FRIENDS_SHEET_ID o configs/friends_sheet.local.json."
        )

    sheet_gid = (
        os.environ.get("QUINIELA_FRIENDS_SHEET_GID")
        or local_config.get("gid")
        or ""
    ).strip()
    sheet_name = (
        os.environ.get("QUINIELA_FRIENDS_SHEET_NAME")
        or local_config.get("sheet_name")
        or ""
    ).strip()
    return {"sheet_id": sheet_id, "gid": sheet_gid, "sheet_name": sheet_name}


def _sheet_url(config: dict[str, str]) -> str:
    url = f"https://docs.google.com/spreadsheets/d/{config['sheet_id']}/export?format=csv"
    if config.get("gid"):
        url += "&gid=" + urllib.parse.quote(config["gid"])
    elif config.get("sheet_name"):
        url += "&sheet=" + urllib.parse.quote(config["sheet_name"])
    return url


def _fetch_sheet(sheet_url: str) -> list[list[str]]:
    req = urllib.request.Request(sheet_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        content = resp.read().decode("utf-8")
    return list(csv.reader(io.StringIO(content)))


def _parse_sheet(rows: list[list[str]]) -> list[dict]:
    # Encontrar la fila de cabecera: la que tiene "#" en la primera columna
    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "#":
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("No se encontró la fila de cabecera ('#') en la hoja.")

    header = rows[header_idx]
    # Columnas de participantes: índice 4 en adelante, ignorar columnas vacías
    participant_cols = [
        (col_idx, header[col_idx].strip())
        for col_idx in range(4, len(header))
        if header[col_idx].strip()
    ]

    if not participant_cols:
        print("  [warn] No hay columnas de participantes en la cabecera.")
        return []

    print(f"  Participantes encontrados: {[name for _, name in participant_cols]}")

    # Inicializar estructura de picks por participante
    friends: dict[int, dict] = {
        col_idx: {"name": name, "picks": {}}
        for col_idx, name in participant_cols
    }

    for row in rows[header_idx + 1:]:
        if not row or not row[0].strip():
            continue
        # Solo procesar filas cuya primera columna sea un número de partido
        try:
            match_id = str(int(row[0].strip()))
        except ValueError:
            continue

        for col_idx, _ in participant_cols:
            raw = row[col_idx].strip() if col_idx < len(row) else ""
            if not raw:
                continue
            m = PICK_RE.match(raw)
            if m:
                friends[col_idx]["picks"][match_id] = f"{m.group(1)}-{m.group(2)}"
            else:
                print(f"  [warn] Partido {match_id}, '{friends[col_idx]['name']}': "
                      f"formato no reconocido '{raw}' (esperado G-G)")

    # Construir lista final con IDs únicos
    result: list[dict] = []
    seen_ids: set[str] = set()
    for col_idx, name in participant_cols:
        base = _slugify(name)
        uid = base
        n = 2
        while uid in seen_ids:
            uid = f"{base}-{n}"
            n += 1
        seen_ids.add(uid)
        picks = friends[col_idx]["picks"]
        result.append({"id": uid, "name": name, "picks": picks, "source": "google_sheets"})
        print(f"    {name}: {len(picks)} pronósticos")

    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write(data: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    try:
        sheet_config = _load_sheet_config()
    except Exception as e:
        print(f"  ERROR de configuracion de Google Sheets: {e}")
        return 1

    print("Descargando hoja desde Google Sheets (fuente privada configurada localmente)...")
    try:
        rows = _fetch_sheet(_sheet_url(sheet_config))
    except Exception as e:
        print(f"  ERROR al descargar la hoja: {e}")
        return 1

    print(f"  {len(rows)} filas descargadas.")

    try:
        friends = _parse_sheet(rows)
    except Exception as e:
        print(f"  ERROR al procesar la hoja: {e}")
        return 1

    _write({"generated_at": _now(), "friends": friends})
    print(f"\nGenerado: {OUTPUT_PATH}")
    print(f"Participantes: {len(friends)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
