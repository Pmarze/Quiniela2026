from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sqlite3
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from quiniela.data.http_client import fetch_url
from quiniela.storage.sqlite_store import SQLiteStore


PRIMARY_HISTORY_SOURCE = "martj42_international_results"

FIFA_CODE_HISTORY_ALIASES = {
    "USA": ("United States", "United States of America", "USMNT"),
    "KOR": ("South Korea", "Korea Republic", "Republic of Korea"),
    "CIV": ("Ivory Coast", "Cote d'Ivoire", "Cote d Ivoire"),
    "COD": ("DR Congo", "Congo DR", "Democratic Republic of the Congo"),
    "CUW": ("Curacao",),
    "CPV": ("Cape Verde",),
    "NED": ("Netherlands", "Holland"),
    "IRN": ("Iran",),
}

TEAM_NORMALIZATION_ALIASES = {
    "usa": "united states",
    "u s a": "united states",
    "us": "united states",
    "united states of america": "united states",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "dr congo": "democratic republic of the congo",
    "congo dr": "democratic republic of the congo",
    "democratic republic congo": "democratic republic of the congo",
    "curacao": "curacao",
}

CONTINENTAL_TOURNAMENT_TERMS = (
    "africa cup of nations",
    "afc asian cup",
    "copa america",
    "concacaf gold cup",
    "uefa euro",
    "uefa nations league",
    "concacaf nations league",
    "ofc nations cup",
)


@dataclass(frozen=True)
class HistoryBuildResult:
    history_run_id: str
    as_of_utc: str
    sources_checked: int
    files_downloaded: int
    matches_imported: int


def build_history_layer(
    db_path: Path,
    project_root: Path,
    sources_config_path: Path | None = None,
    raw_dir: Path | None = None,
    source_ids: list[str] | None = None,
) -> HistoryBuildResult:
    as_of_utc = _utc_now()
    history_run_id = f"history_{_compact_timestamp(as_of_utc)}_{uuid.uuid4().hex[:8]}"
    config_path = sources_config_path or project_root / "configs" / "history_sources.json"
    raw_root = raw_dir or project_root / "data" / "raw" / "history"

    store = SQLiteStore(db_path)
    store.initialize()
    conn = store.conn
    try:
        _start_history_run(conn, history_run_id, as_of_utc)
        try:
            sources = _select_sources(_load_sources(config_path), source_ids)
            file_records: list[dict[str, Any]] = []
            downloaded_paths: dict[tuple[str, str], Path] = {}
            for source in sources:
                if source["source_id"] != PRIMARY_HISTORY_SOURCE:
                    continue
                records, paths = _download_martj42_files(source, history_run_id, as_of_utc, raw_root)
                file_records.extend(records)
                downloaded_paths.update(paths)

            if file_records:
                _insert_source_files(conn, file_records)

            results_path = downloaded_paths.get((PRIMARY_HISTORY_SOURCE, "results_csv"))
            if results_path is None:
                raise RuntimeError("No se descargo results.csv de martj42; no se puede construir historico.")

            matches_imported = _import_martj42_results(
                conn=conn,
                history_run_id=history_run_id,
                as_of_utc=as_of_utc,
                results_path=results_path,
            )
            _finish_history_run(
                conn=conn,
                history_run_id=history_run_id,
                status="completed",
                sources_checked=len(sources),
                files_downloaded=len(file_records),
                matches_imported=matches_imported,
                notes="historical layer built from martj42 international_results",
            )
            return HistoryBuildResult(
                history_run_id=history_run_id,
                as_of_utc=as_of_utc,
                sources_checked=len(sources),
                files_downloaded=len(file_records),
                matches_imported=matches_imported,
            )
        except Exception as exc:
            _finish_history_run(
                conn=conn,
                history_run_id=history_run_id,
                status="failed",
                sources_checked=0,
                files_downloaded=0,
                matches_imported=0,
                notes=str(exc),
            )
            raise
    finally:
        store.close()


def _load_sources(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"No existe la configuracion de fuentes historicas: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("history_sources", []))


def _select_sources(sources: list[dict[str, Any]], source_ids: list[str] | None) -> list[dict[str, Any]]:
    if source_ids:
        requested = set(source_ids)
        selected = [source for source in sources if source.get("source_id") in requested]
    else:
        selected = [source for source in sources if source.get("enabled") and source.get("ingest_enabled")]
    if not selected:
        raise RuntimeError("No hay fuentes historicas seleccionadas para ingesta.")
    unsupported = [source["source_id"] for source in selected if source["source_id"] != PRIMARY_HISTORY_SOURCE]
    if unsupported:
        raise RuntimeError(f"Fuentes historicas aun no implementadas para ingesta: {', '.join(unsupported)}")
    return selected


def _download_martj42_files(
    source: dict[str, Any],
    history_run_id: str,
    as_of_utc: str,
    raw_root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], Path]]:
    source_id = source["source_id"]
    urls = source.get("urls", {})
    resources = {
        resource_name: url
        for resource_name, url in urls.items()
        if resource_name.endswith("_csv") and isinstance(url, str)
    }
    if "results_csv" not in resources:
        raise RuntimeError("La fuente martj42 no tiene configurado results_csv.")

    records = []
    paths = {}
    target_dir = raw_root / history_run_id / source_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for resource_name, url in resources.items():
        response = fetch_url(url)
        if response.status >= 400:
            raise RuntimeError(f"Descarga fallida {url}: HTTP {response.status}")
        filename = _filename_from_url(url, f"{resource_name}.csv")
        raw_path = target_dir / filename
        raw_path.write_bytes(response.body)
        text = response.body.decode("utf-8-sig")
        record = {
            "history_run_id": history_run_id,
            "source_id": source_id,
            "resource_name": resource_name,
            "url": url,
            "downloaded_at_utc": _utc_now(),
            "content_sha256": hashlib.sha256(response.body).hexdigest(),
            "byte_count": len(response.body),
            "row_count": _count_csv_rows(text),
            "raw_path": str(raw_path),
            "status": "downloaded",
        }
        records.append(record)
        paths[(source_id, resource_name)] = raw_path
    return records, paths


def _import_martj42_results(
    conn: sqlite3.Connection,
    history_run_id: str,
    as_of_utc: str,
    results_path: Path,
) -> int:
    team_aliases = _load_current_team_aliases(conn)
    as_of_date = datetime.fromisoformat(as_of_utc.replace("Z", "+00:00")).date()
    created_at = _utc_now()
    rows = []
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_results_columns(reader.fieldnames or [])
        for source_index, row in enumerate(reader, start=1):
            parsed = _normalize_martj42_row(
                row=row,
                source_index=source_index,
                history_run_id=history_run_id,
                as_of_date=as_of_date,
                created_at=created_at,
                team_aliases=team_aliases,
            )
            if parsed is not None:
                rows.append(parsed)

    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO canonical_historical_matches (
                historical_match_id, source_id, source_match_key, match_date,
                team_a_name, team_b_name, team_a_canonical_id, team_b_canonical_id,
                home_score, away_score, tournament, city, country, neutral,
                result_1x2, goal_difference, total_goals, is_world_cup,
                is_qualifier, is_friendly, importance_weight, recency_weight,
                history_run_id, created_at_utc
            )
            VALUES (
                :historical_match_id, :source_id, :source_match_key, :match_date,
                :team_a_name, :team_b_name, :team_a_canonical_id, :team_b_canonical_id,
                :home_score, :away_score, :tournament, :city, :country, :neutral,
                :result_1x2, :goal_difference, :total_goals, :is_world_cup,
                :is_qualifier, :is_friendly, :importance_weight, :recency_weight,
                :history_run_id, :created_at_utc
            )
            """,
            rows,
        )
    return len(rows)


def _normalize_martj42_row(
    row: dict[str, str],
    source_index: int,
    history_run_id: str,
    as_of_date: date,
    created_at: str,
    team_aliases: dict[str, str],
) -> dict[str, Any] | None:
    try:
        match_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
    except (KeyError, TypeError, ValueError):
        return None

    home_team = row["home_team"].strip()
    away_team = row["away_team"].strip()
    tournament = row.get("tournament", "").strip()
    city = row.get("city", "").strip()
    country = row.get("country", "").strip()
    neutral = _parse_bool(row.get("neutral"))
    source_match_key = "|".join(
        (
            str(source_index),
            match_date.isoformat(),
            home_team,
            away_team,
            tournament,
            city,
            country,
        )
    )
    historical_match_id = f"martj42_{hashlib.sha1(source_match_key.encode('utf-8')).hexdigest()[:16]}"
    flags = _classify_tournament(tournament)
    goal_difference = home_score - away_score
    return {
        "historical_match_id": historical_match_id,
        "source_id": PRIMARY_HISTORY_SOURCE,
        "source_match_key": source_match_key,
        "match_date": match_date.isoformat(),
        "team_a_name": home_team,
        "team_b_name": away_team,
        "team_a_canonical_id": team_aliases.get(_normalize_name(home_team)),
        "team_b_canonical_id": team_aliases.get(_normalize_name(away_team)),
        "home_score": home_score,
        "away_score": away_score,
        "tournament": tournament,
        "city": city,
        "country": country,
        "neutral": neutral,
        "result_1x2": _result_1x2(home_score, away_score),
        "goal_difference": goal_difference,
        "total_goals": home_score + away_score,
        "is_world_cup": flags["is_world_cup"],
        "is_qualifier": flags["is_qualifier"],
        "is_friendly": flags["is_friendly"],
        "importance_weight": flags["importance_weight"],
        "recency_weight": _recency_weight(match_date, as_of_date),
        "history_run_id": history_run_id,
        "created_at_utc": created_at,
    }


def _load_current_team_aliases(conn: sqlite3.Connection) -> dict[str, str]:
    aliases: dict[str, str] = {}
    try:
        rows = conn.execute(
            """
            SELECT canonical_team_id, display_name, fifa_code, aliases_json
            FROM v_canonical_teams
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return aliases

    for row in rows:
        canonical_id = row["canonical_team_id"]
        values = [row["display_name"], row["fifa_code"]]
        if row["aliases_json"]:
            values.extend(json.loads(row["aliases_json"]))
        if row["fifa_code"] in FIFA_CODE_HISTORY_ALIASES:
            values.extend(FIFA_CODE_HISTORY_ALIASES[row["fifa_code"]])
        for value in values:
            normalized = _normalize_name(value)
            if normalized:
                aliases[normalized] = canonical_id
    return aliases


def _validate_results_columns(columns: list[str]) -> None:
    expected = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing = expected - set(columns)
    if missing:
        raise RuntimeError(f"results.csv no tiene columnas esperadas: {', '.join(sorted(missing))}")


def _classify_tournament(tournament: str) -> dict[str, int | float]:
    normalized = _normalize_name(tournament)
    is_qualifier = int("qualification" in normalized or "qualifier" in normalized)
    is_friendly = int("friendly" in normalized)
    is_world_cup = int("world cup" in normalized and not is_qualifier and not is_friendly)
    if is_world_cup:
        importance = 1.6
    elif is_qualifier:
        importance = 1.3
    elif any(term in normalized for term in CONTINENTAL_TOURNAMENT_TERMS):
        importance = 1.2
    elif is_friendly:
        importance = 0.6
    else:
        importance = 1.0
    return {
        "is_world_cup": is_world_cup,
        "is_qualifier": is_qualifier,
        "is_friendly": is_friendly,
        "importance_weight": importance,
    }


def _recency_weight(match_date: date, as_of_date: date) -> float:
    age_days = max((as_of_date - match_date).days, 0)
    age_years = age_days / 365.25
    return round(max(0.05, math.exp(-age_years / 8.0)), 6)


def _insert_source_files(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO history_source_files (
                history_run_id, source_id, resource_name, url, downloaded_at_utc,
                content_sha256, byte_count, row_count, raw_path, status
            )
            VALUES (
                :history_run_id, :source_id, :resource_name, :url, :downloaded_at_utc,
                :content_sha256, :byte_count, :row_count, :raw_path, :status
            )
            """,
            rows,
        )


def _start_history_run(conn: sqlite3.Connection, history_run_id: str, as_of_utc: str) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO history_ingestion_runs (
                history_run_id, as_of_utc, started_at_utc, status,
                sources_checked, files_downloaded, matches_imported, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (history_run_id, as_of_utc, _utc_now(), "running", 0, 0, 0, None),
        )


def _finish_history_run(
    conn: sqlite3.Connection,
    history_run_id: str,
    status: str,
    sources_checked: int,
    files_downloaded: int,
    matches_imported: int,
    notes: str,
) -> None:
    with conn:
        conn.execute(
            """
            UPDATE history_ingestion_runs
            SET completed_at_utc = ?, status = ?, sources_checked = ?,
                files_downloaded = ?, matches_imported = ?, notes = ?
            WHERE history_run_id = ?
            """,
            (_utc_now(), status, sources_checked, files_downloaded, matches_imported, notes, history_run_id),
        )


def _count_csv_rows(text: str) -> int:
    reader = csv.reader(io.StringIO(text))
    row_count = sum(1 for _ in reader)
    return max(row_count - 1, 0)


def _filename_from_url(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or fallback


def _parse_bool(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"true", "1", "yes"}:
        return 1
    if text in {"false", "0", "no"}:
        return 0
    return None


def _result_1x2(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "1"
    if home_score < away_score:
        return "2"
    return "X"


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    ascii_text = ascii_text.lower().replace("&", " and ")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()
    ascii_text = re.sub(r"\s+", " ", ascii_text)
    return TEAM_NORMALIZATION_ALIASES.get(ascii_text, ascii_text)


def _compact_timestamp(value: str) -> str:
    return value.replace("-", "").replace(":", "").replace(".", "").replace("+00:00", "Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
