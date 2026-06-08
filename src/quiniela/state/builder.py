from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quiniela.storage.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class TournamentStateResult:
    state_id: str
    source_run_id: str
    as_of_utc: str
    source_name: str
    total_matches: int
    completed_matches: int
    pending_matches: int
    group_matches_completed: int
    teams: int
    groups: int
    output_dir: Path


def build_tournament_state(
    db_path: Path,
    project_root: Path,
    source_run_id: str | None = None,
    as_of_utc: str | None = None,
    source_name: str = "worldcup26_ir",
    state_id: str | None = None,
) -> TournamentStateResult:
    store = SQLiteStore(db_path)
    store.initialize()
    conn = store.conn
    try:
        run = _resolve_source_run(conn, source_run_id)
        resolved_source_run_id = run["run_id"]
        resolved_as_of = as_of_utc or run["as_of_utc"]
        resolved_state_id = state_id or _make_state_id(resolved_as_of)

        source_matches = _load_matches(conn, source_name=source_name)
        seed_standings = _load_seed_group_standings(conn, source_name=source_name)
        state_matches = [_state_match(row) for row in source_matches]
        group_tables = _build_group_tables(seed_standings, state_matches)
        team_form = _build_team_form(seed_standings, state_matches)

        summary = {
            "total_matches": len(state_matches),
            "completed_matches": sum(1 for row in state_matches if row["is_completed"]),
            "pending_matches": sum(1 for row in state_matches if not row["is_completed"]),
            "group_matches_completed": sum(
                1 for row in state_matches if row["is_completed"] and _is_group_stage(row["stage"])
            ),
            "teams": len(team_form),
            "groups": len({row["group_name"] for row in group_tables if row["group_name"]}),
        }

        _replace_state_tables(
            conn=conn,
            state_id=resolved_state_id,
            source_run_id=resolved_source_run_id,
            as_of_utc=resolved_as_of,
            source_name=source_name,
            state_matches=state_matches,
            group_tables=group_tables,
            team_form=team_form,
            summary=summary,
        )

        output_dir = project_root / "data" / "state" / resolved_state_id
        _write_state_outputs(
            output_dir=output_dir,
            state_id=resolved_state_id,
            source_run_id=resolved_source_run_id,
            as_of_utc=resolved_as_of,
            source_name=source_name,
            state_matches=state_matches,
            group_tables=group_tables,
            team_form=team_form,
            summary=summary,
        )

        return TournamentStateResult(
            state_id=resolved_state_id,
            source_run_id=resolved_source_run_id,
            as_of_utc=resolved_as_of,
            source_name=source_name,
            total_matches=summary["total_matches"],
            completed_matches=summary["completed_matches"],
            pending_matches=summary["pending_matches"],
            group_matches_completed=summary["group_matches_completed"],
            teams=summary["teams"],
            groups=summary["groups"],
            output_dir=output_dir,
        )
    finally:
        store.close()


def _resolve_source_run(conn: sqlite3.Connection, source_run_id: str | None) -> sqlite3.Row:
    if source_run_id:
        row = conn.execute(
            "SELECT * FROM ingestion_runs WHERE run_id = ?",
            (source_run_id,),
        ).fetchone()
    else:
        row = conn.execute("SELECT * FROM v_latest_completed_run").fetchone()
    if row is None:
        raise RuntimeError("No hay una corrida de ingesta completada para construir estado.")
    return row


def _load_matches(conn: sqlite3.Connection, source_name: str) -> list[sqlite3.Row]:
    if source_name != "worldcup26_ir":
        raise ValueError("Por ahora el estado operativo usa source_name='worldcup26_ir'.")
    canonical_count = conn.execute("SELECT COUNT(*) AS n FROM v_canonical_matches").fetchone()["n"]
    if canonical_count:
        rows = conn.execute(
            """
            SELECT
                canonical_match_id,
                primary_source_match_id AS source_match_id,
                match_number,
                stage,
                group_name,
                matchday,
                kickoff_local_raw AS kickoff_local,
                kickoff_utc,
                kickoff_local_iso,
                kickoff_timezone,
                kickoff_guatemala,
                team_a_primary_source_id AS team_a_source_id,
                team_a_canonical_id,
                team_a_name,
                team_a_fifa_code,
                team_b_primary_source_id AS team_b_source_id,
                team_b_canonical_id,
                team_b_name,
                team_b_fifa_code,
                stadium_source_id,
                stadium_name,
                stadium_city,
                stadium_country,
                home_score,
                away_score,
                status,
                source_status AS source_status,
                is_completed AS finished
            FROM v_canonical_matches
            ORDER BY match_number
            """
        ).fetchall()
        if not rows:
            raise RuntimeError("No hay partidos en v_canonical_matches.")
        return rows
    rows = conn.execute(
        """
        SELECT
            NULL AS canonical_match_id,
            source_match_id,
            match_number,
            stage,
            group_name,
            matchday,
            kickoff_local,
            kickoff_utc,
            NULL AS kickoff_local_iso,
            NULL AS kickoff_timezone,
            NULL AS kickoff_guatemala,
            team_a_source_id,
            NULL AS team_a_canonical_id,
            team_a_name,
            team_a_fifa_code,
            team_b_source_id,
            NULL AS team_b_canonical_id,
            team_b_name,
            team_b_fifa_code,
            stadium_source_id,
            stadium_name,
            stadium_city,
            stadium_country,
            home_score,
            away_score,
            status,
            status AS source_status,
            finished
        FROM v_worldcup26_matches
        ORDER BY COALESCE(match_number, CAST(source_match_id AS INTEGER))
        """
    ).fetchall()
    if not rows:
        raise RuntimeError("No hay partidos en v_worldcup26_matches.")
    return rows


def _load_seed_group_standings(conn: sqlite3.Connection, source_name: str) -> list[sqlite3.Row]:
    if source_name != "worldcup26_ir":
        raise ValueError("Por ahora el estado operativo usa source_name='worldcup26_ir'.")
    return conn.execute(
        """
        SELECT *
        FROM v_worldcup26_group_standings
        ORDER BY group_name, CAST(team_source_id AS INTEGER)
        """
    ).fetchall()


def _state_match(row: sqlite3.Row) -> dict[str, Any]:
    home_score = row["home_score"]
    away_score = row["away_score"]
    is_completed = _is_completed(row["status"], row["finished"], home_score, away_score)
    winner_1x2 = None
    goal_difference = None
    team_a_points = None
    team_b_points = None
    if is_completed and home_score is not None and away_score is not None:
        goal_difference = home_score - away_score
        if goal_difference > 0:
            winner_1x2 = "A"
            team_a_points = 3
            team_b_points = 0
        elif goal_difference < 0:
            winner_1x2 = "B"
            team_a_points = 0
            team_b_points = 3
        else:
            winner_1x2 = "D"
            team_a_points = 1
            team_b_points = 1

    return {
        "canonical_match_id": row["canonical_match_id"],
        "source_match_id": row["source_match_id"],
        "match_number": row["match_number"],
        "stage": row["stage"],
        "group_name": row["group_name"],
        "matchday": row["matchday"],
        "kickoff_local": row["kickoff_local"],
        "kickoff_utc": row["kickoff_utc"],
        "kickoff_local_iso": row["kickoff_local_iso"],
        "kickoff_timezone": row["kickoff_timezone"],
        "kickoff_guatemala": row["kickoff_guatemala"],
        "team_a_source_id": row["team_a_source_id"],
        "team_a_canonical_id": row["team_a_canonical_id"],
        "team_a_name": row["team_a_name"],
        "team_a_fifa_code": row["team_a_fifa_code"],
        "team_b_source_id": row["team_b_source_id"],
        "team_b_canonical_id": row["team_b_canonical_id"],
        "team_b_name": row["team_b_name"],
        "team_b_fifa_code": row["team_b_fifa_code"],
        "stadium_source_id": row["stadium_source_id"],
        "stadium_name": row["stadium_name"],
        "stadium_city": row["stadium_city"],
        "stadium_country": row["stadium_country"],
        "home_score": home_score,
        "away_score": away_score,
        "status": "completed" if is_completed else "scheduled",
        "source_status": row["status"],
        "is_completed": 1 if is_completed else 0,
        "winner_1x2": winner_1x2,
        "goal_difference": goal_difference,
        "team_a_points": team_a_points,
        "team_b_points": team_b_points,
    }


def _is_completed(status: Any, finished: Any, home_score: Any, away_score: Any) -> bool:
    if finished == 1:
        return True
    text = str(status or "").strip().lower()
    if text in {"scheduled", "not_started", "not started", "upcoming", "fixture", "pending"}:
        return False
    if text in {"finished", "completed", "played", "ft", "full_time", "final"}:
        return True
    return home_score is not None and away_score is not None


def _build_group_tables(
    seed_standings: list[sqlite3.Row],
    state_matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for seed_order, row in enumerate(seed_standings, start=1):
        group_name = row["group_name"]
        team_source_id = row["team_source_id"]
        if not group_name or not team_source_id:
            continue
        table[(group_name, team_source_id)] = {
            "group_name": group_name,
            "team_source_id": team_source_id,
            "team_name": row["team_name"],
            "fifa_code": row["fifa_code"],
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "rank_sort": None,
            "seed_order": seed_order,
        }

    for match in state_matches:
        if not match["is_completed"] or not _is_group_stage(match["stage"]):
            continue
        group_name = match["group_name"]
        if not group_name:
            continue
        _ensure_group_team(table, group_name, match["team_a_source_id"], match["team_a_name"], match["team_a_fifa_code"])
        _ensure_group_team(table, group_name, match["team_b_source_id"], match["team_b_name"], match["team_b_fifa_code"])
        _apply_group_result(table[(group_name, match["team_a_source_id"])], match["home_score"], match["away_score"])
        _apply_group_result(table[(group_name, match["team_b_source_id"])], match["away_score"], match["home_score"])

    rows = list(table.values())
    rows.sort(
        key=lambda row: (
            row["group_name"],
            -row["points"],
            -row["goal_difference"],
            -row["goals_for"],
            row.get("seed_order") or 999,
            row["team_name"] or "",
        )
    )
    current_group = None
    rank = 0
    for row in rows:
        if row["group_name"] != current_group:
            current_group = row["group_name"]
            rank = 1
        else:
            rank += 1
        row["rank_sort"] = rank
    return rows


def _ensure_group_team(
    table: dict[tuple[str, str], dict[str, Any]],
    group_name: str,
    team_source_id: str,
    team_name: str,
    fifa_code: str,
) -> None:
    if not team_source_id:
        return
    key = (group_name, team_source_id)
    if key in table:
        return
    table[key] = {
        "group_name": group_name,
        "team_source_id": team_source_id,
        "team_name": team_name,
        "fifa_code": fifa_code,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "rank_sort": None,
        "seed_order": 999,
    }


def _apply_group_result(row: dict[str, Any], goals_for: int, goals_against: int) -> None:
    row["played"] += 1
    row["goals_for"] += goals_for
    row["goals_against"] += goals_against
    row["goal_difference"] = row["goals_for"] - row["goals_against"]
    if goals_for > goals_against:
        row["wins"] += 1
        row["points"] += 3
    elif goals_for < goals_against:
        row["losses"] += 1
    else:
        row["draws"] += 1
        row["points"] += 1


def _build_team_form(
    seed_standings: list[sqlite3.Row],
    state_matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    teams: dict[str, dict[str, Any]] = {}
    for row in seed_standings:
        team_source_id = row["team_source_id"]
        if not team_source_id:
            continue
        teams[team_source_id] = _empty_team_form(
            team_source_id=team_source_id,
            team_name=row["team_name"],
            fifa_code=row["fifa_code"],
            group_name=row["group_name"],
        )

    real_team_ids = set(teams)
    for match in state_matches:
        for side in ("a", "b"):
            team_source_id = match[f"team_{side}_source_id"]
            if not team_source_id:
                continue
            if real_team_ids and team_source_id not in real_team_ids:
                continue
            teams.setdefault(
                team_source_id,
                _empty_team_form(
                    team_source_id=team_source_id,
                    team_name=match[f"team_{side}_name"],
                    fifa_code=match[f"team_{side}_fifa_code"],
                    group_name=match["group_name"] if _is_group_stage(match["stage"]) else None,
                ),
            )
        if not match["is_completed"]:
            continue
        if match["team_a_source_id"] not in teams or match["team_b_source_id"] not in teams:
            continue
        _apply_team_form_result(
            teams[match["team_a_source_id"]],
            match["match_number"],
            goals_for=match["home_score"],
            goals_against=match["away_score"],
        )
        _apply_team_form_result(
            teams[match["team_b_source_id"]],
            match["match_number"],
            goals_for=match["away_score"],
            goals_against=match["home_score"],
        )

    rows = []
    for row in teams.values():
        row["goal_difference"] = row["goals_for"] - row["goals_against"]
        row["form_last5"] = "".join(row.pop("_form_results")[-5:])
        rows.append(row)
    rows.sort(key=lambda row: (row["group_name"] or "ZZ", row["team_name"] or ""))
    return rows


def _empty_team_form(team_source_id: str, team_name: str, fifa_code: str, group_name: str | None) -> dict[str, Any]:
    return {
        "team_source_id": team_source_id,
        "team_name": team_name,
        "fifa_code": fifa_code,
        "group_name": group_name,
        "matches_played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "last_match_number": None,
        "form_last5": "",
        "_form_results": [],
    }


def _apply_team_form_result(row: dict[str, Any], match_number: int, goals_for: int, goals_against: int) -> None:
    row["matches_played"] += 1
    row["goals_for"] += goals_for
    row["goals_against"] += goals_against
    row["last_match_number"] = match_number
    if goals_for > goals_against:
        row["wins"] += 1
        row["_form_results"].append("W")
    elif goals_for < goals_against:
        row["losses"] += 1
        row["_form_results"].append("L")
    else:
        row["draws"] += 1
        row["_form_results"].append("D")


def _replace_state_tables(
    conn: sqlite3.Connection,
    state_id: str,
    source_run_id: str,
    as_of_utc: str,
    source_name: str,
    state_matches: list[dict[str, Any]],
    group_tables: list[dict[str, Any]],
    team_form: list[dict[str, Any]],
    summary: dict[str, int],
) -> None:
    created_at = _utc_now()
    with conn:
        conn.execute("DELETE FROM state_matches WHERE state_id = ?", (state_id,))
        conn.execute("DELETE FROM state_group_tables WHERE state_id = ?", (state_id,))
        conn.execute("DELETE FROM state_team_form WHERE state_id = ?", (state_id,))
        conn.execute("DELETE FROM tournament_state_runs WHERE state_id = ?", (state_id,))
        conn.execute(
            """
            INSERT INTO tournament_state_runs (
                state_id, source_run_id, as_of_utc, source_name, created_at_utc,
                total_matches, completed_matches, pending_matches,
                group_matches_completed, teams, groups, status, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state_id,
                source_run_id,
                as_of_utc,
                source_name,
                created_at,
                summary["total_matches"],
                summary["completed_matches"],
                summary["pending_matches"],
                summary["group_matches_completed"],
                summary["teams"],
                summary["groups"],
                "completed",
                "state built from normalized SQLite tables",
            ),
        )
        conn.executemany(
            """
            INSERT INTO state_matches (
                state_id, source_match_id, canonical_match_id, match_number, stage,
                group_name, matchday, kickoff_local, kickoff_utc, kickoff_local_iso,
                kickoff_timezone, kickoff_guatemala, team_a_source_id,
                team_a_canonical_id, team_a_name, team_a_fifa_code,
                team_b_source_id, team_b_canonical_id, team_b_name,
                team_b_fifa_code, stadium_source_id, stadium_name, stadium_city,
                stadium_country, home_score, away_score, status, source_status,
                is_completed, winner_1x2, goal_difference, team_a_points,
                team_b_points
            )
            VALUES (
                :state_id, :source_match_id, :canonical_match_id, :match_number,
                :stage, :group_name, :matchday, :kickoff_local, :kickoff_utc,
                :kickoff_local_iso, :kickoff_timezone, :kickoff_guatemala,
                :team_a_source_id, :team_a_canonical_id, :team_a_name,
                :team_a_fifa_code, :team_b_source_id, :team_b_canonical_id,
                :team_b_name, :team_b_fifa_code, :stadium_source_id,
                :stadium_name, :stadium_city, :stadium_country, :home_score,
                :away_score, :status, :source_status, :is_completed,
                :winner_1x2, :goal_difference, :team_a_points, :team_b_points
            )
            """,
            [dict(row, state_id=state_id) for row in state_matches],
        )
        conn.executemany(
            """
            INSERT INTO state_group_tables (
                state_id, group_name, team_source_id, team_name, fifa_code,
                rank_sort, played, wins, draws, losses, points, goals_for,
                goals_against, goal_difference
            )
            VALUES (
                :state_id, :group_name, :team_source_id, :team_name, :fifa_code,
                :rank_sort, :played, :wins, :draws, :losses, :points, :goals_for,
                :goals_against, :goal_difference
            )
            """,
            [dict(row, state_id=state_id) for row in group_tables],
        )
        conn.executemany(
            """
            INSERT INTO state_team_form (
                state_id, team_source_id, team_name, fifa_code, group_name,
                matches_played, wins, draws, losses, goals_for, goals_against,
                goal_difference, last_match_number, form_last5
            )
            VALUES (
                :state_id, :team_source_id, :team_name, :fifa_code, :group_name,
                :matches_played, :wins, :draws, :losses, :goals_for, :goals_against,
                :goal_difference, :last_match_number, :form_last5
            )
            """,
            [dict(row, state_id=state_id) for row in team_form],
        )


def _write_state_outputs(
    output_dir: Path,
    state_id: str,
    source_run_id: str,
    as_of_utc: str,
    source_name: str,
    state_matches: list[dict[str, Any]],
    group_tables: list[dict[str, Any]],
    team_form: list[dict[str, Any]],
    summary: dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "matches.csv", state_matches)
    _write_csv(output_dir / "group_tables.csv", group_tables)
    _write_csv(output_dir / "team_form.csv", team_form)
    metadata = {
        "state_id": state_id,
        "source_run_id": source_run_id,
        "as_of_utc": as_of_utc,
        "source_name": source_name,
        "created_at_utc": _utc_now(),
        **summary,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _is_group_stage(stage: Any) -> bool:
    return str(stage or "").strip().lower() in {"group", "groups", "group_stage", "group stage"}


def _make_state_id(as_of_utc: str) -> str:
    compact = (
        as_of_utc.replace("-", "")
        .replace(":", "")
        .replace(".", "")
        .replace("+00:00", "Z")
    )
    return f"state_{compact}_{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
