from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from io import StringIO
from typing import Any


@dataclass
class NormalizedBatch:
    teams: list[dict[str, Any]] = field(default_factory=list)
    stadiums: list[dict[str, Any]] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    group_standings: list[dict[str, Any]] = field(default_factory=list)


def normalize_payload(normalizer: str, body: bytes, source_name: str) -> NormalizedBatch:
    if normalizer.endswith("_csv"):
        text = body.decode("utf-8-sig")
        payload = list(csv.DictReader(StringIO(text)))
    else:
        payload = _decode_json(body)

    if normalizer == "worldcup26_games":
        return _worldcup26_games(payload, source_name)
    if normalizer == "worldcup26_teams":
        return _worldcup26_teams(payload, source_name)
    if normalizer == "worldcup26_groups":
        return _worldcup26_groups(payload, source_name)
    if normalizer == "worldcup26_stadiums":
        return _worldcup26_stadiums(payload, source_name)
    if normalizer == "openfootball_worldcup_2026":
        return _openfootball_worldcup(payload, source_name)
    if normalizer == "worldcup26_games_csv":
        return _worldcup26_games(payload, source_name)
    if normalizer == "worldcup26_teams_csv":
        return _worldcup26_teams(payload, source_name)
    if normalizer == "worldcup26_groups_csv":
        return _worldcup26_groups(payload, source_name)
    if normalizer == "worldcup26_stadiums_csv":
        return _worldcup26_stadiums(payload, source_name)

    return NormalizedBatch()


def _decode_json(body: bytes) -> Any:
    text = body.decode("utf-8-sig").strip()
    if not text:
        return None
    return json.loads(text)


def _as_list(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "matches", "games", "teams", "groups", "stadiums"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return []


def _first(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _boolish(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "finished", "ft"}:
        return True
    if text in {"false", "0", "no", "n", "scheduled"}:
        return False
    return None


def _payload_json(row: Any) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _worldcup26_teams(payload: Any, source_name: str) -> NormalizedBatch:
    batch = NormalizedBatch()
    for row in _as_list(payload):
        if not isinstance(row, dict):
            continue
        batch.teams.append(
            {
                "source_name": source_name,
                "source_team_id": str(_first(row, "id", "_id", "team_id", default="")),
                "fifa_code": _first(row, "fifa_code", "code", "iso_code"),
                "name": _first(row, "name_en", "name", "team", "team_name"),
                "group_name": _first(row, "groups", "group", "group_name"),
                "payload_json": _payload_json(row),
            }
        )
    return batch


def _worldcup26_stadiums(payload: Any, source_name: str) -> NormalizedBatch:
    batch = NormalizedBatch()
    for row in _as_list(payload):
        if not isinstance(row, dict):
            continue
        batch.stadiums.append(
            {
                "source_name": source_name,
                "source_stadium_id": str(_first(row, "id", "_id", "stadium_id", default="")),
                "name": _first(row, "name_en", "name", "stadium", "fifa_name"),
                "fifa_name": _first(row, "fifa_name"),
                "city": _first(row, "city_en", "city"),
                "country": _first(row, "country_en", "country"),
                "capacity": _to_int(_first(row, "capacity")),
                "payload_json": _payload_json(row),
            }
        )
    return batch


def _worldcup26_games(payload: Any, source_name: str) -> NormalizedBatch:
    batch = NormalizedBatch()
    for row in _as_list(payload):
        if not isinstance(row, dict):
            continue
        finished = _boolish(_first(row, "finished", "is_finished", "status"))
        status = "finished" if finished else "scheduled"
        batch.matches.append(
            {
                "source_name": source_name,
                "source_match_id": str(_first(row, "id", "_id", "match_id", default="")),
                "match_number": _to_int(_first(row, "id", "match_number", "number")),
                "stage": _first(row, "type", "stage", "round"),
                "group_name": _first(row, "group", "group_name"),
                "matchday": _first(row, "matchday", "round"),
                "kickoff_local": _first(row, "local_date", "date", "kickoff_local"),
                "kickoff_utc": _first(row, "kickoff_utc"),
                "team_a_source_id": str(_first(row, "home_team_id", "team1_id", "home_id", default="")),
                "team_b_source_id": str(_first(row, "away_team_id", "team2_id", "away_id", default="")),
                "team_a_name": _first(row, "home_team", "team1", "home_name"),
                "team_b_name": _first(row, "away_team", "team2", "away_name"),
                "home_team_label": _first(row, "home_team_label"),
                "away_team_label": _first(row, "away_team_label"),
                "stadium_source_id": str(_first(row, "stadium_id", default="")),
                "home_score": _to_int(_first(row, "home_score", "score1", "home_goals")),
                "away_score": _to_int(_first(row, "away_score", "score2", "away_goals")),
                "status": status,
                "finished": 1 if finished else 0,
                "payload_json": _payload_json(row),
            }
        )
    return batch


def _worldcup26_groups(payload: Any, source_name: str) -> NormalizedBatch:
    batch = NormalizedBatch()
    for group_row in _as_list(payload):
        if not isinstance(group_row, dict):
            continue
        group_name = _first(group_row, "group", "name", "group_name")
        teams = group_row.get("teams")
        if isinstance(teams, list):
            for team_row in teams:
                if isinstance(team_row, dict):
                    batch.group_standings.append(_standing_row(source_name, group_name, team_row))
        else:
            batch.group_standings.append(_standing_row(source_name, group_name, group_row))
    return batch


def _standing_row(source_name: str, group_name: Any, row: dict[str, Any]) -> dict[str, Any]:
    wins = _to_int(_first(row, "wins", "w", "win", default=0)) or 0
    draws = _to_int(_first(row, "draws", "d", "draw", default=0)) or 0
    losses = _to_int(_first(row, "losses", "l", "loss", default=0)) or 0
    gf = _to_int(_first(row, "gf", "goals_for", default=0)) or 0
    ga = _to_int(_first(row, "ga", "goals_against", default=0)) or 0
    return {
        "source_name": source_name,
        "group_name": group_name,
        "team_source_id": str(_first(row, "team_id", "id", default="")),
        "team_name": _first(row, "team", "team_name", "name", "name_en"),
        "played": _to_int(_first(row, "played", "p", "mp", default=wins + draws + losses)),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": _to_int(_first(row, "pts", "points", default=wins * 3 + draws)),
        "goals_for": gf,
        "goals_against": ga,
        "goal_difference": _to_int(_first(row, "gd", "goal_difference", default=gf - ga)),
        "payload_json": _payload_json(row),
    }


def _openfootball_worldcup(payload: Any, source_name: str) -> NormalizedBatch:
    batch = NormalizedBatch()
    if not isinstance(payload, dict):
        return batch
    for index, row in enumerate(payload.get("matches", []), start=1):
        if not isinstance(row, dict):
            continue
        score = row.get("score") if isinstance(row.get("score"), dict) else {}
        full_time = score.get("ft") if isinstance(score, dict) else None
        home_score = away_score = None
        if isinstance(full_time, list) and len(full_time) >= 2:
            home_score = _to_int(full_time[0])
            away_score = _to_int(full_time[1])
        team_a = _first(row, "team1")
        team_b = _first(row, "team2")
        batch.matches.append(
            {
                "source_name": source_name,
                "source_match_id": str(index),
                "match_number": index,
                "stage": _first(row, "round"),
                "group_name": _first(row, "group"),
                "matchday": _first(row, "round"),
                "kickoff_local": " ".join(str(part) for part in (_first(row, "date", default=""), _first(row, "time", default="")) if part),
                "kickoff_utc": None,
                "team_a_source_id": team_a,
                "team_b_source_id": team_b,
                "team_a_name": team_a,
                "team_b_name": team_b,
                "home_team_label": None,
                "away_team_label": None,
                "stadium_source_id": _first(row, "ground"),
                "home_score": home_score,
                "away_score": away_score,
                "status": "finished" if full_time else "scheduled",
                "finished": 1 if full_time else 0,
                "payload_json": _payload_json(row),
            }
        )
    return batch

