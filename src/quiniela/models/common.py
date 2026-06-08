from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quiniela.storage.sqlite_store import SQLiteStore


HOST_COUNTRIES = {
    "canada": "Canada",
    "mexico": "Mexico",
    "qatar": "Qatar",
    "russia": "Russia",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
}


@dataclass(frozen=True)
class TrainingMatch:
    historical_match_id: str
    match_date: str
    team_a_key: str
    team_b_key: str
    team_a_name: str
    team_b_name: str
    home_score: int
    away_score: int
    neutral: int | None
    importance_weight: float
    recency_weight: float
    tournament: str | None = None
    country: str | None = None
    is_world_cup: int = 0
    is_qualifier: int = 0
    is_friendly: int = 0
    stage: str | None = None


@dataclass(frozen=True)
class PredictionMatch:
    match_id: str
    source_match_id: str
    match_number: int | None
    stage: str | None
    group_name: str | None
    team_a_key: str | None
    team_b_key: str | None
    team_a_name: str | None
    team_b_name: str | None
    kickoff_utc: str | None
    stadium_country: str | None
    status: str | None


@dataclass(frozen=True)
class ModelContext:
    db_path: Path
    as_of_utc: str
    prediction_run_id: str
    tournament_state_id: str
    input_snapshot_id: str
    training_data_version: str
    training_matches: list[TrainingMatch]
    prediction_matches: list[PredictionMatch]


@dataclass(frozen=True)
class ModelPrediction:
    prediction_run_id: str
    as_of_utc: str
    model_id: str
    model_version: str
    match_id: str
    source_match_id: str
    match_number: int | None
    team_a: str | None
    team_b: str | None
    kickoff_utc: str | None
    input_snapshot_id: str
    tournament_state_id: str
    training_data_version: str
    expected_goals_a: float | None
    expected_goals_b: float | None
    p_team_a_win: float | None
    p_draw: float | None
    p_team_b_win: float | None
    score_matrix: dict[str, Any] | None
    top_score: str | None
    top_score_probability: float | None
    selected_score: str | None
    selected_expected_points: float | None
    status: str
    is_evaluation_candidate: bool
    mask_reason: str | None
    warnings: list[str]

    def to_contract_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.prediction_run_id,
            "as_of_utc": self.as_of_utc,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "match_id": self.match_id,
            "source_match_id": self.source_match_id,
            "match_number": self.match_number,
            "team_a": self.team_a,
            "team_b": self.team_b,
            "kickoff_utc": self.kickoff_utc,
            "input_snapshot_id": self.input_snapshot_id,
            "tournament_state_id": self.tournament_state_id,
            "training_data_version": self.training_data_version,
            "expected_goals_a": self.expected_goals_a,
            "expected_goals_b": self.expected_goals_b,
            "p_team_a_win": self.p_team_a_win,
            "p_draw": self.p_draw,
            "p_team_b_win": self.p_team_b_win,
            "score_matrix_json": json.dumps(self.score_matrix, ensure_ascii=False, sort_keys=True)
            if self.score_matrix
            else None,
            "top_score": self.top_score,
            "top_score_probability": self.top_score_probability,
            "selected_score": self.selected_score,
            "selected_expected_points": self.selected_expected_points,
            "status": self.status,
            "is_evaluation_candidate": 1 if self.is_evaluation_candidate else 0,
            "mask_reason": self.mask_reason,
            "warnings": "; ".join(self.warnings),
        }


def load_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"No existe la configuracion: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_context(db_path: Path, prediction_run_id: str, as_of_utc: str | None = None) -> ModelContext:
    store = SQLiteStore(db_path)
    store.initialize()
    conn = store.conn
    try:
        state = _one(conn, "SELECT * FROM v_latest_tournament_state")
        if state is None:
            raise RuntimeError("No hay estado vigente. Ejecuta scripts/build_state.py primero.")
        history = _one(conn, "SELECT * FROM v_latest_history_run")
        if history is None:
            raise RuntimeError("No hay historico vigente. Ejecuta scripts/build_history.py primero.")
        resolved_as_of = as_of_utc or state["as_of_utc"]
        return ModelContext(
            db_path=db_path,
            as_of_utc=resolved_as_of,
            prediction_run_id=prediction_run_id,
            tournament_state_id=state["state_id"],
            input_snapshot_id=state["source_run_id"],
            training_data_version=history["history_run_id"],
            training_matches=_load_training_matches(conn, resolved_as_of),
            prediction_matches=_load_prediction_matches(conn),
        )
    finally:
        store.close()


def _load_training_matches(conn: sqlite3.Connection, as_of_utc: str) -> list[TrainingMatch]:
    cutoff_date = as_of_utc[:10]
    rows = conn.execute(
        """
        SELECT chm.*
        FROM canonical_historical_matches chm
        JOIN v_latest_history_run lhr
          ON lhr.history_run_id = chm.history_run_id
        WHERE chm.match_date < ?
        ORDER BY chm.match_date, chm.historical_match_id
        """,
        (cutoff_date,),
    ).fetchall()
    matches = []
    for row in rows:
        matches.append(
            TrainingMatch(
                historical_match_id=row["historical_match_id"],
                match_date=row["match_date"],
                team_a_key=row["team_a_canonical_id"] or normalize_team_name(row["team_a_name"]),
                team_b_key=row["team_b_canonical_id"] or normalize_team_name(row["team_b_name"]),
                team_a_name=row["team_a_name"],
                team_b_name=row["team_b_name"],
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                neutral=row["neutral"],
                importance_weight=float(row["importance_weight"] or 1.0),
                recency_weight=float(row["recency_weight"] or 1.0),
                tournament=row["tournament"],
                country=row["country"],
                is_world_cup=int(row["is_world_cup"] or 0),
                is_qualifier=int(row["is_qualifier"] or 0),
                is_friendly=int(row["is_friendly"] or 0),
            )
        )
    return matches


def _load_prediction_matches(conn: sqlite3.Connection) -> list[PredictionMatch]:
    rows = conn.execute(
        """
        SELECT *
        FROM v_latest_state_matches
        ORDER BY COALESCE(match_number, CAST(source_match_id AS INTEGER))
        """
    ).fetchall()
    matches = []
    for row in rows:
        match_id = row["canonical_match_id"] or row["source_match_id"]
        matches.append(
            PredictionMatch(
                match_id=match_id,
                source_match_id=str(row["source_match_id"]),
                match_number=row["match_number"],
                stage=row["stage"],
                group_name=row["group_name"],
                team_a_key=row["team_a_canonical_id"] or normalize_team_name(row["team_a_name"]),
                team_b_key=row["team_b_canonical_id"] or normalize_team_name(row["team_b_name"]),
                team_a_name=row["team_a_name"],
                team_b_name=row["team_b_name"],
                kickoff_utc=row["kickoff_utc"],
                stadium_country=row["stadium_country"],
                status=row["status"],
            )
        )
    return matches


def build_score_matrix(lambda_a: float, lambda_b: float, max_goals: int = 8) -> dict[str, Any]:
    probs_a = poisson_probabilities(lambda_a, max_goals)
    probs_b = poisson_probabilities(lambda_b, max_goals)
    raw_scores = {}
    total = 0.0
    for goals_a, prob_a in enumerate(probs_a):
        for goals_b, prob_b in enumerate(probs_b):
            probability = prob_a * prob_b
            raw_scores[f"{goals_a}-{goals_b}"] = probability
            total += probability
    scores = {score: probability / total for score, probability in raw_scores.items()}
    return {"max_goals": max_goals, "scores": scores}


def normalize_score_matrix(score_matrix: dict[str, Any]) -> dict[str, Any]:
    scores = {
        score: max(0.0, float(probability))
        for score, probability in score_matrix.get("scores", {}).items()
    }
    total = sum(scores.values())
    if total <= 0:
        raise RuntimeError("La matriz de marcadores no tiene probabilidad positiva.")
    return {
        "max_goals": int(score_matrix.get("max_goals", 0)),
        "scores": {score: probability / total for score, probability in scores.items()},
    }


def adjust_score_matrix_to_1x2(score_matrix: dict[str, Any], target_probs: dict[str, float]) -> dict[str, Any]:
    normalized = normalize_score_matrix(score_matrix)
    summary = summarize_score_matrix(normalized)
    target = {
        "1": max(0.0, float(target_probs.get("1", 0.0))),
        "X": max(0.0, float(target_probs.get("X", 0.0))),
        "2": max(0.0, float(target_probs.get("2", 0.0))),
    }
    target_total = sum(target.values())
    if target_total <= 0:
        return normalized
    target = {key: value / target_total for key, value in target.items()}
    current = {
        "1": summary["p_team_a_win"],
        "X": summary["p_draw"],
        "2": summary["p_team_b_win"],
    }
    multipliers = {
        key: target[key] / current[key] if current[key] > 0 else 1.0
        for key in ("1", "X", "2")
    }
    adjusted_scores = {}
    for score, probability in normalized["scores"].items():
        goals_a, goals_b = parse_score(score)
        adjusted_scores[score] = probability * multipliers[outcome_1x2(goals_a, goals_b)]
    return normalize_score_matrix({"max_goals": normalized["max_goals"], "scores": adjusted_scores})


def blend_score_matrices(
    first: dict[str, Any],
    second: dict[str, Any],
    first_weight: float,
) -> dict[str, Any]:
    left = normalize_score_matrix(first)
    right = normalize_score_matrix(second)
    weight = clamp(float(first_weight), 0.0, 1.0)
    scores = {}
    for score in set(left["scores"]) | set(right["scores"]):
        scores[score] = left["scores"].get(score, 0.0) * weight + right["scores"].get(score, 0.0) * (1.0 - weight)
    return normalize_score_matrix({"max_goals": max(left["max_goals"], right["max_goals"]), "scores": scores})


def poisson_probabilities(lmbda: float, max_goals: int) -> list[float]:
    lmbda = max(float(lmbda), 0.001)
    probabilities = [math.exp(-lmbda)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * lmbda / goals)
    return probabilities


def summarize_score_matrix(score_matrix: dict[str, Any]) -> dict[str, Any]:
    p_team_a_win = 0.0
    p_draw = 0.0
    p_team_b_win = 0.0
    top_score = None
    top_probability = -1.0
    for score, probability in score_matrix["scores"].items():
        goals_a, goals_b = parse_score(score)
        if goals_a > goals_b:
            p_team_a_win += probability
        elif goals_a < goals_b:
            p_team_b_win += probability
        else:
            p_draw += probability
        if probability > top_probability:
            top_score = score
            top_probability = probability
    return {
        "p_team_a_win": p_team_a_win,
        "p_draw": p_draw,
        "p_team_b_win": p_team_b_win,
        "top_score": top_score,
        "top_score_probability": top_probability,
    }


def expected_goals_from_score_matrix(score_matrix: dict[str, Any]) -> tuple[float, float]:
    matrix = normalize_score_matrix(score_matrix)
    goals_a = 0.0
    goals_b = 0.0
    for score, probability in matrix["scores"].items():
        score_goals_a, score_goals_b = parse_score(score)
        goals_a += score_goals_a * float(probability)
        goals_b += score_goals_b * float(probability)
    return goals_a, goals_b


def parse_score(score: str) -> tuple[int, int]:
    left, right = score.split("-", 1)
    return int(left), int(right)


def outcome_1x2(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "1"
    if goals_a < goals_b:
        return "2"
    return "X"


def mask_reason_for_match(match: PredictionMatch) -> str | None:
    if str(match.status or "").lower() == "completed":
        return "completed_match"
    if not match.team_a_name or not match.team_b_name:
        return "missing_team_assignment"
    text = f"{match.team_a_name} {match.team_b_name}".lower()
    if any(token in text for token in ("winner", "runner-up", "runner up", "loser", "third place")):
        return "unassigned_knockout_placeholder"
    return None


def is_predictable_match(match: PredictionMatch) -> bool:
    return mask_reason_for_match(match) is None


def masked_prediction(
    context: ModelContext,
    model_id: str,
    model_version: str,
    match: PredictionMatch,
    mask_reason: str,
) -> ModelPrediction:
    return ModelPrediction(
        prediction_run_id=context.prediction_run_id,
        as_of_utc=context.as_of_utc,
        model_id=model_id,
        model_version=model_version,
        match_id=match.match_id,
        source_match_id=match.source_match_id,
        match_number=match.match_number,
        team_a=match.team_a_name,
        team_b=match.team_b_name,
        kickoff_utc=match.kickoff_utc,
        input_snapshot_id=context.input_snapshot_id,
        tournament_state_id=context.tournament_state_id,
        training_data_version=context.training_data_version,
        expected_goals_a=None,
        expected_goals_b=None,
        p_team_a_win=None,
        p_draw=None,
        p_team_b_win=None,
        score_matrix=None,
        top_score=None,
        top_score_probability=None,
        selected_score=None,
        selected_expected_points=None,
        status="masked",
        is_evaluation_candidate=False,
        mask_reason=mask_reason,
        warnings=[f"ignorado para evaluacion: {mask_reason}"],
    )


def failed_prediction(
    context: ModelContext,
    model_id: str,
    model_version: str,
    match: PredictionMatch,
    warning: str,
) -> ModelPrediction:
    return ModelPrediction(
        prediction_run_id=context.prediction_run_id,
        as_of_utc=context.as_of_utc,
        model_id=model_id,
        model_version=model_version,
        match_id=match.match_id,
        source_match_id=match.source_match_id,
        match_number=match.match_number,
        team_a=match.team_a_name,
        team_b=match.team_b_name,
        kickoff_utc=match.kickoff_utc,
        input_snapshot_id=context.input_snapshot_id,
        tournament_state_id=context.tournament_state_id,
        training_data_version=context.training_data_version,
        expected_goals_a=None,
        expected_goals_b=None,
        p_team_a_win=None,
        p_draw=None,
        p_team_b_win=None,
        score_matrix=None,
        top_score=None,
        top_score_probability=None,
        selected_score=None,
        selected_expected_points=None,
        status="failed",
        is_evaluation_candidate=False,
        mask_reason=None,
        warnings=[warning],
    )


def successful_prediction(
    context: ModelContext,
    model_id: str,
    model_version: str,
    match: PredictionMatch,
    lambda_a: float,
    lambda_b: float,
    max_goals: int,
    selected_score: str | None,
    selected_expected_points: float | None,
    warnings: list[str] | None = None,
) -> ModelPrediction:
    score_matrix = build_score_matrix(lambda_a, lambda_b, max_goals)
    summary = summarize_score_matrix(score_matrix)
    return ModelPrediction(
        prediction_run_id=context.prediction_run_id,
        as_of_utc=context.as_of_utc,
        model_id=model_id,
        model_version=model_version,
        match_id=match.match_id,
        source_match_id=match.source_match_id,
        match_number=match.match_number,
        team_a=match.team_a_name,
        team_b=match.team_b_name,
        kickoff_utc=match.kickoff_utc,
        input_snapshot_id=context.input_snapshot_id,
        tournament_state_id=context.tournament_state_id,
        training_data_version=context.training_data_version,
        expected_goals_a=round(lambda_a, 6),
        expected_goals_b=round(lambda_b, 6),
        p_team_a_win=round(summary["p_team_a_win"], 10),
        p_draw=round(summary["p_draw"], 10),
        p_team_b_win=round(summary["p_team_b_win"], 10),
        score_matrix=score_matrix,
        top_score=summary["top_score"],
        top_score_probability=round(summary["top_score_probability"], 10),
        selected_score=selected_score,
        selected_expected_points=selected_expected_points,
        status="ok",
        is_evaluation_candidate=True,
        mask_reason=None,
        warnings=warnings or [],
    )


def successful_prediction_from_matrix(
    context: ModelContext,
    model_id: str,
    model_version: str,
    match: PredictionMatch,
    lambda_a: float,
    lambda_b: float,
    score_matrix: dict[str, Any],
    selected_score: str | None,
    selected_expected_points: float | None,
    warnings: list[str] | None = None,
) -> ModelPrediction:
    normalized_matrix = normalize_score_matrix(score_matrix)
    summary = summarize_score_matrix(normalized_matrix)
    matrix_goals_a, matrix_goals_b = expected_goals_from_score_matrix(normalized_matrix)
    expected_goals_a, normalized_a = _safe_expected_goal(lambda_a, matrix_goals_a)
    expected_goals_b, normalized_b = _safe_expected_goal(lambda_b, matrix_goals_b)
    final_warnings = list(warnings or [])
    if normalized_a or normalized_b:
        final_warnings.append("expected_goals_normalized_from_score_matrix")
    return ModelPrediction(
        prediction_run_id=context.prediction_run_id,
        as_of_utc=context.as_of_utc,
        model_id=model_id,
        model_version=model_version,
        match_id=match.match_id,
        source_match_id=match.source_match_id,
        match_number=match.match_number,
        team_a=match.team_a_name,
        team_b=match.team_b_name,
        kickoff_utc=match.kickoff_utc,
        input_snapshot_id=context.input_snapshot_id,
        tournament_state_id=context.tournament_state_id,
        training_data_version=context.training_data_version,
        expected_goals_a=round(expected_goals_a, 6),
        expected_goals_b=round(expected_goals_b, 6),
        p_team_a_win=round(summary["p_team_a_win"], 10),
        p_draw=round(summary["p_draw"], 10),
        p_team_b_win=round(summary["p_team_b_win"], 10),
        score_matrix=normalized_matrix,
        top_score=summary["top_score"],
        top_score_probability=round(summary["top_score_probability"], 10),
        selected_score=selected_score,
        selected_expected_points=selected_expected_points,
        status="ok",
        is_evaluation_candidate=True,
        mask_reason=None,
        warnings=final_warnings,
    )


def _safe_expected_goal(raw_value: float, fallback_value: float, limit: float = 6.0) -> tuple[float, bool]:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return float(fallback_value), True
    if not math.isfinite(value) or value < 0.0 or value > limit:
        return float(fallback_value), True
    return value, False


def write_prediction_artifacts(
    output_dir: Path,
    model_id: str,
    model_version: str,
    context: ModelContext,
    predictions: list[ModelPrediction],
    notes: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{model_id}.json"
    csv_path = output_dir / f"{model_id}.csv"
    payload = {
        "metadata": {
            "prediction_run_id": context.prediction_run_id,
            "as_of_utc": context.as_of_utc,
            "model_id": model_id,
            "model_version": model_version,
            "training_data_version": context.training_data_version,
            "input_snapshot_id": context.input_snapshot_id,
            "tournament_state_id": context.tournament_state_id,
            "created_at_utc": utc_now(),
            "notes": notes,
        },
        "predictions": [prediction.to_contract_dict() for prediction in predictions],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    fieldnames = [
        "run_id",
        "as_of_utc",
        "model_id",
        "model_version",
        "match_id",
        "source_match_id",
        "match_number",
        "team_a",
        "team_b",
        "kickoff_utc",
        "expected_goals_a",
        "expected_goals_b",
        "p_team_a_win",
        "p_draw",
        "p_team_b_win",
        "top_score",
        "top_score_probability",
        "selected_score",
        "selected_expected_points",
        "status",
        "warnings",
        "is_evaluation_candidate",
        "mask_reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            row = prediction.to_contract_dict()
            writer.writerow({field: row.get(field) for field in fieldnames})
    return json_path, csv_path


def store_predictions_in_sqlite(
    db_path: Path,
    model_id: str,
    model_version: str,
    context: ModelContext,
    predictions: list[ModelPrediction],
    json_path: Path,
    csv_path: Path,
    notes: str,
) -> None:
    store = SQLiteStore(db_path)
    store.initialize()
    conn = store.conn
    try:
        successful = sum(1 for prediction in predictions if prediction.status == "ok")
        masked = sum(1 for prediction in predictions if prediction.status == "masked")
        failed = len(predictions) - successful - masked
        created_at = utc_now()
        rows = []
        for prediction in predictions:
            row = prediction.to_contract_dict()
            rows.append(
                {
                    "prediction_run_id": context.prediction_run_id,
                    "model_id": model_id,
                    "model_version": model_version,
                    "match_id": row["match_id"],
                    "source_match_id": row["source_match_id"],
                    "match_number": row["match_number"],
                    "team_a_name": row["team_a"],
                    "team_b_name": row["team_b"],
                    "kickoff_utc": row["kickoff_utc"],
                    "tournament_state_id": context.tournament_state_id,
                    "expected_goals_a": row["expected_goals_a"],
                    "expected_goals_b": row["expected_goals_b"],
                    "p_team_a_win": row["p_team_a_win"],
                    "p_draw": row["p_draw"],
                    "p_team_b_win": row["p_team_b_win"],
                    "score_matrix_json": row["score_matrix_json"],
                    "top_score": row["top_score"],
                    "top_score_probability": row["top_score_probability"],
                    "selected_score": row["selected_score"],
                    "selected_expected_points": row["selected_expected_points"],
                    "status": row["status"],
                    "is_evaluation_candidate": row["is_evaluation_candidate"],
                    "mask_reason": row["mask_reason"],
                    "warnings": row["warnings"],
                    "created_at_utc": created_at,
                }
            )
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_prediction_runs (
                    prediction_run_id, model_id, model_version, as_of_utc,
                    created_at_utc, training_data_version, input_snapshot_id,
                    tournament_state_id, predictions, successful_predictions,
                    failed_predictions, masked_predictions, output_json_path,
                    output_csv_path, status, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    context.prediction_run_id,
                    model_id,
                    model_version,
                    context.as_of_utc,
                    created_at,
                    context.training_data_version,
                    context.input_snapshot_id,
                    context.tournament_state_id,
                    len(predictions),
                    successful,
                    failed,
                    masked,
                    str(json_path),
                    str(csv_path),
                    "completed",
                    notes,
                ),
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO model_predictions (
                    prediction_run_id, model_id, model_version, match_id,
                    source_match_id, match_number, team_a_name, team_b_name,
                    kickoff_utc, tournament_state_id, expected_goals_a,
                    expected_goals_b, p_team_a_win, p_draw, p_team_b_win,
                    score_matrix_json, top_score, top_score_probability,
                    selected_score, selected_expected_points, status,
                    is_evaluation_candidate, mask_reason, warnings, created_at_utc
                )
                VALUES (
                    :prediction_run_id, :model_id, :model_version, :match_id,
                    :source_match_id, :match_number, :team_a_name, :team_b_name,
                    :kickoff_utc, :tournament_state_id, :expected_goals_a,
                    :expected_goals_b, :p_team_a_win, :p_draw, :p_team_b_win,
                    :score_matrix_json, :top_score, :top_score_probability,
                    :selected_score, :selected_expected_points, :status,
                    :is_evaluation_candidate, :mask_reason, :warnings,
                    :created_at_utc
                )
                """,
                rows,
            )
    finally:
        store.close()


def normalize_team_name(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    ascii_text = ascii_text.lower().replace("&", " and ")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()
    ascii_text = re.sub(r"\s+", " ", ascii_text)
    aliases = {
        "usa": "united states",
        "us": "united states",
        "united states of america": "united states",
        "cote d ivoire": "ivory coast",
        "cote divoire": "ivory coast",
        "korea republic": "south korea",
        "republic of korea": "south korea",
    }
    return aliases.get(ascii_text, ascii_text)


def host_bonus_for(team_name: str | None, stadium_country: str | None, home_advantage: float) -> float:
    team = normalize_team_name(team_name)
    country = normalize_team_name(stadium_country)
    canonical_country = HOST_COUNTRIES.get(team)
    if canonical_country and normalize_team_name(canonical_country) == country:
        return home_advantage
    return 0.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _one(conn: sqlite3.Connection, query: str) -> sqlite3.Row | None:
    return conn.execute(query).fetchone()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
