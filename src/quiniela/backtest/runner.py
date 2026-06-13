from __future__ import annotations

import csv
import json
import math
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from quiniela.ensemble import build_weighted_ensemble_predictions
from quiniela.models import (
    run_attack_defense_poisson,
    run_baseline_poisson,
    run_bayesian_monte_carlo_scoreline,
    run_bradley_terry_davidson,
    run_draw_specialist,
    run_elo_dixon_coles,
    run_elo_poisson,
    run_opta_power_poisson,
    run_similar_match_knn_scoreline,
)
from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    PredictionMatch,
    TrainingMatch,
    load_json_config,
    normalize_team_name,
    outcome_1x2,
    parse_score,
)
from quiniela.models.neural_hybrid_v2 import run_neural_hybrid_v2
from quiniela.models.neural_scoreline_mlp import run_neural_scoreline_mlp
from quiniela.scoring.quiniela import resolve_scoring_profile
from quiniela.storage.sqlite_store import SQLiteStore


MODEL_RUNNERS: dict[str, Callable[[ModelContext, dict[str, Any], dict[str, Any]], list[ModelPrediction]]] = {
    "attack_defense_poisson": run_attack_defense_poisson,
    "baseline_poisson": run_baseline_poisson,
    "bayesian_monte_carlo_scoreline": run_bayesian_monte_carlo_scoreline,
    "bradley_terry_davidson": run_bradley_terry_davidson,
    "draw_specialist": run_draw_specialist,
    "elo_dixon_coles": run_elo_dixon_coles,
    "elo_poisson": run_elo_poisson,
    "neural_hybrid_v2": run_neural_hybrid_v2,
    "neural_scoreline_mlp": run_neural_scoreline_mlp,
    "opta_power_poisson": run_opta_power_poisson,
    "similar_match_knn_scoreline": run_similar_match_knn_scoreline,
}

ENSEMBLE_MODEL_IDS = {
    "weighted_ensemble",
    "weighted_points_ensemble",
    "weighted_1x2_ensemble",
    "weighted_exact_ensemble",
    "calibrated_scoreline_ensemble",
}

EPSILON = 1e-12


@dataclass(frozen=True)
class BacktestResult:
    backtest_run_id: str
    years: list[int]
    models: list[str]
    matches: int
    predictions: int
    output_json_path: Path
    output_csv_path: Path


@dataclass(frozen=True)
class BacktestMatch:
    match_id: str
    year: int
    match_number: int
    match_date: str
    stage: str
    team_a_key: str
    team_b_key: str
    team_a_name: str
    team_b_name: str
    home_score: int
    away_score: int
    country: str | None
    neutral: int | None
    tournament: str | None


def run_backtest(
    db_path: Path,
    project_root: Path,
    backtest_config_path: Path | None = None,
    models_config_path: Path | None = None,
    scoring_config_path: Path | None = None,
    output_root: Path | None = None,
    scoring_profile: str | None = None,
) -> BacktestResult:
    backtest_config = load_json_config(backtest_config_path or project_root / "configs" / "backtest.yaml")
    models_config = load_json_config(models_config_path or project_root / "configs" / "models.yaml")
    scoring_config_raw = load_json_config(scoring_config_path or project_root / "configs" / "scoring.yaml")
    scoring_config = resolve_scoring_profile(scoring_config_raw, scoring_profile)
    years = [int(year) for year in backtest_config.get("world_cup_years", [2018, 2022])]
    model_configs = _select_model_configs(models_config, backtest_config)
    base_model_configs = [config for config in model_configs if not _is_ensemble_model(config)]
    ensemble_model_configs = [config for config in model_configs if _is_ensemble_model(config)]
    as_of_utc = _utc_now()
    prefix = str(backtest_config.get("backtest_id_prefix", "backtest"))
    backtest_run_id = f"backtest_{prefix}_{_compact_timestamp(as_of_utc)}_{uuid.uuid4().hex[:8]}"
    output_dir = (output_root or project_root / "data" / "backtests") / backtest_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(db_path)
    store.initialize()
    conn = store.conn
    try:
        _ensure_backtest_schema(conn)
        history_run_id = _latest_history_run_id(conn)
        matches = _load_world_cup_matches(conn, history_run_id, years)
        if not matches:
            raise RuntimeError("No hay partidos historicos de Mundial para los años configurados.")
        prediction_rows: list[dict[str, Any]] = []
        match_rows = [_match_row(backtest_run_id, match) for match in matches]
        predictions_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for match_date, day_matches in _group_matches_by_date(matches).items():
            training_matches = _load_training_matches(conn, history_run_id, match_date)
            prediction_matches = [_prediction_match(match) for match in day_matches]
            context = ModelContext(
                db_path=db_path,
                as_of_utc=f"{match_date}T00:00:00Z",
                prediction_run_id=backtest_run_id,
                tournament_state_id=f"historical_wc_{match_date}",
                input_snapshot_id=history_run_id,
                training_data_version=history_run_id,
                training_matches=training_matches,
                prediction_matches=prediction_matches,
            )
            actual_by_match_id = {match.match_id: match for match in day_matches}
            day_predictions_by_model: dict[str, list[ModelPrediction]] = {}
            for model_config in base_model_configs:
                model_id = str(model_config["model_id"])
                runner = MODEL_RUNNERS.get(model_id)
                if runner is None:
                    continue
                predictions = runner(context, model_config, scoring_config)
                day_predictions_by_model[model_id] = predictions
                for prediction in predictions:
                    actual = actual_by_match_id[prediction.match_id]
                    row = _prediction_row(
                        backtest_run_id=backtest_run_id,
                        prediction=prediction,
                        actual=actual,
                        scoring_config=scoring_config,
                        is_reference_model=_is_reference_model(backtest_config, model_id),
                    )
                    prediction_rows.append(row)
                    predictions_by_model[model_id].append(row)
            for model_config in ensemble_model_configs:
                model_id = str(model_config["model_id"])
                ensemble_config = dict(model_config)
                weight_source = str(ensemble_config.get("weight_source", "latest_backtest"))
                if not ensemble_config.get("allow_backtest_weight_source") and weight_source != "optimized_backtest":
                    ensemble_config["weight_source"] = "fallback"
                predictions = build_weighted_ensemble_predictions(
                    context=context,
                    predictions_by_model=day_predictions_by_model,
                    model_config=ensemble_config,
                    scoring_config=scoring_config,
                )
                day_predictions_by_model[model_id] = predictions
                for prediction in predictions:
                    actual = actual_by_match_id[prediction.match_id]
                    row = _prediction_row(
                        backtest_run_id=backtest_run_id,
                        prediction=prediction,
                        actual=actual,
                        scoring_config=scoring_config,
                        is_reference_model=_is_reference_model(backtest_config, model_id),
                    )
                    prediction_rows.append(row)
                    predictions_by_model[model_id].append(row)

        metric_rows = _metric_rows(backtest_run_id, prediction_rows, scoring_config)
        json_path, csv_path = _write_artifacts(
            output_dir=output_dir,
            backtest_run_id=backtest_run_id,
            as_of_utc=as_of_utc,
            years=years,
            model_configs=model_configs,
            matches=match_rows,
            predictions=prediction_rows,
            metrics=metric_rows,
            notes=str(backtest_config.get("notes", "")),
        )
        _store_backtest(
            conn=conn,
            backtest_run_id=backtest_run_id,
            as_of_utc=as_of_utc,
            history_run_id=history_run_id,
            years=years,
            model_configs=model_configs,
            backtest_config=backtest_config,
            match_rows=match_rows,
            prediction_rows=prediction_rows,
            metric_rows=metric_rows,
            json_path=json_path,
            csv_path=csv_path,
        )
        return BacktestResult(
            backtest_run_id=backtest_run_id,
            years=years,
            models=[str(config["model_id"]) for config in model_configs],
            matches=len(matches),
            predictions=len(prediction_rows),
            output_json_path=json_path,
            output_csv_path=csv_path,
        )
    finally:
        store.close()


def _select_model_configs(models_config: dict[str, Any], backtest_config: dict[str, Any]) -> list[dict[str, Any]]:
    include = set(backtest_config.get("include_models") or [])
    exclude = set(backtest_config.get("exclude_models") or [])
    models = []
    for model_config in models_config.get("models", []):
        model_id = str(model_config.get("model_id"))
        if include and model_id not in include:
            continue
        if not include and backtest_config.get("use_active_models", True) and not model_config.get("active"):
            continue
        if model_id in exclude:
            continue
        if model_id in MODEL_RUNNERS or model_id in ENSEMBLE_MODEL_IDS or model_config.get("ensemble"):
            models.append(dict(model_config))
    if not models:
        raise RuntimeError("No hay modelos implementados seleccionados para backtest.")
    return models


def _is_ensemble_model(model_config: dict[str, Any]) -> bool:
    return bool(model_config.get("ensemble")) or str(model_config.get("model_id")) in ENSEMBLE_MODEL_IDS


def _load_world_cup_matches(conn: sqlite3.Connection, history_run_id: str, years: list[int]) -> list[BacktestMatch]:
    start = f"{min(years)}-01-01"
    end = f"{max(years) + 1}-01-01"
    rows = conn.execute(
        """
        SELECT *
        FROM canonical_historical_matches
        WHERE history_run_id = ?
          AND is_world_cup = 1
          AND match_date >= ?
          AND match_date < ?
        ORDER BY match_date, historical_match_id
        """,
        (history_run_id, start, end),
    ).fetchall()
    selected = [row for row in rows if int(str(row["match_date"])[:4]) in set(years)]
    counters: dict[int, int] = defaultdict(int)
    matches = []
    for row in selected:
        year = int(str(row["match_date"])[:4])
        counters[year] += 1
        match_number = counters[year]
        matches.append(
            BacktestMatch(
                match_id=f"wc{year}_{match_number:02d}",
                year=year,
                match_number=match_number,
                match_date=row["match_date"],
                stage=_infer_world_cup_stage(match_number),
                team_a_key=row["team_a_canonical_id"] or normalize_team_name(row["team_a_name"]),
                team_b_key=row["team_b_canonical_id"] or normalize_team_name(row["team_b_name"]),
                team_a_name=row["team_a_name"],
                team_b_name=row["team_b_name"],
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                country=row["country"],
                neutral=row["neutral"],
                tournament=row["tournament"],
            )
        )
    return matches


def _load_training_matches(conn: sqlite3.Connection, history_run_id: str, cutoff_date: str) -> list[TrainingMatch]:
    rows = conn.execute(
        """
        SELECT *
        FROM canonical_historical_matches
        WHERE history_run_id = ?
          AND match_date < ?
        ORDER BY match_date, historical_match_id
        """,
        (history_run_id, cutoff_date),
    ).fetchall()
    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    matches = []
    for row in rows:
        match_date = datetime.strptime(row["match_date"], "%Y-%m-%d").date()
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
                recency_weight=_recency_weight(match_date, cutoff),
                tournament=row["tournament"],
                country=row["country"],
                is_world_cup=int(row["is_world_cup"] or 0),
                is_qualifier=int(row["is_qualifier"] or 0),
                is_friendly=int(row["is_friendly"] or 0),
                stage=None,
            )
        )
    return matches


def _prediction_match(match: BacktestMatch) -> PredictionMatch:
    return PredictionMatch(
        match_id=match.match_id,
        source_match_id=match.match_id,
        match_number=match.match_number,
        stage=match.stage,
        group_name=None,
        team_a_key=match.team_a_key,
        team_b_key=match.team_b_key,
        team_a_name=match.team_a_name,
        team_b_name=match.team_b_name,
        kickoff_utc=f"{match.match_date}T00:00:00Z",
        stadium_country=match.country,
        status="historical_backtest",
    )


def _prediction_row(
    backtest_run_id: str,
    prediction: ModelPrediction,
    actual: BacktestMatch,
    scoring_config: dict[str, Any],
    is_reference_model: bool = False,
) -> dict[str, Any]:
    actual_score = f"{actual.home_score}-{actual.away_score}"
    selected_score = prediction.selected_score or prediction.top_score
    if not selected_score:
        points = 0.0
        exact_hit = margin_or_draw_hit = winner_hit = 0
        selected_outcome = None
    else:
        points, exact_hit, margin_or_draw_hit, winner_hit = _score_pick(selected_score, actual_score, scoring_config)
        selected_a, selected_b = parse_score(selected_score)
        selected_outcome = outcome_1x2(selected_a, selected_b)
    if not prediction.top_score:
        top_points = 0.0
        top_exact_hit = top_margin_or_draw_hit = top_winner_hit = 0
        top_outcome = None
    else:
        top_points, top_exact_hit, top_margin_or_draw_hit, top_winner_hit = _score_pick(
            prediction.top_score,
            actual_score,
            scoring_config,
        )
        top_a, top_b = parse_score(prediction.top_score)
        top_outcome = outcome_1x2(top_a, top_b)
    actual_outcome = outcome_1x2(actual.home_score, actual.away_score)
    brier = _brier_1x2(prediction, actual_outcome)
    log_loss = -math.log(max(EPSILON, _prob_actual_outcome(prediction, actual_outcome)))
    scoreline_probability = _scoreline_probability(prediction, actual_score)
    scoreline_log_loss = -math.log(max(EPSILON, scoreline_probability))
    warnings = "; ".join(prediction.warnings)
    if is_reference_model:
        warnings = "; ".join(
            item
            for item in (
                warnings,
                "referencia visual: artefacto final entrenado con informacion posterior al mundial evaluado",
            )
            if item
        )
    return {
        "backtest_run_id": backtest_run_id,
        "model_id": prediction.model_id,
        "model_version": prediction.model_version,
        "match_id": actual.match_id,
        "year": actual.year,
        "match_number": actual.match_number,
        "match_date": actual.match_date,
        "stage": actual.stage,
        "team_a_name": actual.team_a_name,
        "team_b_name": actual.team_b_name,
        "actual_score": actual_score,
        "actual_outcome": actual_outcome,
        "expected_goals_a": prediction.expected_goals_a,
        "expected_goals_b": prediction.expected_goals_b,
        "p_team_a_win": prediction.p_team_a_win,
        "p_draw": prediction.p_draw,
        "p_team_b_win": prediction.p_team_b_win,
        "score_matrix_json": json.dumps(prediction.score_matrix, ensure_ascii=False, sort_keys=True)
        if prediction.score_matrix
        else None,
        "top_score": prediction.top_score,
        "top_score_probability": prediction.top_score_probability,
        "top_outcome": top_outcome,
        "top_actual_points": top_points,
        "top_exact_hit": top_exact_hit,
        "top_margin_or_draw_hit": top_margin_or_draw_hit,
        "top_winner_hit": top_winner_hit,
        "selected_score": selected_score,
        "selected_outcome": selected_outcome,
        "selected_expected_points": prediction.selected_expected_points,
        "actual_points": points,
        "exact_hit": exact_hit,
        "margin_or_draw_hit": margin_or_draw_hit,
        "winner_hit": winner_hit,
        "brier_1x2": brier,
        "log_loss_1x2": log_loss,
        "scoreline_probability": scoreline_probability,
        "scoreline_log_loss": scoreline_log_loss,
        "status": prediction.status,
        "warnings": warnings,
    }


def _is_reference_model(backtest_config: dict[str, Any], model_id: str) -> bool:
    return model_id in set(backtest_config.get("reference_models") or [])


def _score_pick(candidate_score: str, actual_score: str, scoring_config: dict[str, Any]) -> tuple[float, int, int, int]:
    candidate_a, candidate_b = parse_score(candidate_score)
    actual_a, actual_b = parse_score(actual_score)
    exact_points = float(scoring_config.get("exact_score", 5))
    margin_points = float(scoring_config.get("same_margin_or_draw", scoring_config.get("margin_or_draw", 3)))
    winner_points = float(scoring_config.get("winner", 1))
    exact_hit = int(candidate_a == actual_a and candidate_b == actual_b)
    candidate_outcome = outcome_1x2(candidate_a, candidate_b)
    actual_outcome = outcome_1x2(actual_a, actual_b)
    margin_or_draw_hit = int(
        (candidate_outcome == "X" and actual_outcome == "X")
        or ((candidate_a - candidate_b) == (actual_a - actual_b))
    )
    winner_hit = int(candidate_outcome == actual_outcome)
    if exact_hit:
        return exact_points, exact_hit, margin_or_draw_hit, winner_hit
    if margin_or_draw_hit:
        return margin_points, exact_hit, margin_or_draw_hit, winner_hit
    if winner_hit:
        return winner_points, exact_hit, margin_or_draw_hit, winner_hit
    return 0.0, exact_hit, margin_or_draw_hit, winner_hit


def _metric_rows(
    backtest_run_id: str,
    prediction_rows: list[dict[str, Any]],
    scoring_config: dict[str, Any],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        buckets[(row["model_id"], "all")].append(row)
        buckets[(row["model_id"], str(row["year"]))].append(row)
    metrics = []
    for (model_id, year_label), rows in sorted(buckets.items()):
        n = len(rows)
        if n == 0:
            continue
        max_possible_points = n * float(scoring_config.get("exact_score", 5))
        actual_draws = [row for row in rows if row["actual_outcome"] == "X"]
        perspectives = [
            ("max_points", "actual_points", "exact_hit", "margin_or_draw_hit", "winner_hit", "selected_outcome"),
            (
                "most_probable",
                "top_actual_points",
                "top_exact_hit",
                "top_margin_or_draw_hit",
                "top_winner_hit",
                "top_outcome",
            ),
        ]
        for perspective, points_key, exact_key, margin_key, winner_key, outcome_key in perspectives:
            total_points = sum(float(row.get(points_key) or 0.0) for row in rows)
            draw_predictions = [row for row in rows if row.get(outcome_key) == "X"]
            correct_draw_predictions = [row for row in draw_predictions if row["actual_outcome"] == "X"]
            metrics.append(
                {
                    "backtest_run_id": backtest_run_id,
                    "model_id": model_id,
                    "perspective": perspective,
                    "year": year_label,
                    "matches_evaluated": n,
                    "exact_hits": sum(int(row.get(exact_key) or 0) for row in rows),
                    "margin_or_draw_hits": sum(int(row.get(margin_key) or 0) for row in rows),
                    "winner_hits": sum(int(row.get(winner_key) or 0) for row in rows),
                    "total_quiniela_points": round(total_points, 6),
                    "max_possible_points": round(max_possible_points, 6),
                    "points_efficiency": round(total_points / max_possible_points, 6) if max_possible_points else 0.0,
                    "mean_quiniela_points": round(total_points / n, 6),
                    "exact_score_accuracy": _mean(rows, exact_key),
                    "margin_or_draw_accuracy": _mean(rows, margin_key),
                    "winner_accuracy": _mean(rows, winner_key),
                    "brier_1x2": _mean(rows, "brier_1x2"),
                    "log_loss_1x2": _mean(rows, "log_loss_1x2"),
                    "scoreline_log_loss": _mean(rows, "scoreline_log_loss"),
                    "draw_predictions": len(draw_predictions),
                    "actual_draws": len(actual_draws),
                    "draw_precision": round(len(correct_draw_predictions) / len(draw_predictions), 6)
                    if draw_predictions
                    else None,
                    "draw_recall": round(len(correct_draw_predictions) / len(actual_draws), 6) if actual_draws else None,
                }
            )
    return metrics


def _write_artifacts(
    output_dir: Path,
    backtest_run_id: str,
    as_of_utc: str,
    years: list[int],
    model_configs: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    notes: str,
) -> tuple[Path, Path]:
    json_path = output_dir / "backtest_results.json"
    csv_path = output_dir / "backtest_predictions.csv"
    payload = {
        "metadata": {
            "backtest_run_id": backtest_run_id,
            "as_of_utc": as_of_utc,
            "years": years,
            "models": [config["model_id"] for config in model_configs],
            "notes": notes,
        },
        "metrics": metrics,
        "matches": matches,
        "predictions": predictions,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = list(predictions[0]) if predictions else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    return json_path, csv_path


def _store_backtest(
    conn: sqlite3.Connection,
    backtest_run_id: str,
    as_of_utc: str,
    history_run_id: str,
    years: list[int],
    model_configs: list[dict[str, Any]],
    backtest_config: dict[str, Any],
    match_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    json_path: Path,
    csv_path: Path,
) -> None:
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO backtest_runs (
                backtest_run_id, as_of_utc, created_at_utc, history_run_id,
                years_json, models_json, config_json, matches_evaluated,
                predictions, output_json_path, output_csv_path, status, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                backtest_run_id,
                as_of_utc,
                _utc_now(),
                history_run_id,
                json.dumps(years),
                json.dumps([config["model_id"] for config in model_configs]),
                json.dumps(backtest_config, sort_keys=True),
                len(match_rows),
                len(prediction_rows),
                str(json_path),
                str(csv_path),
                "completed",
                str(backtest_config.get("notes", "")),
            ),
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO backtest_matches (
                backtest_run_id, match_id, year, match_number, match_date,
                stage, team_a_name, team_b_name, actual_score, actual_outcome,
                country, neutral, tournament
            )
            VALUES (
                :backtest_run_id, :match_id, :year, :match_number, :match_date,
                :stage, :team_a_name, :team_b_name, :actual_score,
                :actual_outcome, :country, :neutral, :tournament
            )
            """,
            match_rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO backtest_predictions (
                backtest_run_id, model_id, model_version, match_id, year,
                match_number, match_date, stage, team_a_name, team_b_name,
                actual_score, actual_outcome, expected_goals_a, expected_goals_b,
                p_team_a_win, p_draw, p_team_b_win, score_matrix_json, top_score,
                top_score_probability, top_outcome, top_actual_points,
                top_exact_hit, top_margin_or_draw_hit, top_winner_hit,
                selected_score, selected_outcome, selected_expected_points, actual_points, exact_hit,
                margin_or_draw_hit, winner_hit, brier_1x2, log_loss_1x2,
                scoreline_probability, scoreline_log_loss, status, warnings
            )
            VALUES (
                :backtest_run_id, :model_id, :model_version, :match_id, :year,
                :match_number, :match_date, :stage, :team_a_name, :team_b_name,
                :actual_score, :actual_outcome, :expected_goals_a,
                :expected_goals_b, :p_team_a_win, :p_draw, :p_team_b_win,
                :score_matrix_json, :top_score, :top_score_probability, :top_outcome,
                :top_actual_points, :top_exact_hit, :top_margin_or_draw_hit,
                :top_winner_hit, :selected_score, :selected_outcome,
                :selected_expected_points, :actual_points, :exact_hit,
                :margin_or_draw_hit, :winner_hit, :brier_1x2,
                :log_loss_1x2, :scoreline_probability, :scoreline_log_loss,
                :status, :warnings
            )
            """,
            prediction_rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO backtest_model_metrics (
                backtest_run_id, model_id, year, matches_evaluated,
                exact_hits, margin_or_draw_hits, winner_hits,
                total_quiniela_points, max_possible_points,
                points_efficiency, mean_quiniela_points,
                exact_score_accuracy, margin_or_draw_accuracy,
                winner_accuracy, brier_1x2, log_loss_1x2,
                scoreline_log_loss, draw_predictions, actual_draws,
                draw_precision, draw_recall
            )
            VALUES (
                :backtest_run_id, :model_id, :year, :matches_evaluated,
                :exact_hits, :margin_or_draw_hits, :winner_hits,
                :total_quiniela_points, :max_possible_points,
                :points_efficiency, :mean_quiniela_points,
                :exact_score_accuracy, :margin_or_draw_accuracy,
                :winner_accuracy, :brier_1x2, :log_loss_1x2,
                :scoreline_log_loss, :draw_predictions, :actual_draws,
                :draw_precision, :draw_recall
            )
            """,
            [row for row in metric_rows if row.get("perspective", "max_points") == "max_points"],
        )


def _ensure_backtest_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            backtest_run_id TEXT PRIMARY KEY,
            as_of_utc TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            history_run_id TEXT NOT NULL,
            years_json TEXT NOT NULL,
            models_json TEXT NOT NULL,
            config_json TEXT NOT NULL,
            matches_evaluated INTEGER NOT NULL,
            predictions INTEGER NOT NULL,
            output_json_path TEXT,
            output_csv_path TEXT,
            status TEXT NOT NULL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS backtest_matches (
            backtest_run_id TEXT NOT NULL,
            match_id TEXT NOT NULL,
            year INTEGER NOT NULL,
            match_number INTEGER NOT NULL,
            match_date TEXT NOT NULL,
            stage TEXT NOT NULL,
            team_a_name TEXT NOT NULL,
            team_b_name TEXT NOT NULL,
            actual_score TEXT NOT NULL,
            actual_outcome TEXT NOT NULL,
            country TEXT,
            neutral INTEGER,
            tournament TEXT,
            PRIMARY KEY (backtest_run_id, match_id)
        );

        CREATE TABLE IF NOT EXISTS backtest_predictions (
            backtest_run_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            match_id TEXT NOT NULL,
            year INTEGER NOT NULL,
            match_number INTEGER NOT NULL,
            match_date TEXT NOT NULL,
            stage TEXT NOT NULL,
            team_a_name TEXT NOT NULL,
            team_b_name TEXT NOT NULL,
            actual_score TEXT NOT NULL,
            actual_outcome TEXT NOT NULL,
            expected_goals_a REAL,
            expected_goals_b REAL,
            p_team_a_win REAL,
            p_draw REAL,
            p_team_b_win REAL,
            score_matrix_json TEXT,
            top_score TEXT,
            top_score_probability REAL,
            top_outcome TEXT,
            top_actual_points REAL NOT NULL DEFAULT 0,
            top_exact_hit INTEGER NOT NULL DEFAULT 0,
            top_margin_or_draw_hit INTEGER NOT NULL DEFAULT 0,
            top_winner_hit INTEGER NOT NULL DEFAULT 0,
            selected_score TEXT,
            selected_outcome TEXT,
            selected_expected_points REAL,
            actual_points REAL NOT NULL,
            exact_hit INTEGER NOT NULL,
            margin_or_draw_hit INTEGER NOT NULL,
            winner_hit INTEGER NOT NULL,
            brier_1x2 REAL,
            log_loss_1x2 REAL,
            scoreline_probability REAL,
            scoreline_log_loss REAL,
            status TEXT NOT NULL,
            warnings TEXT,
            PRIMARY KEY (backtest_run_id, model_id, match_id)
        );

        CREATE TABLE IF NOT EXISTS backtest_model_metrics (
            backtest_run_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            year TEXT NOT NULL,
            matches_evaluated INTEGER NOT NULL,
            exact_hits INTEGER NOT NULL,
            margin_or_draw_hits INTEGER NOT NULL,
            winner_hits INTEGER NOT NULL,
            total_quiniela_points REAL NOT NULL,
            max_possible_points REAL NOT NULL DEFAULT 0,
            points_efficiency REAL NOT NULL DEFAULT 0,
            mean_quiniela_points REAL NOT NULL,
            exact_score_accuracy REAL NOT NULL,
            margin_or_draw_accuracy REAL NOT NULL,
            winner_accuracy REAL NOT NULL,
            brier_1x2 REAL NOT NULL,
            log_loss_1x2 REAL NOT NULL,
            scoreline_log_loss REAL NOT NULL,
            draw_predictions INTEGER NOT NULL,
            actual_draws INTEGER NOT NULL,
            draw_precision REAL,
            draw_recall REAL,
            PRIMARY KEY (backtest_run_id, model_id, year)
        );

        CREATE TABLE IF NOT EXISTS backtest_parameter_trials (
            backtest_run_id TEXT NOT NULL,
            trial_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            params_json TEXT NOT NULL,
            objective_metric TEXT NOT NULL,
            objective_value REAL,
            status TEXT NOT NULL,
            notes TEXT,
            PRIMARY KEY (backtest_run_id, trial_id)
        );

        DROP VIEW IF EXISTS v_latest_backtest_run;
        CREATE VIEW v_latest_backtest_run AS
        SELECT *
        FROM backtest_runs
        WHERE status = 'completed'
        ORDER BY created_at_utc DESC
        LIMIT 1;

        DROP VIEW IF EXISTS v_latest_backtest_model_metrics;
        CREATE VIEW v_latest_backtest_model_metrics AS
        SELECT bmm.*
        FROM backtest_model_metrics bmm
        JOIN v_latest_backtest_run lbr
          ON lbr.backtest_run_id = bmm.backtest_run_id;

        DROP VIEW IF EXISTS v_latest_backtest_predictions;
        CREATE VIEW v_latest_backtest_predictions AS
        SELECT bp.*
        FROM backtest_predictions bp
        JOIN v_latest_backtest_run lbr
          ON lbr.backtest_run_id = bp.backtest_run_id;
        """
    )
    _ensure_columns(
        conn,
        "backtest_model_metrics",
        {
            "max_possible_points": "REAL NOT NULL DEFAULT 0",
            "points_efficiency": "REAL NOT NULL DEFAULT 0",
        },
    )
    _ensure_columns(
        conn,
        "backtest_predictions",
        {
            "top_outcome": "TEXT",
            "score_matrix_json": "TEXT",
            "top_actual_points": "REAL NOT NULL DEFAULT 0",
            "top_exact_hit": "INTEGER NOT NULL DEFAULT 0",
            "top_margin_or_draw_hit": "INTEGER NOT NULL DEFAULT 0",
            "top_winner_hit": "INTEGER NOT NULL DEFAULT 0",
        },
    )


def _ensure_columns(conn: sqlite3.Connection, table_name: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    for column_name, column_type in columns.items():
        if column_name not in existing:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def _latest_history_run_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT history_run_id FROM v_latest_history_run").fetchone()
    if row is None:
        raise RuntimeError("No hay historico vigente. Ejecuta scripts/build_history.py primero.")
    return str(row["history_run_id"])


def _group_matches_by_date(matches: list[BacktestMatch]) -> dict[str, list[BacktestMatch]]:
    grouped: dict[str, list[BacktestMatch]] = defaultdict(list)
    for match in matches:
        grouped[match.match_date].append(match)
    return dict(sorted(grouped.items()))


def _match_row(backtest_run_id: str, match: BacktestMatch) -> dict[str, Any]:
    actual_score = f"{match.home_score}-{match.away_score}"
    return {
        "backtest_run_id": backtest_run_id,
        "match_id": match.match_id,
        "year": match.year,
        "match_number": match.match_number,
        "match_date": match.match_date,
        "stage": match.stage,
        "team_a_name": match.team_a_name,
        "team_b_name": match.team_b_name,
        "actual_score": actual_score,
        "actual_outcome": outcome_1x2(match.home_score, match.away_score),
        "country": match.country,
        "neutral": match.neutral,
        "tournament": match.tournament,
    }


def _infer_world_cup_stage(match_number: int) -> str:
    if match_number <= 48:
        return "group"
    if match_number <= 56:
        return "r16"
    if match_number <= 60:
        return "qf"
    if match_number <= 62:
        return "sf"
    if match_number == 63:
        return "third_place"
    return "final"


def _recency_weight(match_date: Any, cutoff_date: Any) -> float:
    age_days = max((cutoff_date - match_date).days, 0)
    age_years = age_days / 365.25
    return round(max(0.05, math.exp(-age_years / 8.0)), 6)


def _prob_actual_outcome(prediction: ModelPrediction, actual_outcome: str) -> float:
    if actual_outcome == "1":
        return float(prediction.p_team_a_win or 0.0)
    if actual_outcome == "X":
        return float(prediction.p_draw or 0.0)
    return float(prediction.p_team_b_win or 0.0)


def _scoreline_probability(prediction: ModelPrediction, actual_score: str) -> float:
    if not prediction.score_matrix:
        return 0.0
    return float(prediction.score_matrix.get("scores", {}).get(actual_score, 0.0))


def _brier_1x2(prediction: ModelPrediction, actual_outcome: str) -> float:
    probs = {
        "1": float(prediction.p_team_a_win or 0.0),
        "X": float(prediction.p_draw or 0.0),
        "2": float(prediction.p_team_b_win or 0.0),
    }
    return round(sum((probs[key] - (1.0 if key == actual_outcome else 0.0)) ** 2 for key in ("1", "X", "2")), 10)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _compact_timestamp(value: str) -> str:
    return value.replace("-", "").replace(":", "").replace(".", "").replace("+00:00", "Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
