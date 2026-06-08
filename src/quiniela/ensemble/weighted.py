from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    normalize_score_matrix,
    outcome_1x2,
    parse_score,
    summarize_score_matrix,
    successful_prediction_from_matrix,
)
from quiniela.scoring import select_best_score


ENSEMBLE_MODEL_IDS = {
    "weighted_ensemble",
    "weighted_points_ensemble",
    "weighted_1x2_ensemble",
    "weighted_exact_ensemble",
    "calibrated_scoreline_ensemble",
}


@dataclass(frozen=True)
class EnsembleWeight:
    model_id: str
    weight: float
    source: str


def build_weighted_ensemble_predictions(
    context: ModelContext,
    predictions_by_model: dict[str, list[ModelPrediction]],
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    ensemble_model_id = str(model_config.get("model_id", "weighted_ensemble"))
    model_version = str(model_config.get("model_version", "0.1.0"))
    base_weights = _resolve_weights(context.db_path, model_config, predictions_by_model)
    by_source_match: dict[str, list[ModelPrediction]] = {}
    for base_model_id, predictions in predictions_by_model.items():
        if base_model_id in ENSEMBLE_MODEL_IDS or _is_excluded(base_model_id, model_config):
            continue
        if base_model_id not in base_weights:
            continue
        for prediction in predictions:
            by_source_match.setdefault(prediction.source_match_id, []).append(prediction)

    predictions: list[ModelPrediction] = []
    for match in context.prediction_matches:
        candidates = [
            prediction
            for prediction in by_source_match.get(match.source_match_id, [])
            if prediction.status == "ok" and prediction.score_matrix
        ]
        if not candidates:
            predictions.append(_fallback_unavailable(context, ensemble_model_id, model_version, match, "sin predicciones base validas"))
            continue
        blended_matrix, used_weights = _blend_matrices(candidates, base_weights, model_config)
        blended_matrix, calibration_warnings = _maybe_calibrate_scorelines(
            blended_matrix,
            context,
            model_config,
        )
        selected = select_best_score(blended_matrix, scoring_config)
        goals_a, goals_b = _matrix_expected_goals(blended_matrix)
        predictions.append(
            successful_prediction_from_matrix(
                context=context,
                model_id=ensemble_model_id,
                model_version=model_version,
                match=match,
                lambda_a=goals_a,
                lambda_b=goals_b,
                score_matrix=blended_matrix,
                selected_score=selected["score"],
                selected_expected_points=selected["expected_points"],
                warnings=[
                    _objective_warning(model_config),
                    _weights_warning(used_weights),
                    *calibration_warnings,
                ],
            )
        )
    return predictions


def _resolve_weights(
    db_path: Path,
    model_config: dict[str, Any],
    predictions_by_model: dict[str, list[ModelPrediction]],
) -> dict[str, EnsembleWeight]:
    available = {
        model_id
        for model_id, predictions in predictions_by_model.items()
        if model_id not in ENSEMBLE_MODEL_IDS and any(prediction.status == "ok" for prediction in predictions)
    }
    weights: dict[str, EnsembleWeight] = {}
    weight_source = str(model_config.get("weight_source", "latest_backtest"))
    if weight_source == "optimized_backtest":
        weights.update(_load_configured_weights(model_config, available, "optimized_weights", "optimized_backtest"))
    elif weight_source == "latest_backtest":
        weights.update(_load_backtest_weights(db_path, model_config, available))
    fallback_weights = {
        str(model_id): float(weight)
        for model_id, weight in dict(model_config.get("fallback_weights", {})).items()
    }
    for model_id in available:
        if model_id in weights or _is_excluded(model_id, model_config):
            continue
        weights[model_id] = EnsembleWeight(
            model_id=model_id,
            weight=fallback_weights.get(model_id, float(model_config.get("default_weight", 1.0))),
            source="fallback",
        )
    return _normalize_weight_map(weights)


def _load_configured_weights(
    model_config: dict[str, Any],
    available: set[str],
    key: str,
    source: str,
) -> dict[str, EnsembleWeight]:
    weights: dict[str, EnsembleWeight] = {}
    for model_id, weight in dict(model_config.get(key, {})).items():
        model_id = str(model_id)
        if model_id not in available or _is_excluded(model_id, model_config):
            continue
        weights[model_id] = EnsembleWeight(model_id=model_id, weight=float(weight), source=source)
    return weights


def _load_backtest_weights(
    db_path: Path,
    model_config: dict[str, Any],
    available: set[str],
) -> dict[str, EnsembleWeight]:
    if not available:
        return {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT model_id, year, total_quiniela_points, points_efficiency,
                   exact_score_accuracy, margin_or_draw_accuracy, winner_accuracy
            FROM v_latest_backtest_model_metrics
            WHERE year = 'all'
            """
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        try:
            conn.close()
        except UnboundLocalError:
            pass

    metric_weights = dict(model_config.get("metric_weights", {}))
    points_weight = float(metric_weights.get("points_efficiency", 0.55))
    exact_weight = float(metric_weights.get("exact_score_accuracy", 0.20))
    margin_weight = float(metric_weights.get("margin_or_draw_accuracy", 0.15))
    winner_weight = float(metric_weights.get("winner_accuracy", 0.10))
    floor = float(model_config.get("min_weight", 0.05))
    weights: dict[str, EnsembleWeight] = {}
    for row in rows:
        model_id = str(row["model_id"])
        if model_id not in available or _is_excluded(model_id, model_config):
            continue
        score = (
            points_weight * float(row["points_efficiency"] or 0.0)
            + exact_weight * float(row["exact_score_accuracy"] or 0.0)
            + margin_weight * float(row["margin_or_draw_accuracy"] or 0.0)
            + winner_weight * float(row["winner_accuracy"] or 0.0)
        )
        weights[model_id] = EnsembleWeight(model_id=model_id, weight=max(floor, score), source="backtest")
    return weights


def _blend_matrices(
    candidates: list[ModelPrediction],
    weights: dict[str, EnsembleWeight],
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    confidence_power = float(model_config.get("confidence_power", 0.0))
    raw_weights: dict[str, float] = {}
    scores: dict[str, float] = {}
    max_goals = 0
    for prediction in candidates:
        weight = weights.get(prediction.model_id)
        if weight is None:
            continue
        adjusted_weight = weight.weight
        if confidence_power > 0:
            confidence = max(
                float(prediction.p_team_a_win or 0.0),
                float(prediction.p_draw or 0.0),
                float(prediction.p_team_b_win or 0.0),
            )
            adjusted_weight *= max(0.001, confidence) ** confidence_power
        if adjusted_weight <= 0:
            continue
        matrix = normalize_score_matrix(prediction.score_matrix or {"scores": {}})
        max_goals = max(max_goals, int(matrix.get("max_goals", 0)))
        raw_weights[prediction.model_id] = adjusted_weight
        for score, probability in matrix["scores"].items():
            scores[score] = scores.get(score, 0.0) + adjusted_weight * probability
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        fallback = candidates[0]
        return normalize_score_matrix(fallback.score_matrix or {"scores": {}}), {fallback.model_id: 1.0}
    blended = {
        "max_goals": max_goals,
        "scores": {score: probability / total_weight for score, probability in scores.items()},
    }
    return normalize_score_matrix(blended), {
        model_id: weight / total_weight for model_id, weight in sorted(raw_weights.items())
    }


def _maybe_calibrate_scorelines(
    score_matrix: dict[str, Any],
    context: ModelContext,
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    calibration = dict(model_config.get("scoreline_calibration") or {})
    if not calibration.get("enabled"):
        return score_matrix, []

    matrix = normalize_score_matrix(score_matrix)
    max_goals = int(matrix.get("max_goals", model_config.get("max_goals", 8)))
    priors, prior_count = _historical_scoreline_priors(context, max_goals, calibration)
    if prior_count <= 0:
        return matrix, ["scoreline_calibration=skipped_no_historical_prior"]

    model_power = max(0.01, float(calibration.get("model_power", 0.74)))
    prior_power = max(0.0, float(calibration.get("prior_power", 0.34)))
    low_score_penalty = max(0.0, min(0.95, float(calibration.get("low_score_penalty", 0.18))))
    high_goal_bonus = max(0.0, float(calibration.get("high_goal_bonus", 0.04)))
    high_goal_min_total = int(calibration.get("high_goal_min_total", 3))
    epsilon = max(1e-12, float(calibration.get("epsilon", 1e-9)))
    central_low_scores = set(calibration.get("central_low_scores", ["1-0", "1-1", "0-1"]))

    summary = summarize_score_matrix(matrix)
    outcome_masses = {
        "1": float(summary["p_team_a_win"]),
        "X": float(summary["p_draw"]),
        "2": float(summary["p_team_b_win"]),
    }
    adjusted_scores: dict[str, float] = {}
    all_scores = [f"{goals_a}-{goals_b}" for goals_a in range(max_goals + 1) for goals_b in range(max_goals + 1)]
    for outcome in ("1", "X", "2"):
        mass = outcome_masses[outcome]
        outcome_scores = [score for score in all_scores if outcome_1x2(*parse_score(score)) == outcome]
        if mass <= 0 or not outcome_scores:
            continue
        raw: dict[str, float] = {}
        for score in outcome_scores:
            model_conditional = matrix["scores"].get(score, 0.0) / mass
            prior_conditional = priors[outcome].get(score, epsilon)
            goals_a, goals_b = parse_score(score)
            shape_factor = 1.0
            if score in central_low_scores:
                shape_factor *= 1.0 - low_score_penalty
            total_goals = goals_a + goals_b
            if total_goals >= high_goal_min_total:
                shape_factor *= 1.0 + high_goal_bonus * (total_goals - high_goal_min_total + 1)
            raw[score] = (
                max(epsilon, model_conditional) ** model_power
                * max(epsilon, prior_conditional) ** prior_power
                * max(epsilon, shape_factor)
            )
        raw_total = sum(raw.values())
        if raw_total <= 0:
            continue
        for score, probability in raw.items():
            adjusted_scores[score] = mass * probability / raw_total

    calibrated = normalize_score_matrix({"max_goals": max_goals, "scores": adjusted_scores})
    note = {
        "matches": prior_count,
        "model_power": round(model_power, 4),
        "prior_power": round(prior_power, 4),
        "low_score_penalty": round(low_score_penalty, 4),
        "high_goal_bonus": round(high_goal_bonus, 4),
    }
    return calibrated, ["scoreline_calibration=" + json.dumps(note, ensure_ascii=False, sort_keys=True)]


def _historical_scoreline_priors(
    context: ModelContext,
    max_goals: int,
    calibration: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], int]:
    min_year = int(calibration.get("min_year", 1974))
    world_cup_only = bool(calibration.get("world_cup_only", True))
    min_matches = int(calibration.get("min_matches", 120))
    smoothing = max(0.0, float(calibration.get("smoothing", 0.35)))
    scores = [f"{goals_a}-{goals_b}" for goals_a in range(max_goals + 1) for goals_b in range(max_goals + 1)]

    def build_counts(allow_all_tournaments: bool) -> tuple[dict[str, dict[str, float]], int]:
        counts = {
            outcome: {
                score: smoothing
                for score in scores
                if outcome_1x2(*parse_score(score)) == outcome
            }
            for outcome in ("1", "X", "2")
        }
        used = 0
        for match in context.training_matches:
            try:
                year = int(str(match.match_date)[:4])
            except ValueError:
                continue
            if year < min_year:
                continue
            if world_cup_only and not allow_all_tournaments and not match.is_world_cup:
                continue
            if allow_all_tournaments and match.is_friendly:
                continue
            goals_a = min(max_goals, int(match.home_score))
            goals_b = min(max_goals, int(match.away_score))
            score = f"{goals_a}-{goals_b}"
            outcome = outcome_1x2(goals_a, goals_b)
            counts[outcome][score] = counts[outcome].get(score, smoothing) + 1.0
            used += 1
        return counts, used

    counts, used = build_counts(allow_all_tournaments=False)
    if used < min_matches:
        counts, used = build_counts(allow_all_tournaments=True)
    if used <= 0:
        return {"1": {}, "X": {}, "2": {}}, 0

    priors = {}
    for outcome, outcome_counts in counts.items():
        total = sum(outcome_counts.values())
        priors[outcome] = {
            score: probability / total
            for score, probability in outcome_counts.items()
            if total > 0
        }
    return priors, used


def _weighted_goals(candidates: list[ModelPrediction], used_weights: dict[str, float]) -> tuple[float, float]:
    goals_a = 0.0
    goals_b = 0.0
    total = 0.0
    for prediction in candidates:
        weight = used_weights.get(prediction.model_id, 0.0)
        if weight <= 0:
            continue
        goals_a += weight * float(prediction.expected_goals_a or 0.0)
        goals_b += weight * float(prediction.expected_goals_b or 0.0)
        total += weight
    if total <= 0:
        return _matrix_expected_goals(candidates[0].score_matrix or {"scores": {}})
    return goals_a / total, goals_b / total


def _fallback_unavailable(
    context: ModelContext,
    model_id: str,
    model_version: str,
    match: Any,
    warning: str,
) -> ModelPrediction:
    from quiniela.models.common import failed_prediction, mask_reason_for_match, masked_prediction

    mask_reason = mask_reason_for_match(match)
    if mask_reason:
        return masked_prediction(context, model_id, model_version, match, mask_reason)
    return failed_prediction(context, model_id, model_version, match, warning)


def _normalize_weight_map(weights: dict[str, EnsembleWeight]) -> dict[str, EnsembleWeight]:
    total = sum(max(0.0, item.weight) for item in weights.values())
    if total <= 0:
        return {}
    return {
        model_id: EnsembleWeight(model_id=model_id, weight=max(0.0, item.weight) / total, source=item.source)
        for model_id, item in weights.items()
    }


def _is_excluded(model_id: str, model_config: dict[str, Any]) -> bool:
    return model_id in set(model_config.get("exclude_models", []))


def _weights_warning(weights: dict[str, float]) -> str:
    compact = {model_id: round(weight, 4) for model_id, weight in weights.items()}
    return "ensemble_weights=" + json.dumps(compact, ensure_ascii=False, sort_keys=True)


def _objective_warning(model_config: dict[str, Any]) -> str:
    return f"ensemble_objective={model_config.get('objective_label', model_config.get('model_id', 'weighted'))}"


def _matrix_expected_goals(score_matrix: dict[str, Any]) -> tuple[float, float]:
    matrix = normalize_score_matrix(score_matrix)
    goals_a = 0.0
    goals_b = 0.0
    for score, probability in matrix["scores"].items():
        left, right = score.split("-", 1)
        goals_a += int(left) * float(probability)
        goals_b += int(right) * float(probability)
    return goals_a, goals_b
