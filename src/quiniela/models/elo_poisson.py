from __future__ import annotations

import math
from typing import Any

from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    clamp,
    failed_prediction,
    host_bonus_for,
    mask_reason_for_match,
    masked_prediction,
    successful_prediction,
)
from quiniela.scoring import select_best_score


MODEL_ID = "elo_poisson"


def run_elo_poisson(
    context: ModelContext,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    model_version = str(model_config.get("model_version", "0.1.0"))
    params = _params(model_config)
    ratings = _fit_elo_ratings(context, params)
    base_goals = _global_weighted_goals_per_team(context)

    predictions = []
    for match in context.prediction_matches:
        mask_reason = mask_reason_for_match(match)
        if mask_reason:
            predictions.append(
                masked_prediction(
                    context=context,
                    model_id=MODEL_ID,
                    model_version=model_version,
                    match=match,
                    mask_reason=mask_reason,
                )
            )
            continue
        if not match.team_a_key or not match.team_b_key:
            predictions.append(
                failed_prediction(
                    context=context,
                    model_id=MODEL_ID,
                    model_version=model_version,
                    match=match,
                    warning="faltan identificadores de equipos",
                )
            )
            continue

        lambda_a, lambda_b, warnings = _match_lambdas(match, ratings, base_goals, params)

        preview = successful_prediction(
            context=context,
            model_id=MODEL_ID,
            model_version=model_version,
            match=match,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            max_goals=params["max_goals"],
            selected_score=None,
            selected_expected_points=None,
            warnings=warnings,
        )
        selected = select_best_score(preview.score_matrix or {}, scoring_config)
        predictions.append(
            successful_prediction(
                context=context,
                model_id=MODEL_ID,
                model_version=model_version,
                match=match,
                lambda_a=lambda_a,
                lambda_b=lambda_b,
                max_goals=params["max_goals"],
                selected_score=selected["score"],
                selected_expected_points=selected["expected_points"],
                warnings=warnings,
            )
        )
    return predictions


def _params(model_config: dict[str, Any]) -> dict[str, float | int]:
    return {
        "max_goals": int(model_config.get("max_goals", 8)),
        "initial_rating": float(model_config.get("initial_rating", 1500.0)),
        "k_factor": float(model_config.get("k_factor", 22.0)),
        "home_advantage": float(model_config.get("home_advantage", 55.0)),
        "goal_scale": float(model_config.get("goal_scale", 0.55)),
        "min_expected_goals": float(model_config.get("min_expected_goals", 0.2)),
        "max_expected_goals": float(model_config.get("max_expected_goals", 4.5)),
        "min_importance_for_rating": float(model_config.get("min_importance_for_rating", 0.0)),
    }


def _match_lambdas(match: Any, ratings: dict[str, float], base_goals: float, params: dict[str, float | int]) -> tuple[float, float, list[str]]:
    rating_a = ratings.get(match.team_a_key, params["initial_rating"])
    rating_b = ratings.get(match.team_b_key, params["initial_rating"])
    host_a = host_bonus_for(match.team_a_name, match.stadium_country, float(params["home_advantage"]))
    host_b = host_bonus_for(match.team_b_name, match.stadium_country, float(params["home_advantage"]))
    rating_diff = (float(rating_a) + host_a) - (float(rating_b) + host_b)
    lambda_a = clamp(
        base_goals * math.exp(float(params["goal_scale"]) * rating_diff / 400.0),
        float(params["min_expected_goals"]),
        float(params["max_expected_goals"]),
    )
    lambda_b = clamp(
        base_goals * math.exp(-float(params["goal_scale"]) * rating_diff / 400.0),
        float(params["min_expected_goals"]),
        float(params["max_expected_goals"]),
    )

    warnings = []
    if match.team_a_key not in ratings:
        warnings.append(f"rating fallback para {match.team_a_name}")
    if match.team_b_key not in ratings:
        warnings.append(f"rating fallback para {match.team_b_name}")
    if host_a or host_b:
        warnings.append("incluye ajuste de anfitrion")
    return lambda_a, lambda_b, warnings


def _fit_elo_ratings(context: ModelContext, params: dict[str, float | int]) -> dict[str, float]:
    ratings: dict[str, float] = {}
    initial_rating = float(params["initial_rating"])
    k_factor = float(params["k_factor"])
    home_advantage = float(params["home_advantage"])
    min_importance = float(params["min_importance_for_rating"])

    for match in context.training_matches:
        if min_importance > 0.0 and match.importance_weight < min_importance:
            continue
        team_a = match.team_a_key
        team_b = match.team_b_key
        rating_a = ratings.setdefault(team_a, initial_rating)
        rating_b = ratings.setdefault(team_b, initial_rating)
        effective_home_advantage = 0.0 if match.neutral == 1 else home_advantage
        expected_a = 1.0 / (1.0 + 10 ** (-((rating_a + effective_home_advantage) - rating_b) / 400.0))
        actual_a = _actual_score(match.home_score, match.away_score)
        goal_diff_scale = math.log1p(abs(match.home_score - match.away_score)) if match.home_score != match.away_score else 1.0
        combined_weight = max(0.05, match.importance_weight * match.recency_weight)
        delta = k_factor * combined_weight * goal_diff_scale * (actual_a - expected_a)
        ratings[team_a] = rating_a + delta
        ratings[team_b] = rating_b - delta
    return ratings


def _actual_score(goals_a: int, goals_b: int) -> float:
    if goals_a > goals_b:
        return 1.0
    if goals_a < goals_b:
        return 0.0
    return 0.5


def _global_weighted_goals_per_team(context: ModelContext) -> float:
    weighted_goals = 0.0
    weighted_team_matches = 0.0
    for match in context.training_matches:
        weight = match.importance_weight * match.recency_weight
        weighted_goals += (match.home_score + match.away_score) * weight
        weighted_team_matches += 2.0 * weight
    if weighted_team_matches <= 0:
        return 1.25
    return weighted_goals / weighted_team_matches
