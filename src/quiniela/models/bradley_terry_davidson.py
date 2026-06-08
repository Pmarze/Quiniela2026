from __future__ import annotations

import math
from typing import Any

from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    adjust_score_matrix_to_1x2,
    build_score_matrix,
    failed_prediction,
    host_bonus_for,
    mask_reason_for_match,
    masked_prediction,
    successful_prediction_from_matrix,
)
from quiniela.models.elo_poisson import _fit_elo_ratings, _global_weighted_goals_per_team, _match_lambdas, _params
from quiniela.scoring import select_best_score


MODEL_ID = "bradley_terry_davidson"


def run_bradley_terry_davidson(
    context: ModelContext,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    model_version = str(model_config.get("model_version", "0.1.0"))
    params = _params(model_config)
    rating_scale = float(model_config.get("rating_scale", 400.0))
    draw_parameter = float(model_config.get("draw_parameter", 0.65))
    ratings = _fit_elo_ratings(context, params)
    base_goals = _global_weighted_goals_per_team(context)

    predictions = []
    for match in context.prediction_matches:
        mask_reason = mask_reason_for_match(match)
        if mask_reason:
            predictions.append(masked_prediction(context, MODEL_ID, model_version, match, mask_reason))
            continue
        if not match.team_a_key or not match.team_b_key:
            predictions.append(failed_prediction(context, MODEL_ID, model_version, match, "faltan identificadores de equipos"))
            continue

        lambda_a, lambda_b, warnings = _match_lambdas(match, ratings, base_goals, params)
        raw_matrix = build_score_matrix(lambda_a, lambda_b, int(params["max_goals"]))
        target_1x2 = _davidson_1x2(match, ratings, params, rating_scale, draw_parameter)
        score_matrix = adjust_score_matrix_to_1x2(raw_matrix, target_1x2)
        selected = select_best_score(score_matrix, scoring_config)
        predictions.append(
            successful_prediction_from_matrix(
                context=context,
                model_id=MODEL_ID,
                model_version=model_version,
                match=match,
                lambda_a=lambda_a,
                lambda_b=lambda_b,
                score_matrix=score_matrix,
                selected_score=selected["score"],
                selected_expected_points=selected["expected_points"],
                warnings=warnings + [f"Davidson draw_parameter={draw_parameter:.3f}"],
            )
        )
    return predictions


def _davidson_1x2(
    match: Any,
    ratings: dict[str, float],
    params: dict[str, float | int],
    rating_scale: float,
    draw_parameter: float,
) -> dict[str, float]:
    rating_a = ratings.get(match.team_a_key, float(params["initial_rating"]))
    rating_b = ratings.get(match.team_b_key, float(params["initial_rating"]))
    host_a = host_bonus_for(match.team_a_name, match.stadium_country, float(params["home_advantage"]))
    host_b = host_bonus_for(match.team_b_name, match.stadium_country, float(params["home_advantage"]))
    strength_a = math.exp((rating_a + host_a) / rating_scale)
    strength_b = math.exp((rating_b + host_b) / rating_scale)
    draw_strength = max(0.0, draw_parameter) * math.sqrt(strength_a * strength_b)
    total = strength_a + strength_b + draw_strength
    if total <= 0:
        return {"1": 1.0 / 3.0, "X": 1.0 / 3.0, "2": 1.0 / 3.0}
    return {
        "1": strength_a / total,
        "X": draw_strength / total,
        "2": strength_b / total,
    }
