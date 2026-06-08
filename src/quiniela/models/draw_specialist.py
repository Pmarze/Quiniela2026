from __future__ import annotations

from typing import Any

from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    build_score_matrix,
    failed_prediction,
    mask_reason_for_match,
    masked_prediction,
    normalize_score_matrix,
    parse_score,
    successful_prediction_from_matrix,
)
from quiniela.models.elo_poisson import _fit_elo_ratings, _global_weighted_goals_per_team, _match_lambdas, _params
from quiniela.scoring import select_best_score


MODEL_ID = "draw_specialist"


def run_draw_specialist(
    context: ModelContext,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    model_version = str(model_config.get("model_version", "0.1.0"))
    params = _params(model_config)
    max_draw_boost = float(model_config.get("max_draw_boost", 0.22))
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
        score_matrix = build_score_matrix(lambda_a, lambda_b, int(params["max_goals"]))
        rating_a = ratings.get(match.team_a_key, float(params["initial_rating"]))
        rating_b = ratings.get(match.team_b_key, float(params["initial_rating"]))
        draw_boost = _draw_boost(rating_a - rating_b, lambda_a + lambda_b, max_draw_boost)
        score_matrix = _boost_draws(score_matrix, draw_boost)
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
                warnings=warnings + [f"especialista empates boost={draw_boost:.3f}"],
            )
        )
    return predictions


def _draw_boost(rating_diff: float, total_goals: float, max_draw_boost: float) -> float:
    similarity = max(0.0, 1.0 - abs(rating_diff) / 450.0)
    low_total = max(0.0, min(1.0, (3.4 - total_goals) / 1.8))
    return max(0.0, max_draw_boost * similarity * low_total)


def _boost_draws(score_matrix: dict[str, Any], draw_boost: float) -> dict[str, Any]:
    adjusted = {}
    for score, probability in score_matrix["scores"].items():
        goals_a, goals_b = parse_score(score)
        adjusted[score] = probability * (1.0 + draw_boost if goals_a == goals_b else 1.0)
    return normalize_score_matrix({"max_goals": score_matrix["max_goals"], "scores": adjusted})
