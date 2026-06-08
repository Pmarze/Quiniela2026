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


MODEL_ID = "elo_dixon_coles"


def run_elo_dixon_coles(
    context: ModelContext,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    model_version = str(model_config.get("model_version", "0.1.0"))
    params = _params(model_config)
    rho = float(model_config.get("dixon_coles_rho", -0.10))
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
        score_matrix = _apply_dixon_coles(score_matrix, lambda_a, lambda_b, rho)
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
                warnings=warnings + [f"ajuste Dixon-Coles rho={rho:.3f}"],
            )
        )
    return predictions


def _apply_dixon_coles(score_matrix: dict[str, Any], lambda_a: float, lambda_b: float, rho: float) -> dict[str, Any]:
    adjusted = {}
    for score, probability in score_matrix["scores"].items():
        goals_a, goals_b = parse_score(score)
        adjusted[score] = probability * _tau(goals_a, goals_b, lambda_a, lambda_b, rho)
    return normalize_score_matrix({"max_goals": score_matrix["max_goals"], "scores": adjusted})


def _tau(goals_a: int, goals_b: int, lambda_a: float, lambda_b: float, rho: float) -> float:
    if goals_a == 0 and goals_b == 0:
        return max(0.01, 1.0 - lambda_a * lambda_b * rho)
    if goals_a == 0 and goals_b == 1:
        return max(0.01, 1.0 + lambda_a * rho)
    if goals_a == 1 and goals_b == 0:
        return max(0.01, 1.0 + lambda_b * rho)
    if goals_a == 1 and goals_b == 1:
        return max(0.01, 1.0 - rho)
    return 1.0
