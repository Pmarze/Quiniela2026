from __future__ import annotations

from typing import Any

from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    mask_reason_for_match,
    masked_prediction,
    successful_prediction,
)
from quiniela.scoring import select_best_score


MODEL_ID = "baseline_poisson"


def run_baseline_poisson(
    context: ModelContext,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    model_version = str(model_config.get("model_version", "0.1.0"))
    max_goals = int(model_config.get("max_goals", 8))
    lambda_team = _global_weighted_goals_per_team(context)

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
        preview = successful_prediction(
            context=context,
            model_id=MODEL_ID,
            model_version=model_version,
            match=match,
            lambda_a=lambda_team,
            lambda_b=lambda_team,
            max_goals=max_goals,
            selected_score=None,
            selected_expected_points=None,
            warnings=["baseline global: no usa fuerza de equipos"],
        )
        selected = select_best_score(preview.score_matrix or {}, scoring_config)
        predictions.append(
            successful_prediction(
                context=context,
                model_id=MODEL_ID,
                model_version=model_version,
                match=match,
                lambda_a=lambda_team,
                lambda_b=lambda_team,
                max_goals=max_goals,
                selected_score=selected["score"],
                selected_expected_points=selected["expected_points"],
                warnings=["baseline global: no usa fuerza de equipos"],
            )
        )
    return predictions


def _global_weighted_goals_per_team(context: ModelContext) -> float:
    weighted_goals = 0.0
    weighted_team_matches = 0.0
    for match in context.training_matches:
        weight = match.importance_weight * match.recency_weight
        weighted_goals += (match.home_score + match.away_score) * weight
        weighted_team_matches += 2.0 * weight
    if weighted_team_matches <= 0:
        return 1.25
    return max(0.2, min(4.5, weighted_goals / weighted_team_matches))
