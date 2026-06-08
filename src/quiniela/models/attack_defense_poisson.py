from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    clamp,
    host_bonus_for,
    mask_reason_for_match,
    masked_prediction,
    successful_prediction,
)
from quiniela.scoring import select_best_score


MODEL_ID = "attack_defense_poisson"


def run_attack_defense_poisson(
    context: ModelContext,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    model_version = str(model_config.get("model_version", "0.1.0"))
    params = _params(model_config)
    profiles = _fit_attack_defense_profiles(context, params)
    global_goals = profiles["_global"]["goals_per_team"]

    predictions = []
    for match in context.prediction_matches:
        mask_reason = mask_reason_for_match(match)
        if mask_reason:
            predictions.append(masked_prediction(context, MODEL_ID, model_version, match, mask_reason))
            continue

        team_a = profiles.get(str(match.team_a_key), profiles["_fallback"])
        team_b = profiles.get(str(match.team_b_key), profiles["_fallback"])
        host_a = host_bonus_for(match.team_a_name, match.stadium_country, params["home_advantage"])
        host_b = host_bonus_for(match.team_b_name, match.stadium_country, params["home_advantage"])
        host_multiplier_a = math.exp(host_a / 900.0)
        host_multiplier_b = math.exp(host_b / 900.0)
        lambda_a = clamp(
            global_goals * team_a["attack"] * team_b["defense_weakness"] * host_multiplier_a,
            params["min_expected_goals"],
            params["max_expected_goals"],
        )
        lambda_b = clamp(
            global_goals * team_b["attack"] * team_a["defense_weakness"] * host_multiplier_b,
            params["min_expected_goals"],
            params["max_expected_goals"],
        )
        warnings = []
        if str(match.team_a_key) not in profiles:
            warnings.append(f"perfil ataque/defensa fallback para {match.team_a_name}")
        if str(match.team_b_key) not in profiles:
            warnings.append(f"perfil ataque/defensa fallback para {match.team_b_name}")
        if host_a or host_b:
            warnings.append("incluye ajuste de anfitrion")

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
        "home_advantage": float(model_config.get("home_advantage", 45.0)),
        "min_expected_goals": float(model_config.get("min_expected_goals", 0.2)),
        "max_expected_goals": float(model_config.get("max_expected_goals", 4.5)),
        "min_strength": float(model_config.get("min_strength", 0.45)),
        "max_strength": float(model_config.get("max_strength", 2.40)),
        "fallback_matches": float(model_config.get("fallback_matches", 4.0)),
    }


def _fit_attack_defense_profiles(context: ModelContext, params: dict[str, float | int]) -> dict[str, dict[str, float]]:
    goals_for = defaultdict(float)
    goals_against = defaultdict(float)
    weights = defaultdict(float)
    total_goals = 0.0
    total_team_weight = 0.0
    for match in context.training_matches:
        weight = max(0.05, match.importance_weight * match.recency_weight)
        goals_for[match.team_a_key] += match.home_score * weight
        goals_for[match.team_b_key] += match.away_score * weight
        goals_against[match.team_a_key] += match.away_score * weight
        goals_against[match.team_b_key] += match.home_score * weight
        weights[match.team_a_key] += weight
        weights[match.team_b_key] += weight
        total_goals += (match.home_score + match.away_score) * weight
        total_team_weight += 2.0 * weight

    global_goals = total_goals / total_team_weight if total_team_weight > 0 else 1.25
    global_goals = clamp(global_goals, float(params["min_expected_goals"]), float(params["max_expected_goals"]))
    profiles: dict[str, dict[str, float]] = {
        "_global": {"goals_per_team": global_goals},
        "_fallback": {"attack": 1.0, "defense_weakness": 1.0},
    }
    fallback_matches = float(params["fallback_matches"])
    for team_key, team_weight in weights.items():
        smoothed_weight = team_weight + fallback_matches
        goals_for_rate = (goals_for[team_key] + global_goals * fallback_matches) / smoothed_weight
        goals_against_rate = (goals_against[team_key] + global_goals * fallback_matches) / smoothed_weight
        profiles[team_key] = {
            "attack": clamp(
                goals_for_rate / global_goals,
                float(params["min_strength"]),
                float(params["max_strength"]),
            ),
            "defense_weakness": clamp(
                goals_against_rate / global_goals,
                float(params["min_strength"]),
                float(params["max_strength"]),
            ),
        }
    return profiles
