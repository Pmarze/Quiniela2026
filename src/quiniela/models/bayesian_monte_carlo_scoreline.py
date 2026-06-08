from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    clamp,
    host_bonus_for,
    mask_reason_for_match,
    masked_prediction,
    normalize_score_matrix,
    successful_prediction_from_matrix,
)
from quiniela.models.elo_dixon_coles import _apply_dixon_coles
from quiniela.models.elo_poisson import _fit_elo_ratings, _global_weighted_goals_per_team
from quiniela.scoring import select_best_score


MODEL_ID = "bayesian_monte_carlo_scoreline"


@dataclass(frozen=True)
class TeamProfile:
    attack_log: float
    defense_log: float
    sample_size: float


def run_bayesian_monte_carlo_scoreline(
    context: ModelContext,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    model_version = str(model_config.get("model_version", "0.1.0"))
    params = _params(model_config, context.prediction_run_id)
    ratings = _fit_elo_ratings(context, params)
    base_goals = _global_weighted_goals_per_team(context)
    team_profiles = _fit_team_profiles(context, base_goals, params)

    predictions: list[ModelPrediction] = []
    for match in context.prediction_matches:
        mask_reason = mask_reason_for_match(match)
        if mask_reason:
            predictions.append(masked_prediction(context, MODEL_ID, model_version, match, mask_reason))
            continue

        seed = _match_seed(params["seed"], context.as_of_utc, match.source_match_id, match.team_a_key, match.team_b_key)
        score_matrix, expected_goals_a, expected_goals_b = _simulate_match_score_matrix(
            match=match,
            ratings=ratings,
            profiles=team_profiles,
            base_goals=base_goals,
            params=params,
            seed=seed,
        )
        if float(params["dixon_coles_rho"]) != 0.0:
            score_matrix = _apply_dixon_coles(
                score_matrix,
                expected_goals_a,
                expected_goals_b,
                float(params["dixon_coles_rho"]),
            )
        selected = select_best_score(score_matrix, scoring_config)
        warnings = [
            f"monte_carlo_simulations={params['num_simulations']}",
            f"seed={seed}",
            "sin Opta ni mercados externos",
            f"rating_uncertainty_sd={params['rating_uncertainty_sd']}",
            f"lambda_log_sigma={params['lambda_log_sigma']}",
        ]
        predictions.append(
            successful_prediction_from_matrix(
                context=context,
                model_id=MODEL_ID,
                model_version=model_version,
                match=match,
                lambda_a=expected_goals_a,
                lambda_b=expected_goals_b,
                score_matrix=score_matrix,
                selected_score=selected["score"],
                selected_expected_points=selected["expected_points"],
                warnings=warnings,
            )
        )
    return predictions


def _params(model_config: dict[str, Any], prediction_run_id: str) -> dict[str, float | int]:
    is_backtest = prediction_run_id.startswith("backtest_")
    num_simulations = int(
        model_config.get(
            "backtest_num_simulations" if is_backtest else "num_simulations",
            model_config.get("num_simulations", 20000),
        )
    )
    return {
        "max_goals": int(model_config.get("max_goals", 8)),
        "num_simulations": max(500, num_simulations),
        "seed": int(model_config.get("seed", 20260608)),
        "initial_rating": float(model_config.get("initial_rating", 1500.0)),
        "k_factor": float(model_config.get("k_factor", 32.0)),
        "home_advantage": float(model_config.get("home_advantage", 80.0)),
        "goal_scale": float(model_config.get("goal_scale", 0.38)),
        "min_expected_goals": float(model_config.get("min_expected_goals", 0.2)),
        "max_expected_goals": float(model_config.get("max_expected_goals", 4.5)),
        "min_importance_for_rating": float(model_config.get("min_importance_for_rating", 0.0)),
        "attack_weight": float(model_config.get("attack_weight", 0.55)),
        "defense_weight": float(model_config.get("defense_weight", 0.45)),
        "profile_shrinkage_matches": float(model_config.get("profile_shrinkage_matches", 8.0)),
        "min_profile_log": float(model_config.get("min_profile_log", -0.45)),
        "max_profile_log": float(model_config.get("max_profile_log", 0.45)),
        "rating_uncertainty_sd": float(model_config.get("rating_uncertainty_sd", 45.0)),
        "lambda_log_sigma": float(model_config.get("lambda_log_sigma", 0.18)),
        "lambda_overdispersion": float(model_config.get("lambda_overdispersion", 0.15)),
        "dixon_coles_rho": float(model_config.get("dixon_coles_rho", -0.06)),
    }


def _fit_team_profiles(
    context: ModelContext,
    base_goals: float,
    params: dict[str, float | int],
) -> dict[str, TeamProfile]:
    goals_for: dict[str, float] = {}
    goals_against: dict[str, float] = {}
    sample_size: dict[str, float] = {}
    for match in context.training_matches:
        weight = max(0.01, match.importance_weight * match.recency_weight)
        for_key = match.team_a_key
        against_key = match.team_b_key
        goals_for[for_key] = goals_for.get(for_key, 0.0) + match.home_score * weight
        goals_against[for_key] = goals_against.get(for_key, 0.0) + match.away_score * weight
        sample_size[for_key] = sample_size.get(for_key, 0.0) + weight

        goals_for[against_key] = goals_for.get(against_key, 0.0) + match.away_score * weight
        goals_against[against_key] = goals_against.get(against_key, 0.0) + match.home_score * weight
        sample_size[against_key] = sample_size.get(against_key, 0.0) + weight

    shrink = float(params["profile_shrinkage_matches"])
    profiles: dict[str, TeamProfile] = {}
    for team_key, n_matches in sample_size.items():
        attack_rate = (goals_for.get(team_key, 0.0) + shrink * base_goals) / max(0.001, n_matches + shrink)
        defense_rate = (goals_against.get(team_key, 0.0) + shrink * base_goals) / max(0.001, n_matches + shrink)
        profiles[team_key] = TeamProfile(
            attack_log=clamp(
                math.log(max(0.001, attack_rate) / max(0.001, base_goals)),
                float(params["min_profile_log"]),
                float(params["max_profile_log"]),
            ),
            defense_log=clamp(
                math.log(max(0.001, defense_rate) / max(0.001, base_goals)),
                float(params["min_profile_log"]),
                float(params["max_profile_log"]),
            ),
            sample_size=n_matches,
        )
    return profiles


def _simulate_match_score_matrix(
    match: Any,
    ratings: dict[str, float],
    profiles: dict[str, TeamProfile],
    base_goals: float,
    params: dict[str, float | int],
    seed: int,
) -> tuple[dict[str, Any], float, float]:
    rng = np.random.default_rng(seed)
    simulations = int(params["num_simulations"])
    max_goals = int(params["max_goals"])
    lambda_a, lambda_b = _base_lambdas(match, ratings, profiles, base_goals, params)

    rating_noise = rng.normal(0.0, float(params["rating_uncertainty_sd"]), simulations)
    rating_multiplier_a = np.exp(float(params["goal_scale"]) * rating_noise / 400.0)
    rating_multiplier_b = np.exp(-float(params["goal_scale"]) * rating_noise / 400.0)
    lambda_noise_a = rng.lognormal(mean=-0.5 * float(params["lambda_log_sigma"]) ** 2, sigma=float(params["lambda_log_sigma"]), size=simulations)
    lambda_noise_b = rng.lognormal(mean=-0.5 * float(params["lambda_log_sigma"]) ** 2, sigma=float(params["lambda_log_sigma"]), size=simulations)

    sampled_lambda_a = lambda_a * rating_multiplier_a * lambda_noise_a
    sampled_lambda_b = lambda_b * rating_multiplier_b * lambda_noise_b
    sampled_lambda_a = _apply_gamma_overdispersion(rng, sampled_lambda_a, float(params["lambda_overdispersion"]))
    sampled_lambda_b = _apply_gamma_overdispersion(rng, sampled_lambda_b, float(params["lambda_overdispersion"]))
    sampled_lambda_a = np.clip(sampled_lambda_a, float(params["min_expected_goals"]), float(params["max_expected_goals"]))
    sampled_lambda_b = np.clip(sampled_lambda_b, float(params["min_expected_goals"]), float(params["max_expected_goals"]))

    goals_a = np.minimum(rng.poisson(sampled_lambda_a), max_goals)
    goals_b = np.minimum(rng.poisson(sampled_lambda_b), max_goals)
    counts: dict[str, float] = {}
    for left, right in zip(goals_a.tolist(), goals_b.tolist(), strict=True):
        score = f"{left}-{right}"
        counts[score] = counts.get(score, 0.0) + 1.0
    scores = {score: count / simulations for score, count in counts.items()}
    matrix = normalize_score_matrix({"max_goals": max_goals, "scores": scores})
    return matrix, float(np.mean(sampled_lambda_a)), float(np.mean(sampled_lambda_b))


def _base_lambdas(
    match: Any,
    ratings: dict[str, float],
    profiles: dict[str, TeamProfile],
    base_goals: float,
    params: dict[str, float | int],
) -> tuple[float, float]:
    rating_a = ratings.get(match.team_a_key, float(params["initial_rating"]))
    rating_b = ratings.get(match.team_b_key, float(params["initial_rating"]))
    host_a = host_bonus_for(match.team_a_name, match.stadium_country, float(params["home_advantage"]))
    host_b = host_bonus_for(match.team_b_name, match.stadium_country, float(params["home_advantage"]))
    rating_diff = (rating_a + host_a) - (rating_b + host_b)
    profile_a = profiles.get(match.team_a_key, TeamProfile(0.0, 0.0, 0.0))
    profile_b = profiles.get(match.team_b_key, TeamProfile(0.0, 0.0, 0.0))
    rating_term = float(params["goal_scale"]) * rating_diff / 400.0
    lambda_a = base_goals * math.exp(
        rating_term
        + float(params["attack_weight"]) * profile_a.attack_log
        + float(params["defense_weight"]) * profile_b.defense_log
    )
    lambda_b = base_goals * math.exp(
        -rating_term
        + float(params["attack_weight"]) * profile_b.attack_log
        + float(params["defense_weight"]) * profile_a.defense_log
    )
    return (
        clamp(lambda_a, float(params["min_expected_goals"]), float(params["max_expected_goals"])),
        clamp(lambda_b, float(params["min_expected_goals"]), float(params["max_expected_goals"])),
    )


def _apply_gamma_overdispersion(
    rng: np.random.Generator,
    lambdas: np.ndarray,
    overdispersion: float,
) -> np.ndarray:
    if overdispersion <= 0.0:
        return lambdas
    shape = 1.0 / max(1e-6, overdispersion * overdispersion)
    scale = lambdas / shape
    return rng.gamma(shape=shape, scale=scale)


def _match_seed(base_seed: int | float, *parts: Any) -> int:
    raw = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)

