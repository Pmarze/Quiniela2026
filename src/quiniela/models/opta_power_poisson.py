from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    clamp,
    host_bonus_for,
    mask_reason_for_match,
    masked_prediction,
    normalize_team_name,
    successful_prediction,
)
from quiniela.scoring import select_best_score


MODEL_ID = "opta_power_poisson"


def run_opta_power_poisson(
    context: ModelContext,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    model_version = str(model_config.get("model_version", "0.1.0"))
    params = _params(model_config)
    source = _load_opta_source(context, model_config)
    profiles = _team_profiles(source)
    fallback_ratings = _fit_fallback_elo(context, params)
    base_goals = _global_weighted_goals_per_team(context)
    ratings, rating_sources = _build_ratings(context, profiles, fallback_ratings, params)
    live_updates = _apply_completed_tournament_updates(context, ratings, params)

    predictions: list[ModelPrediction] = []
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

        rating_a = ratings.get(match.team_a_key or "", float(params["initial_rating"]))
        rating_b = ratings.get(match.team_b_key or "", float(params["initial_rating"]))
        host_a = host_bonus_for(match.team_a_name, match.stadium_country, float(params["home_advantage"]))
        host_b = host_bonus_for(match.team_b_name, match.stadium_country, float(params["home_advantage"]))
        rating_diff = (rating_a + host_a) - (rating_b + host_b)
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

        warnings = _warnings_for_match(
            source=source,
            match=match,
            rating_sources=rating_sources,
            host_adjusted=bool(host_a or host_b),
            live_updates=live_updates,
        )
        preview = successful_prediction(
            context=context,
            model_id=MODEL_ID,
            model_version=model_version,
            match=match,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            max_goals=int(params["max_goals"]),
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
                max_goals=int(params["max_goals"]),
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
        "home_advantage": float(model_config.get("home_advantage", 80.0)),
        "goal_scale": float(model_config.get("goal_scale", 0.38)),
        "min_expected_goals": float(model_config.get("min_expected_goals", 0.2)),
        "max_expected_goals": float(model_config.get("max_expected_goals", 4.5)),
        "opta_neutral_rating": float(model_config.get("opta_neutral_rating", 75.0)),
        "opta_rating_scale": float(model_config.get("opta_rating_scale", 10.0)),
        "opta_rating_weight": float(model_config.get("opta_rating_weight", 0.75)),
        "opta_rank_weight": float(model_config.get("opta_rank_weight", 0.45)),
        "opta_rank_floor": float(model_config.get("opta_rank_floor", 45.0)),
        "opta_rank_ceiling": float(model_config.get("opta_rank_ceiling", 100.0)),
        "opta_rank_max": float(model_config.get("opta_rank_max", 131.0)),
        "opta_rank_curve": float(model_config.get("opta_rank_curve", 0.65)),
        "fallback_k_factor": float(model_config.get("fallback_k_factor", 24.0)),
        "live_update_k_factor": float(model_config.get("live_update_k_factor", 18.0)),
    }


def _load_opta_source(context: ModelContext, model_config: dict[str, Any]) -> dict[str, Any]:
    configured_path = model_config.get("data_path")
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = context.db_path.parents[1] / path
    else:
        path = context.db_path.parents[1] / "data" / "external" / "opta" / "opta_power_ratings_20260607.json"
    if not path.exists():
        return {
            "metadata": {
                "source_name": "opta_power_missing",
                "notes": f"No existe {path}; modelo usa fallback Elo interno.",
            },
            "teams": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _team_profiles(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for row in source.get("teams", []):
        key = normalize_team_name(row.get("team"))
        if key:
            profiles[key] = dict(row)
    return profiles


def _build_ratings(
    context: ModelContext,
    profiles: dict[str, dict[str, Any]],
    fallback_ratings: dict[str, float],
    params: dict[str, float | int],
) -> tuple[dict[str, float], dict[str, str]]:
    key_to_profile_key = _profile_key_by_team_key(context)
    keys = set(fallback_ratings)
    for match in context.prediction_matches:
        if match.team_a_key:
            keys.add(match.team_a_key)
        if match.team_b_key:
            keys.add(match.team_b_key)
    ratings: dict[str, float] = {}
    sources: dict[str, str] = {}
    for key in keys:
        fallback = fallback_ratings.get(key, float(params["initial_rating"]))
        profile = profiles.get(key) or profiles.get(key_to_profile_key.get(key, ""))
        if not profile:
            ratings[key] = fallback
            sources[key] = "fallback_elo"
            continue
        opta_rating = profile.get("opta_power_rating")
        opta_rank = profile.get("opta_power_rank")
        if opta_rating is not None:
            power_rating = float(opta_rating)
            weight = float(params["opta_rating_weight"])
            sources[key] = "opta_rating"
        elif opta_rank is not None:
            power_rating = _estimate_rating_from_rank(float(opta_rank), params)
            weight = float(params["opta_rank_weight"])
            sources[key] = "opta_rank_estimate"
        else:
            ratings[key] = fallback
            sources[key] = "fallback_elo"
            continue
        opta_as_elo = float(params["initial_rating"]) + (
            power_rating - float(params["opta_neutral_rating"])
        ) * float(params["opta_rating_scale"])
        ratings[key] = weight * opta_as_elo + (1.0 - weight) * fallback
    return ratings, sources


def _profile_key_by_team_key(context: ModelContext) -> dict[str, str]:
    mapping: dict[str, str] = {}

    def register(team_key: str | None, team_name: str | None) -> None:
        if not team_key or not team_name:
            return
        profile_key = normalize_team_name(team_name)
        if profile_key:
            mapping.setdefault(team_key, profile_key)

    for match in context.training_matches:
        register(match.team_a_key, match.team_a_name)
        register(match.team_b_key, match.team_b_name)
    for match in context.prediction_matches:
        register(match.team_a_key, match.team_a_name)
        register(match.team_b_key, match.team_b_name)
    return mapping


def _estimate_rating_from_rank(rank: float, params: dict[str, float | int]) -> float:
    rank = max(1.0, rank)
    rank_max = max(1.0, float(params["opta_rank_max"]))
    progress = clamp((rank - 1.0) / max(1.0, rank_max - 1.0), 0.0, 1.0)
    curved = progress ** float(params["opta_rank_curve"])
    return clamp(
        float(params["opta_rank_ceiling"]) - 50.0 * curved,
        float(params["opta_rank_floor"]),
        float(params["opta_rank_ceiling"]),
    )


def _fit_fallback_elo(context: ModelContext, params: dict[str, float | int]) -> dict[str, float]:
    ratings: dict[str, float] = {}
    initial_rating = float(params["initial_rating"])
    k_factor = float(params["fallback_k_factor"])
    home_advantage = float(params["home_advantage"])
    for match in context.training_matches:
        team_a = match.team_a_key
        team_b = match.team_b_key
        rating_a = ratings.setdefault(team_a, initial_rating)
        rating_b = ratings.setdefault(team_b, initial_rating)
        effective_home = 0.0 if match.neutral == 1 else home_advantage
        expected_a = 1.0 / (1.0 + 10 ** (-((rating_a + effective_home) - rating_b) / 400.0))
        actual_a = _actual_score(match.home_score, match.away_score)
        goal_diff_scale = math.log1p(abs(match.home_score - match.away_score)) if match.home_score != match.away_score else 1.0
        combined_weight = max(0.05, match.importance_weight * match.recency_weight)
        delta = k_factor * combined_weight * goal_diff_scale * (actual_a - expected_a)
        ratings[team_a] = rating_a + delta
        ratings[team_b] = rating_b - delta
    return ratings


def _apply_completed_tournament_updates(
    context: ModelContext,
    ratings: dict[str, float],
    params: dict[str, float | int],
) -> int:
    try:
        conn = sqlite3.connect(context.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM v_latest_state_matches
            WHERE is_completed = 1
              AND kickoff_utc IS NOT NULL
              AND kickoff_utc < ?
            ORDER BY kickoff_utc, COALESCE(match_number, CAST(source_match_id AS INTEGER))
            """,
            (context.as_of_utc,),
        ).fetchall()
    except sqlite3.Error:
        return 0
    finally:
        try:
            conn.close()
        except UnboundLocalError:
            pass

    updates = 0
    k_factor = float(params["live_update_k_factor"])
    home_advantage = float(params["home_advantage"])
    initial_rating = float(params["initial_rating"])
    for row in rows:
        team_a = row["team_a_canonical_id"] or normalize_team_name(row["team_a_name"])
        team_b = row["team_b_canonical_id"] or normalize_team_name(row["team_b_name"])
        if not team_a or not team_b:
            continue
        try:
            goals_a = int(row["home_score"])
            goals_b = int(row["away_score"])
        except (TypeError, ValueError):
            continue
        rating_a = ratings.setdefault(team_a, initial_rating)
        rating_b = ratings.setdefault(team_b, initial_rating)
        host_a = host_bonus_for(row["team_a_name"], row["stadium_country"], home_advantage)
        host_b = host_bonus_for(row["team_b_name"], row["stadium_country"], home_advantage)
        expected_a = 1.0 / (1.0 + 10 ** (-((rating_a + host_a) - (rating_b + host_b)) / 400.0))
        actual_a = _actual_score(goals_a, goals_b)
        goal_diff_scale = math.log1p(abs(goals_a - goals_b)) if goals_a != goals_b else 1.0
        delta = k_factor * goal_diff_scale * (actual_a - expected_a)
        ratings[team_a] = rating_a + delta
        ratings[team_b] = rating_b - delta
        updates += 1
    return updates


def _warnings_for_match(
    source: dict[str, Any],
    match: Any,
    rating_sources: dict[str, str],
    host_adjusted: bool,
    live_updates: int,
) -> list[str]:
    metadata = source.get("metadata", {})
    warnings = [f"opta_source={metadata.get('as_of_utc', metadata.get('source_name', 'unknown'))}"]
    for team_key, team_name in ((match.team_a_key, match.team_a_name), (match.team_b_key, match.team_b_name)):
        source_label = rating_sources.get(team_key or "", "fallback_elo")
        if source_label == "fallback_elo":
            warnings.append(f"sin rating Opta publico para {team_name}; fallback Elo interno")
        elif source_label == "opta_rank_estimate":
            warnings.append(f"rating Opta estimado por ranking para {team_name}")
    if host_adjusted:
        warnings.append("incluye ajuste de anfitrion")
    if live_updates:
        warnings.append(f"opta_live_updates={live_updates}")
    return warnings


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
