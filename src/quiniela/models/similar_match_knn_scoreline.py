from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    PredictionMatch,
    TrainingMatch,
    blend_score_matrices,
    build_score_matrix,
    clamp,
    expected_goals_from_score_matrix,
    failed_prediction,
    host_bonus_for,
    mask_reason_for_match,
    masked_prediction,
    normalize_score_matrix,
    successful_prediction_from_matrix,
)
from quiniela.scoring import select_best_score


MODEL_ID = "similar_match_knn_scoreline"


@dataclass(frozen=True)
class _AnalogRecord:
    features: dict[str, float]
    score: str
    weight: float


@dataclass
class _RollingState:
    params: dict[str, Any]
    ratings: dict[str, float] = field(default_factory=dict)
    goals_for: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    goals_against: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    team_weight: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    total_goals: float = 0.0
    total_team_weight: float = 0.0

    def rating(self, team_key: str | None) -> float:
        if not team_key:
            return float(self.params["initial_rating"])
        return self.ratings.get(str(team_key), float(self.params["initial_rating"]))

    def global_goals(self) -> float:
        if self.total_team_weight <= 0.0:
            return float(self.params["base_goals_fallback"])
        return clamp(
            self.total_goals / self.total_team_weight,
            float(self.params["min_expected_goals"]),
            float(self.params["max_expected_goals"]),
        )

    def profile(self, team_key: str | None) -> dict[str, float]:
        global_goals = self.global_goals()
        if not team_key:
            return {"attack": 1.0, "defense_weakness": 1.0, "sample_weight": 0.0}
        key = str(team_key)
        fallback = float(self.params["profile_shrinkage_matches"])
        weight = self.team_weight.get(key, 0.0)
        smoothed = weight + fallback
        if smoothed <= 0.0 or global_goals <= 0.0:
            return {"attack": 1.0, "defense_weakness": 1.0, "sample_weight": weight}
        goals_for_rate = (self.goals_for[key] + global_goals * fallback) / smoothed
        goals_against_rate = (self.goals_against[key] + global_goals * fallback) / smoothed
        return {
            "attack": clamp(
                goals_for_rate / global_goals,
                float(self.params["min_strength"]),
                float(self.params["max_strength"]),
            ),
            "defense_weakness": clamp(
                goals_against_rate / global_goals,
                float(self.params["min_strength"]),
                float(self.params["max_strength"]),
            ),
            "sample_weight": weight,
        }

    def update(self, match: TrainingMatch) -> None:
        team_a = str(match.team_a_key)
        team_b = str(match.team_b_key)
        weight = max(0.05, float(match.importance_weight) * float(match.recency_weight))
        self._update_ratings(match, team_a, team_b, weight)

        self.goals_for[team_a] += match.home_score * weight
        self.goals_for[team_b] += match.away_score * weight
        self.goals_against[team_a] += match.away_score * weight
        self.goals_against[team_b] += match.home_score * weight
        self.team_weight[team_a] += weight
        self.team_weight[team_b] += weight
        self.total_goals += (match.home_score + match.away_score) * weight
        self.total_team_weight += 2.0 * weight

    def _update_ratings(self, match: TrainingMatch, team_a: str, team_b: str, weight: float) -> None:
        if match.importance_weight < float(self.params["min_importance_for_rating"]):
            return
        initial = float(self.params["initial_rating"])
        rating_a = self.ratings.setdefault(team_a, initial)
        rating_b = self.ratings.setdefault(team_b, initial)
        home_advantage = 0.0 if match.neutral == 1 else float(self.params["home_advantage"])
        expected_a = 1.0 / (1.0 + 10 ** (-((rating_a + home_advantage) - rating_b) / 400.0))
        actual_a = _actual_score(match.home_score, match.away_score)
        goal_diff_scale = math.log1p(abs(match.home_score - match.away_score)) if match.home_score != match.away_score else 1.0
        delta = float(self.params["k_factor"]) * weight * goal_diff_scale * (actual_a - expected_a)
        self.ratings[team_a] = rating_a + delta
        self.ratings[team_b] = rating_b - delta


def run_similar_match_knn_scoreline(
    context: ModelContext,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[ModelPrediction]:
    model_version = str(model_config.get("model_version", "0.1.0"))
    params = _params(model_config)
    analog_records, state = _build_analog_index(context, params)

    predictions: list[ModelPrediction] = []
    for match in context.prediction_matches:
        mask_reason = mask_reason_for_match(match)
        if mask_reason:
            predictions.append(masked_prediction(context, MODEL_ID, model_version, match, mask_reason))
            continue
        if not match.team_a_key or not match.team_b_key:
            predictions.append(failed_prediction(context, MODEL_ID, model_version, match, "faltan identificadores de equipos"))
            continue

        query_features, lambda_a, lambda_b = _prediction_features(match, state, params)
        prior_matrix = build_score_matrix(lambda_a, lambda_b, int(params["max_goals"]))
        score_matrix, warnings = _blend_analog_matrix(
            query_features=query_features,
            analog_records=analog_records,
            prior_matrix=prior_matrix,
            params=params,
        )
        selected = select_best_score(score_matrix, scoring_config)
        matrix_lambda_a, matrix_lambda_b = expected_goals_from_score_matrix(score_matrix)
        predictions.append(
            successful_prediction_from_matrix(
                context=context,
                model_id=MODEL_ID,
                model_version=model_version,
                match=match,
                lambda_a=matrix_lambda_a,
                lambda_b=matrix_lambda_b,
                score_matrix=score_matrix,
                selected_score=selected["score"],
                selected_expected_points=selected["expected_points"],
                warnings=warnings,
            )
        )
    return predictions


def _params(model_config: dict[str, Any]) -> dict[str, Any]:
    default_feature_weights = {
        "rating_diff": 1.8,
        "rating_avg": 0.6,
        "lambda_a": 1.4,
        "lambda_b": 1.4,
        "attack_a": 0.8,
        "attack_b": 0.8,
        "defense_a": 0.7,
        "defense_b": 0.7,
        "goal_env": 0.5,
        "neutral": 0.25,
        "world_cup": 0.45,
        "qualifier": 0.2,
        "friendly": 0.2,
    }
    feature_weights = dict(default_feature_weights)
    feature_weights.update({str(k): float(v) for k, v in dict(model_config.get("feature_weights", {})).items()})
    return {
        "max_goals": int(model_config.get("max_goals", 8)),
        "initial_rating": float(model_config.get("initial_rating", 1500.0)),
        "k_factor": float(model_config.get("k_factor", 28.0)),
        "home_advantage": float(model_config.get("home_advantage", 80.0)),
        "goal_scale": float(model_config.get("goal_scale", 0.45)),
        "min_expected_goals": float(model_config.get("min_expected_goals", 0.2)),
        "max_expected_goals": float(model_config.get("max_expected_goals", 4.5)),
        "base_goals_fallback": float(model_config.get("base_goals_fallback", 1.25)),
        "min_importance_for_rating": float(model_config.get("min_importance_for_rating", 0.0)),
        "min_strength": float(model_config.get("min_strength", 0.55)),
        "max_strength": float(model_config.get("max_strength", 1.80)),
        "profile_shrinkage_matches": float(model_config.get("profile_shrinkage_matches", 8.0)),
        "min_index_matches": int(model_config.get("min_index_matches", 200)),
        "neighbors": int(model_config.get("neighbors", 50)),
        "min_neighbors_for_full_weight": float(model_config.get("min_neighbors_for_full_weight", 18.0)),
        "distance_temperature": float(model_config.get("distance_temperature", 0.85)),
        "analog_shrinkage_neighbors": float(model_config.get("analog_shrinkage_neighbors", 35.0)),
        "max_analog_weight": float(model_config.get("max_analog_weight", 0.70)),
        "feature_weights": feature_weights,
    }


def _build_analog_index(context: ModelContext, params: dict[str, Any]) -> tuple[list[_AnalogRecord], _RollingState]:
    state = _RollingState(params=params)
    records: list[_AnalogRecord] = []
    min_index_matches = int(params["min_index_matches"])
    for idx, match in enumerate(context.training_matches):
        if idx >= min_index_matches:
            records.extend(_training_records(match, state, params))
        state.update(match)
    return records, state


def _training_records(match: TrainingMatch, state: _RollingState, params: dict[str, Any]) -> list[_AnalogRecord]:
    country = None if match.neutral == 1 else match.country
    base_weight = max(0.05, float(match.importance_weight) * float(match.recency_weight))
    original_features, _, _ = _features(
        team_a_key=match.team_a_key,
        team_b_key=match.team_b_key,
        team_a_name=match.team_a_name,
        team_b_name=match.team_b_name,
        stadium_country=country,
        neutral=match.neutral,
        is_world_cup=match.is_world_cup,
        is_qualifier=match.is_qualifier,
        is_friendly=match.is_friendly,
        state=state,
        params=params,
    )
    mirrored_features, _, _ = _features(
        team_a_key=match.team_b_key,
        team_b_key=match.team_a_key,
        team_a_name=match.team_b_name,
        team_b_name=match.team_a_name,
        stadium_country=country,
        neutral=match.neutral,
        is_world_cup=match.is_world_cup,
        is_qualifier=match.is_qualifier,
        is_friendly=match.is_friendly,
        state=state,
        params=params,
    )
    return [
        _AnalogRecord(
            features=original_features,
            score=_cap_score(match.home_score, match.away_score, int(params["max_goals"])),
            weight=base_weight,
        ),
        _AnalogRecord(
            features=mirrored_features,
            score=_cap_score(match.away_score, match.home_score, int(params["max_goals"])),
            weight=base_weight,
        ),
    ]


def _prediction_features(
    match: PredictionMatch,
    state: _RollingState,
    params: dict[str, Any],
) -> tuple[dict[str, float], float, float]:
    host_a = host_bonus_for(match.team_a_name, match.stadium_country, float(params["home_advantage"]))
    host_b = host_bonus_for(match.team_b_name, match.stadium_country, float(params["home_advantage"]))
    neutral = 0 if host_a or host_b else 1
    return _features(
        team_a_key=str(match.team_a_key),
        team_b_key=str(match.team_b_key),
        team_a_name=match.team_a_name,
        team_b_name=match.team_b_name,
        stadium_country=match.stadium_country,
        neutral=neutral,
        is_world_cup=1,
        is_qualifier=0,
        is_friendly=0,
        state=state,
        params=params,
    )


def _features(
    team_a_key: str | None,
    team_b_key: str | None,
    team_a_name: str | None,
    team_b_name: str | None,
    stadium_country: str | None,
    neutral: int | None,
    is_world_cup: int,
    is_qualifier: int,
    is_friendly: int,
    state: _RollingState,
    params: dict[str, Any],
) -> tuple[dict[str, float], float, float]:
    rating_a = state.rating(team_a_key)
    rating_b = state.rating(team_b_key)
    host_a = host_bonus_for(team_a_name, stadium_country, float(params["home_advantage"]))
    host_b = host_bonus_for(team_b_name, stadium_country, float(params["home_advantage"]))
    effective_a = rating_a + host_a
    effective_b = rating_b + host_b
    rating_diff = effective_a - effective_b
    rating_avg = ((effective_a + effective_b) / 2.0) - float(params["initial_rating"])

    profile_a = state.profile(team_a_key)
    profile_b = state.profile(team_b_key)
    global_goals = state.global_goals()
    rating_multiplier_a = math.exp(float(params["goal_scale"]) * rating_diff / 400.0)
    rating_multiplier_b = math.exp(-float(params["goal_scale"]) * rating_diff / 400.0)
    lambda_a = clamp(
        global_goals * profile_a["attack"] * profile_b["defense_weakness"] * rating_multiplier_a,
        float(params["min_expected_goals"]),
        float(params["max_expected_goals"]),
    )
    lambda_b = clamp(
        global_goals * profile_b["attack"] * profile_a["defense_weakness"] * rating_multiplier_b,
        float(params["min_expected_goals"]),
        float(params["max_expected_goals"]),
    )
    return {
        "rating_diff": rating_diff / 400.0,
        "rating_avg": rating_avg / 400.0,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "attack_a": profile_a["attack"],
        "attack_b": profile_b["attack"],
        "defense_a": profile_a["defense_weakness"],
        "defense_b": profile_b["defense_weakness"],
        "goal_env": global_goals,
        "neutral": 1.0 if neutral == 1 else 0.0,
        "world_cup": float(is_world_cup),
        "qualifier": float(is_qualifier),
        "friendly": float(is_friendly),
    }, lambda_a, lambda_b


def _blend_analog_matrix(
    query_features: dict[str, float],
    analog_records: list[_AnalogRecord],
    prior_matrix: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not analog_records:
        return prior_matrix, ["sin analogos; fallback poisson"]

    distances = sorted(
        ((_distance(query_features, record.features, params), record) for record in analog_records),
        key=lambda item: item[0],
    )
    neighbors = distances[: max(1, int(params["neighbors"]))]
    sigma = max(1e-6, float(params["distance_temperature"]))
    scores: dict[str, float] = defaultdict(float)
    total_weight = 0.0
    squared_weight = 0.0
    weighted_distance = 0.0
    used = 0
    for distance, record in neighbors:
        closeness = math.exp(-0.5 * (distance / sigma) ** 2)
        weight = max(0.0, record.weight) * closeness
        if weight <= 0.0:
            continue
        scores[record.score] += weight
        total_weight += weight
        squared_weight += weight * weight
        weighted_distance += distance * weight
        used += 1

    if total_weight <= 0.0 or used == 0:
        return prior_matrix, ["analogos sin peso positivo; fallback poisson"]

    analog_matrix = normalize_score_matrix({"max_goals": int(params["max_goals"]), "scores": dict(scores)})
    effective_neighbors = (total_weight * total_weight / squared_weight) if squared_weight > 0.0 else float(used)
    reliability = effective_neighbors / (effective_neighbors + float(params["analog_shrinkage_neighbors"]))
    reliability *= min(1.0, used / max(1.0, float(params["min_neighbors_for_full_weight"])))
    reliability = clamp(reliability, 0.0, float(params["max_analog_weight"]))
    blended = blend_score_matrices(analog_matrix, prior_matrix, reliability)
    avg_distance = weighted_distance / total_weight
    warnings = [
        f"similar_match_knn: analogos={used}",
        f"eff_neighbors={effective_neighbors:.1f}",
        f"dist={avg_distance:.3f}",
        f"analog_weight={reliability:.3f}",
    ]
    return blended, warnings


def _distance(left: dict[str, float], right: dict[str, float], params: dict[str, Any]) -> float:
    total = 0.0
    weight_total = 0.0
    for name, weight in params["feature_weights"].items():
        if weight <= 0.0:
            continue
        delta = float(left.get(name, 0.0)) - float(right.get(name, 0.0))
        total += float(weight) * delta * delta
        weight_total += float(weight)
    if weight_total <= 0.0:
        return 0.0
    return math.sqrt(total / weight_total)


def _cap_score(goals_a: int, goals_b: int, max_goals: int) -> str:
    return f"{min(max_goals, int(goals_a))}-{min(max_goals, int(goals_b))}"


def _actual_score(goals_a: int, goals_b: int) -> float:
    if goals_a > goals_b:
        return 1.0
    if goals_a < goals_b:
        return 0.0
    return 0.5
