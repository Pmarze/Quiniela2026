from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from quiniela.models.common import (
    PredictionMatch,
    TrainingMatch,
    host_bonus_for,
    normalize_team_name,
    outcome_1x2,
)


FEATURE_COLUMNS = [
    "elo_a",
    "elo_b",
    "elo_diff",
    "host_a",
    "host_b",
    "neutral",
    "global_goals_per_team",
    "a_goals_for_5",
    "a_goals_against_5",
    "a_points_5",
    "a_win_rate_5",
    "a_draw_rate_5",
    "b_goals_for_5",
    "b_goals_against_5",
    "b_points_5",
    "b_win_rate_5",
    "b_draw_rate_5",
    "a_goals_for_10",
    "a_goals_against_10",
    "a_points_10",
    "a_win_rate_10",
    "a_draw_rate_10",
    "b_goals_for_10",
    "b_goals_against_10",
    "b_points_10",
    "b_win_rate_10",
    "b_draw_rate_10",
    "a_attack_strength",
    "a_defense_weakness",
    "b_attack_strength",
    "b_defense_weakness",
    "is_world_cup",
    "is_qualifier",
    "is_friendly",
    "stage_group",
    "stage_r16",
    "stage_qf",
    "stage_sf",
    "stage_final",
    "year_norm",
]


@dataclass(frozen=True)
class HistoricalMatchRecord:
    match_id: str
    match_date: str
    team_a_key: str
    team_b_key: str
    team_a_name: str
    team_b_name: str
    home_score: int
    away_score: int
    neutral: int | None
    tournament: str | None
    country: str | None
    is_world_cup: int
    is_qualifier: int
    is_friendly: int
    importance_weight: float
    recency_weight: float
    stage: str | None = None


@dataclass(frozen=True)
class NeuralExample:
    match_id: str
    match_date: str | None
    team_a_key: str
    team_b_key: str
    team_a_id: int
    team_b_id: int
    features: list[float]
    score_index: int | None = None
    outcome_index: int | None = None
    goals: tuple[float, float] | None = None
    weight: float = 1.0


def build_team_vocabulary(
    historical_matches: Iterable[HistoricalMatchRecord | TrainingMatch],
    prediction_matches: Iterable[PredictionMatch] | None = None,
) -> dict[str, int]:
    teams = {"<UNK>"}
    for match in historical_matches:
        teams.add(match.team_a_key)
        teams.add(match.team_b_key)
    for match in prediction_matches or []:
        if match.team_a_key:
            teams.add(match.team_a_key)
        if match.team_b_key:
            teams.add(match.team_b_key)
    return {team_key: idx for idx, team_key in enumerate(sorted(teams))}


def build_examples_online(
    matches: list[HistoricalMatchRecord],
    include_match: Callable[[HistoricalMatchRecord], bool],
    team_vocab: dict[str, int],
    max_goals: int,
) -> list[NeuralExample]:
    builder = OnlineFeatureBuilder()
    examples: list[NeuralExample] = []
    current_date = None
    same_day: list[HistoricalMatchRecord] = []
    for match in sorted(matches, key=lambda item: (item.match_date, item.match_id)):
        if current_date is None:
            current_date = match.match_date
        if match.match_date != current_date:
            builder.update_many(same_day)
            same_day = []
            current_date = match.match_date
        if include_match(match) and match.home_score <= max_goals and match.away_score <= max_goals:
            examples.append(_supervised_example(match, builder, team_vocab, max_goals))
        same_day.append(match)
    if same_day:
        builder.update_many(same_day)
    return examples


def build_prediction_features(
    training_matches: list[TrainingMatch],
    prediction_matches: list[PredictionMatch],
    team_vocab: dict[str, int],
) -> list[NeuralExample]:
    builder = OnlineFeatureBuilder()
    builder.update_many([historical_record_from_training(match) for match in training_matches])
    examples: list[NeuralExample] = []
    for match in prediction_matches:
        if not match.team_a_key or not match.team_b_key:
            continue
        features = builder.features_for(
            team_a_key=match.team_a_key,
            team_b_key=match.team_b_key,
            team_a_name=match.team_a_name,
            team_b_name=match.team_b_name,
            neutral=1,
            country=match.stadium_country,
            stage=match.stage,
            is_world_cup=1,
            is_qualifier=0,
            is_friendly=0,
            match_date=(match.kickoff_utc or "")[:10] or None,
        )
        examples.append(
            NeuralExample(
                match_id=match.match_id,
                match_date=(match.kickoff_utc or "")[:10] or None,
                team_a_key=match.team_a_key,
                team_b_key=match.team_b_key,
                team_a_id=team_vocab.get(match.team_a_key, team_vocab["<UNK>"]),
                team_b_id=team_vocab.get(match.team_b_key, team_vocab["<UNK>"]),
                features=features,
            )
        )
    return examples


def historical_record_from_training(match: TrainingMatch) -> HistoricalMatchRecord:
    return HistoricalMatchRecord(
        match_id=match.historical_match_id,
        match_date=match.match_date,
        team_a_key=match.team_a_key,
        team_b_key=match.team_b_key,
        team_a_name=match.team_a_name,
        team_b_name=match.team_b_name,
        home_score=match.home_score,
        away_score=match.away_score,
        neutral=match.neutral,
        tournament=match.tournament,
        country=match.country,
        is_world_cup=match.is_world_cup,
        is_qualifier=match.is_qualifier,
        is_friendly=match.is_friendly,
        importance_weight=match.importance_weight,
        recency_weight=match.recency_weight,
        stage=match.stage,
    )


class OnlineFeatureBuilder:
    def __init__(self) -> None:
        self.elo: dict[str, float] = defaultdict(lambda: 1500.0)
        self.recent: dict[str, deque[tuple[int, int, int]]] = defaultdict(lambda: deque(maxlen=10))
        self.attack_for: dict[str, float] = defaultdict(float)
        self.attack_weight: dict[str, float] = defaultdict(float)
        self.defense_against: dict[str, float] = defaultdict(float)
        self.defense_weight: dict[str, float] = defaultdict(float)
        self.global_goals = 0.0
        self.global_team_matches = 0.0

    def update_many(self, matches: list[HistoricalMatchRecord]) -> None:
        for match in matches:
            self.update(match)

    def update(self, match: HistoricalMatchRecord) -> None:
        key_a = match.team_a_key
        key_b = match.team_b_key
        weight = max(0.05, match.importance_weight * match.recency_weight)
        self._update_elo(match, weight)
        points_a, points_b = _points(match.home_score, match.away_score)
        self.recent[key_a].append((match.home_score, match.away_score, points_a))
        self.recent[key_b].append((match.away_score, match.home_score, points_b))
        self.attack_for[key_a] += match.home_score * weight
        self.attack_for[key_b] += match.away_score * weight
        self.attack_weight[key_a] += weight
        self.attack_weight[key_b] += weight
        self.defense_against[key_a] += match.away_score * weight
        self.defense_against[key_b] += match.home_score * weight
        self.defense_weight[key_a] += weight
        self.defense_weight[key_b] += weight
        self.global_goals += (match.home_score + match.away_score) * weight
        self.global_team_matches += 2.0 * weight

    def features_for(
        self,
        team_a_key: str,
        team_b_key: str,
        team_a_name: str | None,
        team_b_name: str | None,
        neutral: int | None,
        country: str | None,
        stage: str | None,
        is_world_cup: int,
        is_qualifier: int,
        is_friendly: int,
        match_date: str | None,
    ) -> list[float]:
        host_a = 1.0 if host_bonus_for(team_a_name, country, 1.0) else 0.0
        host_b = 1.0 if host_bonus_for(team_b_name, country, 1.0) else 0.0
        elo_a = self.elo[team_a_key]
        elo_b = self.elo[team_b_key]
        global_goals = self.global_goals / self.global_team_matches if self.global_team_matches else 1.25
        stage_key = normalize_stage(stage)
        values = {
            "elo_a": elo_a / 2000.0,
            "elo_b": elo_b / 2000.0,
            "elo_diff": (elo_a - elo_b) / 400.0,
            "host_a": host_a,
            "host_b": host_b,
            "neutral": 1.0 if neutral == 1 or neutral is None else 0.0,
            "global_goals_per_team": global_goals / 3.0,
            "a_attack_strength": self._rate(self.attack_for, self.attack_weight, team_a_key, global_goals) / 3.0,
            "a_defense_weakness": self._rate(self.defense_against, self.defense_weight, team_a_key, global_goals) / 3.0,
            "b_attack_strength": self._rate(self.attack_for, self.attack_weight, team_b_key, global_goals) / 3.0,
            "b_defense_weakness": self._rate(self.defense_against, self.defense_weight, team_b_key, global_goals) / 3.0,
            "is_world_cup": float(is_world_cup),
            "is_qualifier": float(is_qualifier),
            "is_friendly": float(is_friendly),
            "stage_group": 1.0 if stage_key == "group" else 0.0,
            "stage_r16": 1.0 if stage_key == "r16" else 0.0,
            "stage_qf": 1.0 if stage_key == "qf" else 0.0,
            "stage_sf": 1.0 if stage_key == "sf" else 0.0,
            "stage_final": 1.0 if stage_key == "final" else 0.0,
            "year_norm": _year_norm(match_date),
        }
        values.update(_recent_features("a", self.recent[team_a_key], global_goals))
        values.update(_recent_features("b", self.recent[team_b_key], global_goals))
        return [float(values[column]) for column in FEATURE_COLUMNS]

    def _rate(
        self,
        totals: dict[str, float],
        weights: dict[str, float],
        team_key: str,
        fallback: float,
    ) -> float:
        if weights[team_key] <= 0:
            return fallback
        return totals[team_key] / weights[team_key]

    def _update_elo(self, match: HistoricalMatchRecord, weight: float) -> None:
        rating_a = self.elo[match.team_a_key]
        rating_b = self.elo[match.team_b_key]
        home_advantage = 0.0 if match.neutral == 1 else 40.0
        expected_a = 1.0 / (1.0 + 10 ** (-((rating_a + home_advantage) - rating_b) / 400.0))
        actual_a = _actual(match.home_score, match.away_score)
        goal_scale = math.log1p(abs(match.home_score - match.away_score)) if match.home_score != match.away_score else 1.0
        delta = 28.0 * weight * goal_scale * (actual_a - expected_a)
        self.elo[match.team_a_key] = rating_a + delta
        self.elo[match.team_b_key] = rating_b - delta


def normalize_stage(stage: str | None) -> str:
    text = normalize_team_name(stage)
    if "group" in text:
        return "group"
    if "round of 16" in text or "last 16" in text:
        return "r16"
    if "quarter" in text:
        return "qf"
    if "semi" in text:
        return "sf"
    if "final" in text and "third" not in text:
        return "final"
    return "other"


def score_index(home_score: int, away_score: int, max_goals: int) -> int:
    return home_score * (max_goals + 1) + away_score


def score_matrix_from_probs(probabilities: Iterable[float], max_goals: int) -> dict[str, Any]:
    scores = {}
    total = 0.0
    for index, probability in enumerate(probabilities):
        goals_a = index // (max_goals + 1)
        goals_b = index % (max_goals + 1)
        value = max(0.0, float(probability))
        scores[f"{goals_a}-{goals_b}"] = value
        total += value
    if total <= 0:
        uniform = 1.0 / ((max_goals + 1) ** 2)
        scores = {score: uniform for score in scores}
    else:
        scores = {score: value / total for score, value in scores.items()}
    return {"max_goals": max_goals, "scores": scores}


def _supervised_example(
    match: HistoricalMatchRecord,
    builder: OnlineFeatureBuilder,
    team_vocab: dict[str, int],
    max_goals: int,
) -> NeuralExample:
    features = builder.features_for(
        team_a_key=match.team_a_key,
        team_b_key=match.team_b_key,
        team_a_name=match.team_a_name,
        team_b_name=match.team_b_name,
        neutral=match.neutral,
        country=match.country,
        stage=match.stage,
        is_world_cup=match.is_world_cup,
        is_qualifier=match.is_qualifier,
        is_friendly=match.is_friendly,
        match_date=match.match_date,
    )
    return NeuralExample(
        match_id=match.match_id,
        match_date=match.match_date,
        team_a_key=match.team_a_key,
        team_b_key=match.team_b_key,
        team_a_id=team_vocab.get(match.team_a_key, team_vocab["<UNK>"]),
        team_b_id=team_vocab.get(match.team_b_key, team_vocab["<UNK>"]),
        features=features,
        score_index=score_index(match.home_score, match.away_score, max_goals),
        outcome_index={"1": 0, "X": 1, "2": 2}[outcome_1x2(match.home_score, match.away_score)],
        goals=(float(match.home_score), float(match.away_score)),
        weight=max(0.05, match.importance_weight * match.recency_weight),
    )


def _recent_features(prefix: str, rows: deque[tuple[int, int, int]], fallback_goals: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for window in (5, 10):
        sample = list(rows)[-window:]
        if not sample:
            goals_for = fallback_goals
            goals_against = fallback_goals
            points = 1.0
            win_rate = 0.0
            draw_rate = 0.33
        else:
            count = len(sample)
            goals_for = sum(row[0] for row in sample) / count
            goals_against = sum(row[1] for row in sample) / count
            points = sum(row[2] for row in sample) / count
            win_rate = sum(1 for row in sample if row[2] == 3) / count
            draw_rate = sum(1 for row in sample if row[2] == 1) / count
        values[f"{prefix}_goals_for_{window}"] = goals_for / 3.0
        values[f"{prefix}_goals_against_{window}"] = goals_against / 3.0
        values[f"{prefix}_points_{window}"] = points / 3.0
        values[f"{prefix}_win_rate_{window}"] = win_rate
        values[f"{prefix}_draw_rate_{window}"] = draw_rate
    return values


def _points(home_score: int, away_score: int) -> tuple[int, int]:
    if home_score > away_score:
        return 3, 0
    if home_score < away_score:
        return 0, 3
    return 1, 1


def _actual(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def _year_norm(match_date: str | None) -> float:
    if not match_date:
        return 0.0
    try:
        year = int(match_date[:4])
    except ValueError:
        return 0.0
    return max(-1.0, min(1.0, (year - 2000) / 40.0))
