from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from quiniela.features.neural_features import (
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS,
    HistoricalMatchRecord,
    NeuralExample,
    OnlineFeatureBuilder,
    build_team_vocabulary,
    historical_record_from_training,
    normalize_stage,
    score_index,
)
from quiniela.models.common import ModelContext, PredictionMatch, normalize_team_name, outcome_1x2
from quiniela.storage.sqlite_store import SQLiteStore


TOURNAMENT_FEATURE_COLUMNS = [
    "a_tournament_played",
    "a_tournament_points",
    "a_tournament_goals_for",
    "a_tournament_goals_against",
    "a_tournament_goal_diff",
    "a_tournament_win_rate",
    "a_tournament_draw_rate",
    "a_tournament_rest_days",
    "b_tournament_played",
    "b_tournament_points",
    "b_tournament_goals_for",
    "b_tournament_goals_against",
    "b_tournament_goal_diff",
    "b_tournament_win_rate",
    "b_tournament_draw_rate",
    "b_tournament_rest_days",
    "tournament_points_diff",
    "tournament_goal_diff_delta",
    "tournament_rest_delta",
    "is_same_tournament_active",
]

HYBRID_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + TOURNAMENT_FEATURE_COLUMNS


@dataclass
class TournamentTeamState:
    played: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    last_match_date: str | None = None


class HybridFeatureBuilder:
    def __init__(self) -> None:
        self.base = OnlineFeatureBuilder()
        self.tournament_state: dict[tuple[int, str], TournamentTeamState] = defaultdict(TournamentTeamState)

    def update_many(self, matches: list[HistoricalMatchRecord]) -> None:
        for match in matches:
            self.update(match)

    def update(self, match: HistoricalMatchRecord) -> None:
        self.base.update(match)
        if match.is_world_cup:
            year = int(match.match_date[:4])
            self._update_tournament_team(year, match.team_a_key, match.home_score, match.away_score, match.match_date)
            self._update_tournament_team(year, match.team_b_key, match.away_score, match.home_score, match.match_date)

    def features_for(
        self,
        match_date: str | None,
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
    ) -> list[float]:
        base = self.base.features_for(
            team_a_key=team_a_key,
            team_b_key=team_b_key,
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            neutral=neutral,
            country=country,
            stage=stage,
            is_world_cup=is_world_cup,
            is_qualifier=is_qualifier,
            is_friendly=is_friendly,
            match_date=match_date,
        )
        tournament = self._tournament_features(match_date, team_a_key, team_b_key)
        return base + tournament

    def _update_tournament_team(self, year: int, team_key: str, goals_for: int, goals_against: int, match_date: str) -> None:
        state = self.tournament_state[(year, team_key)]
        state.played += 1
        state.goals_for += goals_for
        state.goals_against += goals_against
        if goals_for > goals_against:
            state.points += 3
            state.wins += 1
        elif goals_for < goals_against:
            state.losses += 1
        else:
            state.points += 1
            state.draws += 1
        state.last_match_date = match_date

    def _tournament_features(self, match_date: str | None, team_a_key: str, team_b_key: str) -> list[float]:
        year = int(match_date[:4]) if match_date else 0
        state_a = self.tournament_state.get((year, team_a_key), TournamentTeamState())
        state_b = self.tournament_state.get((year, team_b_key), TournamentTeamState())
        a = _team_tournament_values(state_a, match_date)
        b = _team_tournament_values(state_b, match_date)
        return [
            *a,
            *b,
            (state_a.points - state_b.points) / 9.0,
            ((state_a.goals_for - state_a.goals_against) - (state_b.goals_for - state_b.goals_against)) / 9.0,
            (_rest_days(state_a.last_match_date, match_date) - _rest_days(state_b.last_match_date, match_date)) / 10.0,
            1.0 if state_a.played or state_b.played else 0.0,
        ]


def build_hybrid_examples_previous_day(
    matches: list[HistoricalMatchRecord],
    include_match: Callable[[HistoricalMatchRecord], bool],
    team_vocab: dict[str, int],
    max_goals: int,
) -> list[NeuralExample]:
    builder = HybridFeatureBuilder()
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
            examples.append(_hybrid_example(match, builder, team_vocab, max_goals))
        same_day.append(match)
    if same_day:
        builder.update_many(same_day)
    return examples


def build_hybrid_prediction_features(
    context: ModelContext,
    team_vocab: dict[str, int],
) -> list[NeuralExample]:
    base_records = [historical_record_from_training(match) for match in context.training_matches]
    state_records = load_completed_state_records(context.db_path)
    as_of_date = context.as_of_utc[:10]
    examples = []
    for match in context.prediction_matches:
        if not match.team_a_key or not match.team_b_key:
            continue
        match_date = (match.kickoff_utc or "")[:10] or as_of_date
        usable_records = [
            record
            for record in base_records + state_records
            if record.match_date < match_date and record.match_date <= as_of_date
        ]
        builder = HybridFeatureBuilder()
        builder.update_many(sorted(usable_records, key=lambda item: (item.match_date, item.match_id)))
        features = builder.features_for(
            match_date=match_date,
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
        )
        examples.append(
            NeuralExample(
                match_id=match.match_id,
                match_date=match_date,
                team_a_key=match.team_a_key,
                team_b_key=match.team_b_key,
                team_a_id=team_vocab.get(match.team_a_key, team_vocab["<UNK>"]),
                team_b_id=team_vocab.get(match.team_b_key, team_vocab["<UNK>"]),
                features=features,
            )
        )
    return examples


def load_completed_state_records(db_path: Path) -> list[HistoricalMatchRecord]:
    store = SQLiteStore(db_path)
    store.initialize()
    conn = store.conn
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM v_latest_state_matches
            WHERE is_completed = 1
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
            ORDER BY kickoff_utc, source_match_id
            """
        ).fetchall()
        return [_state_record_from_row(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        store.close()


def _hybrid_example(
    match: HistoricalMatchRecord,
    builder: HybridFeatureBuilder,
    team_vocab: dict[str, int],
    max_goals: int,
) -> NeuralExample:
    features = builder.features_for(
        match_date=match.match_date,
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


def _state_record_from_row(row: sqlite3.Row) -> HistoricalMatchRecord:
    match_date = str(row["kickoff_utc"] or "")[:10]
    return HistoricalMatchRecord(
        match_id=f"wc2026_state_{row['source_match_id']}",
        match_date=match_date,
        team_a_key=row["team_a_canonical_id"] or normalize_team_name(row["team_a_name"]),
        team_b_key=row["team_b_canonical_id"] or normalize_team_name(row["team_b_name"]),
        team_a_name=row["team_a_name"],
        team_b_name=row["team_b_name"],
        home_score=int(row["home_score"]),
        away_score=int(row["away_score"]),
        neutral=1,
        tournament="FIFA World Cup",
        country=row["stadium_country"],
        is_world_cup=1,
        is_qualifier=0,
        is_friendly=0,
        importance_weight=2.0,
        recency_weight=1.0,
        stage=normalize_stage(row["stage"]),
    )


def _team_tournament_values(state: TournamentTeamState, match_date: str | None) -> list[float]:
    played = max(1, state.played)
    return [
        state.played / 7.0,
        state.points / 18.0,
        state.goals_for / 18.0,
        state.goals_against / 18.0,
        (state.goals_for - state.goals_against) / 12.0,
        state.wins / played,
        state.draws / played,
        _rest_days(state.last_match_date, match_date) / 10.0,
    ]


def _rest_days(last_match_date: str | None, match_date: str | None) -> float:
    if not last_match_date or not match_date:
        return 0.0
    try:
        from datetime import date

        left = date.fromisoformat(last_match_date)
        right = date.fromisoformat(match_date)
    except ValueError:
        return 0.0
    return float(max(0, min(10, (right - left).days)))


__all__ = [
    "HYBRID_FEATURE_COLUMNS",
    "HybridFeatureBuilder",
    "build_hybrid_examples_previous_day",
    "build_hybrid_prediction_features",
    "build_team_vocabulary",
    "load_completed_state_records",
]
