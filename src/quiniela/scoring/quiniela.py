from __future__ import annotations

from typing import Any

from quiniela.models.common import outcome_1x2, parse_score


def select_best_score(score_matrix: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    exact_points = float(scoring.get("exact_score", 5))
    margin_points = float(scoring.get("same_margin_or_draw", scoring.get("margin_or_draw", 3)))
    winner_points = float(scoring.get("winner", 1))

    best_score = None
    best_ev = -1.0
    best_breakdown = None
    for candidate_score in score_matrix["scores"]:
        expected_value, breakdown = expected_points_for_score(
            candidate_score=candidate_score,
            score_matrix=score_matrix,
            exact_points=exact_points,
            margin_points=margin_points,
            winner_points=winner_points,
        )
        if expected_value > best_ev:
            best_score = candidate_score
            best_ev = expected_value
            best_breakdown = breakdown
    return {
        "score": best_score,
        "expected_points": round(best_ev, 6),
        "breakdown": best_breakdown,
    }


def expected_points_for_score(
    candidate_score: str,
    score_matrix: dict[str, Any],
    exact_points: float,
    margin_points: float,
    winner_points: float,
) -> tuple[float, dict[str, float]]:
    candidate_a, candidate_b = parse_score(candidate_score)
    candidate_diff = candidate_a - candidate_b
    candidate_outcome = outcome_1x2(candidate_a, candidate_b)

    p_exact = 0.0
    p_margin_or_draw = 0.0
    p_winner = 0.0
    for actual_score, probability in score_matrix["scores"].items():
        actual_a, actual_b = parse_score(actual_score)
        actual_diff = actual_a - actual_b
        actual_outcome = outcome_1x2(actual_a, actual_b)

        if actual_score == candidate_score:
            p_exact += probability
        elif candidate_outcome == "X" and actual_outcome == "X":
            p_margin_or_draw += probability
        elif candidate_outcome != "X" and actual_diff == candidate_diff:
            p_margin_or_draw += probability
        elif actual_outcome == candidate_outcome:
            p_winner += probability

    expected_value = (
        p_exact * exact_points
        + p_margin_or_draw * margin_points
        + p_winner * winner_points
    )
    return expected_value, {
        "p_exact": round(p_exact, 10),
        "p_margin_or_draw_not_exact": round(p_margin_or_draw, 10),
        "p_winner_not_margin_not_exact": round(p_winner, 10),
    }
