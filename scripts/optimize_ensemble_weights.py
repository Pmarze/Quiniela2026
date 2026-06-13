from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.models.common import load_json_config, normalize_score_matrix, outcome_1x2, parse_score
from quiniela.scoring.quiniela import resolve_scoring_profile


ENSEMBLE_MODEL_IDS = {
    "weighted_ensemble",
    "weighted_points_ensemble",
    "weighted_1x2_ensemble",
    "weighted_exact_ensemble",
    "calibrated_scoreline_ensemble",
}

OBJECTIVE_BY_MODEL = {
    "weighted_ensemble": "blend",
    "weighted_points_ensemble": "points",
    "calibrated_scoreline_ensemble": "points",
    "weighted_1x2_ensemble": "winner",
    "weighted_exact_ensemble": "exact",
}


@dataclass(frozen=True)
class OptimizationResult:
    model_id: str
    objective: str
    backtest_run_id: str
    base_models: list[str]
    weights: dict[str, float]
    total_points: float
    exact_hits: int
    margin_or_draw_hits: int
    winner_hits: int
    max_possible_points: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimiza pesos de ensambles usando el ultimo backtest con matrices de marcadores."
    )
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "quiniela.db"))
    parser.add_argument("--models-config", default=str(PROJECT_ROOT / "configs" / "models.yaml"))
    parser.add_argument("--scoring-config", default=str(PROJECT_ROOT / "configs" / "scoring.yaml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "backtests" / "ensemble_weight_optimization_latest.json"))
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--scoring-profile", default=None, help="Perfil de scoring (ej: 3-1-0).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models_path = Path(args.models_config)
    models_config = load_json_config(models_path)
    scoring_config = resolve_scoring_profile(load_json_config(Path(args.scoring_config)), args.scoring_profile)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        run_row = conn.execute("SELECT backtest_run_id FROM v_latest_backtest_run").fetchone()
        if run_row is None:
            raise RuntimeError("No hay backtest vigente. Ejecuta scripts/run_backtest.py primero.")
        backtest_run_id = str(run_row["backtest_run_id"])
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM v_latest_backtest_predictions
                WHERE status = 'ok'
                  AND score_matrix_json IS NOT NULL
                """
            ).fetchall()
        ]
    finally:
        conn.close()
    if not rows:
        raise RuntimeError("El ultimo backtest no tiene score_matrix_json. Ejecuta de nuevo scripts/run_backtest.py.")

    ensembles = [
        dict(model)
        for model in models_config.get("models", [])
        if str(model.get("model_id")) in ENSEMBLE_MODEL_IDS and model.get("active", True)
    ]
    rng = np.random.default_rng(args.seed)
    results: list[OptimizationResult] = []
    for ensemble_config in ensembles:
        result = optimize_one_ensemble(
            ensemble_config=ensemble_config,
            backtest_run_id=backtest_run_id,
            rows=rows,
            scoring_config=scoring_config,
            iterations=args.iterations,
            min_weight=args.min_weight,
            rng=rng,
        )
        results.append(result)
        print(
            f"{result.model_id}: {result.objective} "
            f"pts={result.total_points:.0f}/{result.max_possible_points:.0f} "
            f"exact={result.exact_hits} 1x2={result.winner_hits} "
            f"pesos={json.dumps(result.weights, sort_keys=True)}"
        )

    update_models_config(models_path, models_config, results, args.iterations, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "backtest_run_id": backtest_run_id,
        "iterations": args.iterations,
        "seed": args.seed,
        "results": [result.__dict__ for result in results],
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"config: {models_path}")
    print(f"output: {output_path}")
    return 0


def optimize_one_ensemble(
    ensemble_config: dict[str, Any],
    backtest_run_id: str,
    rows: list[dict[str, Any]],
    scoring_config: dict[str, Any],
    iterations: int,
    min_weight: float,
    rng: np.random.Generator,
) -> OptimizationResult:
    model_id = str(ensemble_config["model_id"])
    objective = OBJECTIVE_BY_MODEL.get(model_id, "points")
    max_goals = int(ensemble_config.get("max_goals", 8))
    scores = [f"{a}-{b}" for a in range(max_goals + 1) for b in range(max_goals + 1)]
    score_index = {score: idx for idx, score in enumerate(scores)}
    match_ids = sorted({str(row["match_id"]) for row in rows})
    row_by_model_match = {
        (str(row["model_id"]), str(row["match_id"])): row
        for row in rows
        if str(row["model_id"]) not in ENSEMBLE_MODEL_IDS
    }
    excluded = {str(item) for item in ensemble_config.get("exclude_models", [])}
    candidate_models = sorted(
        {
            model
            for model, _match in row_by_model_match
            if model not in excluded and all((model, match_id) in row_by_model_match for match_id in match_ids)
        }
    )
    if not candidate_models:
        raise RuntimeError(f"{model_id}: no hay modelos base completos para optimizar.")

    matrices = np.zeros((len(candidate_models), len(match_ids), len(scores)), dtype=np.float64)
    confidence_power = float(ensemble_config.get("confidence_power", 0.0))
    confidences = np.ones((len(candidate_models), len(match_ids)), dtype=np.float64)
    for model_idx, base_model_id in enumerate(candidate_models):
        for match_idx, match_id in enumerate(match_ids):
            row = row_by_model_match[(base_model_id, match_id)]
            matrix_raw = json.loads(str(row["score_matrix_json"]))
            matrix = normalize_score_matrix(matrix_raw)
            for score, probability in matrix["scores"].items():
                if score in score_index:
                    matrices[model_idx, match_idx, score_index[score]] = float(probability)
            row_sum = matrices[model_idx, match_idx].sum()
            if row_sum > 0:
                matrices[model_idx, match_idx] /= row_sum
            if confidence_power > 0:
                confidence = max(
                    float(row.get("p_team_a_win") or 0.0),
                    float(row.get("p_draw") or 0.0),
                    float(row.get("p_team_b_win") or 0.0),
                )
                confidences[model_idx, match_idx] = max(0.001, confidence) ** confidence_power

    actual_scores = []
    for match_id in match_ids:
        row = rows[next(i for i, item in enumerate(rows) if str(item["match_id"]) == match_id)]
        actual_scores.append(str(row["actual_score"]))
    actual_idx = np.array([score_index[clamp_score(score, max_goals)] for score in actual_scores], dtype=np.int64)
    points_matrix, exact_matrix, margin_matrix, winner_matrix = build_metric_matrices(scores, scoring_config)
    max_possible_points = len(match_ids) * float(scoring_config.get("exact_score", 5))
    initial = initial_weights(candidate_models, ensemble_config)

    best_weights = initial
    best_metrics = evaluate_weights(
        best_weights,
        matrices,
        confidences,
        actual_idx,
        points_matrix,
        exact_matrix,
        margin_matrix,
        winner_matrix,
    )
    best_score = objective_score(objective, best_metrics, max_possible_points, len(match_ids))

    for i in range(max(0, iterations)):
        if i % 5 == 0:
            alpha = np.maximum(0.25, initial * len(candidate_models) * 2.0)
        else:
            alpha = np.ones(len(candidate_models), dtype=np.float64)
        weights = rng.dirichlet(alpha)
        if min_weight > 0:
            weights = np.maximum(min_weight, weights)
            weights = weights / weights.sum()
        metrics = evaluate_weights(
            weights,
            matrices,
            confidences,
            actual_idx,
            points_matrix,
            exact_matrix,
            margin_matrix,
            winner_matrix,
        )
        score = objective_score(objective, metrics, max_possible_points, len(match_ids))
        if score > best_score:
            best_score = score
            best_weights = weights
            best_metrics = metrics

    return OptimizationResult(
        model_id=model_id,
        objective=objective,
        backtest_run_id=backtest_run_id,
        base_models=candidate_models,
        weights={model: round(float(weight), 6) for model, weight in zip(candidate_models, best_weights)},
        total_points=float(best_metrics["points"]),
        exact_hits=int(best_metrics["exact"]),
        margin_or_draw_hits=int(best_metrics["margin"]),
        winner_hits=int(best_metrics["winner"]),
        max_possible_points=float(max_possible_points),
        score=float(best_score),
    )


def evaluate_weights(
    weights: np.ndarray,
    matrices: np.ndarray,
    confidences: np.ndarray,
    actual_idx: np.ndarray,
    points_matrix: np.ndarray,
    exact_matrix: np.ndarray,
    margin_matrix: np.ndarray,
    winner_matrix: np.ndarray,
) -> dict[str, float]:
    effective_weights = weights[:, None] * confidences
    blend = np.sum(matrices * effective_weights[:, :, None], axis=0)
    row_sums = blend.sum(axis=1, keepdims=True)
    blend = np.divide(blend, row_sums, out=np.zeros_like(blend), where=row_sums > 0)
    expected_points = blend @ points_matrix.T
    selected_idx = expected_points.argmax(axis=1)
    rows = np.arange(len(actual_idx))
    return {
        "points": float(points_matrix[selected_idx, actual_idx].sum()),
        "exact": float(exact_matrix[selected_idx, actual_idx].sum()),
        "margin": float(margin_matrix[selected_idx, actual_idx].sum()),
        "winner": float(winner_matrix[selected_idx, actual_idx].sum()),
    }


def objective_score(objective: str, metrics: dict[str, float], max_points: float, n_matches: int) -> float:
    points_eff = metrics["points"] / max(1.0, max_points)
    exact_acc = metrics["exact"] / max(1, n_matches)
    margin_acc = metrics["margin"] / max(1, n_matches)
    winner_acc = metrics["winner"] / max(1, n_matches)
    if objective == "exact":
        return exact_acc + 0.10 * points_eff + 0.01 * winner_acc
    if objective == "winner":
        return winner_acc + 0.12 * points_eff + 0.02 * exact_acc
    if objective == "blend":
        return 0.55 * points_eff + 0.20 * exact_acc + 0.15 * margin_acc + 0.10 * winner_acc
    return points_eff + 0.05 * exact_acc + 0.02 * margin_acc + 0.01 * winner_acc


def build_metric_matrices(
    scores: list[str],
    scoring_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    exact_points = float(scoring_config.get("exact_score", 5))
    margin_points = float(scoring_config.get("same_margin_or_draw", scoring_config.get("margin_or_draw", 3)))
    winner_points = float(scoring_config.get("winner", 1))
    n = len(scores)
    points = np.zeros((n, n), dtype=np.float64)
    exact = np.zeros((n, n), dtype=np.float64)
    margin = np.zeros((n, n), dtype=np.float64)
    winner = np.zeros((n, n), dtype=np.float64)
    parsed = [parse_score(score) for score in scores]
    outcomes = [outcome_1x2(a, b) for a, b in parsed]
    for pred_idx, (pred_a, pred_b) in enumerate(parsed):
        for actual_idx, (actual_a, actual_b) in enumerate(parsed):
            exact_hit = pred_a == actual_a and pred_b == actual_b
            margin_hit = (outcomes[pred_idx] == "X" and outcomes[actual_idx] == "X") or (
                pred_a - pred_b == actual_a - actual_b
            )
            winner_hit = outcomes[pred_idx] == outcomes[actual_idx]
            exact[pred_idx, actual_idx] = 1.0 if exact_hit else 0.0
            margin[pred_idx, actual_idx] = 1.0 if margin_hit else 0.0
            winner[pred_idx, actual_idx] = 1.0 if winner_hit else 0.0
            if exact_hit:
                points[pred_idx, actual_idx] = exact_points
            elif margin_hit:
                points[pred_idx, actual_idx] = margin_points
            elif winner_hit:
                points[pred_idx, actual_idx] = winner_points
    return points, exact, margin, winner


def initial_weights(base_models: list[str], ensemble_config: dict[str, Any]) -> np.ndarray:
    configured = dict(ensemble_config.get("optimized_weights") or ensemble_config.get("fallback_weights") or {})
    default = float(ensemble_config.get("default_weight", 1.0))
    weights = np.array([max(0.0, float(configured.get(model, default))) for model in base_models], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(base_models), dtype=np.float64)
    return weights / weights.sum()


def clamp_score(score: str, max_goals: int) -> str:
    left, right = parse_score(score)
    return f"{min(left, max_goals)}-{min(right, max_goals)}"


def update_models_config(
    models_path: Path,
    models_config: dict[str, Any],
    results: list[OptimizationResult],
    iterations: int,
    seed: int,
) -> None:
    by_model = {result.model_id: result for result in results}
    for model in models_config.get("models", []):
        model_id = str(model.get("model_id"))
        if model_id not in by_model:
            continue
        result = by_model[model_id]
        model["weight_source"] = "optimized_backtest"
        model["optimized_weights"] = result.weights
        model["optimization"] = {
            "method": "random_dirichlet_search_score_matrix",
            "backtest_run_id": result.backtest_run_id,
            "objective": result.objective,
            "iterations": iterations,
            "seed": seed,
            "total_points": round(result.total_points, 4),
            "max_possible_points": round(result.max_possible_points, 4),
            "exact_hits": result.exact_hits,
            "margin_or_draw_hits": result.margin_or_draw_hits,
            "winner_hits": result.winner_hits,
            "score": round(result.score, 8),
        }
    models_path.write_text(json.dumps(models_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
