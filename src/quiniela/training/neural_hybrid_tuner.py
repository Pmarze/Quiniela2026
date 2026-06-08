from __future__ import annotations

import csv
import itertools
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from quiniela.models.common import load_json_config, utc_now
from quiniela.training.neural_hybrid_trainer import train_neural_hybrid_v2


def run_neural_hybrid_tuning(
    db_path: Path,
    base_config_path: Path,
    output_root: Path,
    device_name: str | None,
    max_trials: int | None,
    fresh: bool,
) -> dict[str, Any]:
    base_config = load_json_config(base_config_path)
    tuning_config = base_config.get("tuning", {})
    objective = str(tuning_config.get("objective", "guarded_ev_points"))
    trial_limit = int(max_trials or tuning_config.get("max_trials", 48))
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "tuning_results.csv"
    summary_path = output_root / "tuning_summary.json"
    best_config_path = output_root / "best_config.json"
    trials = _trial_configs(base_config, trial_limit)
    rows: list[dict[str, Any]] = []
    print(
        f"[hybrid-v2-tune] trials={len(trials)} objective={objective} output={output_root}",
        flush=True,
    )

    for index, trial_config in enumerate(trials, start=1):
        trial_id = f"trial_{index:03d}"
        trial_dir = output_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_config_path = trial_dir / "config.json"
        trial_config_path.write_text(
            json.dumps(trial_config, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[hybrid-v2-tune][{trial_id}] params={_compact_params(trial_config)}", flush=True)
        existing_summary = trial_dir / "training_summary.json"
        if existing_summary.exists() and not fresh:
            print(f"[hybrid-v2-tune][{trial_id}] usando resultado existente", flush=True)
            metrics = json.loads(existing_summary.read_text(encoding="utf-8"))
        else:
            metrics = train_neural_hybrid_v2(
                db_path=db_path,
                config_path=trial_config_path,
                output_root=trial_dir,
                device_name=device_name,
                folds_only=True,
                final_only=False,
                resume=not fresh,
            )
        row = _score_trial(trial_id, trial_config, metrics, objective, tuning_config)
        rows.append(row)
        _write_results(results_path, rows)
        best = max(rows, key=lambda item: float(item["objective_value"]))
        if best["trial_id"] == trial_id:
            best_config_path.write_text(
                json.dumps(trial_config, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(
            f"[hybrid-v2-tune][{trial_id}] objective={row['objective_value']:.4f} "
            f"exact={row['mean_exact_accuracy']:.4f} matrix_1x2={row['mean_matrix_outcome_accuracy']:.4f} "
            f"ev_pts={row['mean_ev_points']:.4f} best={best['trial_id']}:{best['objective_value']:.4f}",
            flush=True,
        )

    rows = sorted(rows, key=lambda item: float(item["objective_value"]), reverse=True)
    summary = {
        "created_at_utc": utc_now(),
        "objective": objective,
        "trials": len(rows),
        "best_trial": rows[0] if rows else None,
        "best_config_path": str(best_config_path) if rows else None,
        "results_path": str(results_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[hybrid-v2-tune] terminado best={summary['best_trial']}", flush=True)
    return summary


def _trial_configs(base_config: dict[str, Any], max_trials: int) -> list[dict[str, Any]]:
    search_space = base_config.get("tuning", {}).get("search_space", {})
    if not search_space:
        return [base_config]
    keys = list(search_space)
    values = [list(search_space[key]) for key in keys]
    combinations = list(itertools.product(*values))
    random.Random(int(base_config.get("training", {}).get("seed", 42))).shuffle(combinations)
    configs = []
    for combo in combinations[:max_trials]:
        trial_config = deepcopy(base_config)
        trial_config["model_version"] = "0.1.0"
        for key, value in zip(keys, combo):
            _set_nested(trial_config, key.split("."), value)
        configs.append(trial_config)
    return configs


def _set_nested(payload: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = payload
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _score_trial(
    trial_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    objective: str,
    tuning_config: dict[str, Any],
) -> dict[str, Any]:
    folds = [fold for fold in metrics.get("folds", []) if fold.get("status") == "ok"]
    objective_value = _objective_value(folds, objective, tuning_config)
    training = config["training"]
    weights = training["loss_weights"]
    return {
        "trial_id": trial_id,
        "objective": objective,
        "objective_value": objective_value,
        "mean_exact_accuracy": _mean(folds, "exact_accuracy"),
        "mean_outcome_accuracy": _mean(folds, "outcome_accuracy"),
        "mean_matrix_outcome_accuracy": _mean(folds, "matrix_outcome_accuracy"),
        "mean_ev_outcome_accuracy": _mean(folds, "ev_outcome_accuracy"),
        "mean_ev_points": _mean(folds, "ev_mean_points"),
        "mean_top_points": _mean(folds, "top_mean_points"),
        "folds": len(folds),
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "dropout": training["dropout"],
        "hidden_dim": training["hidden_dim"],
        "num_blocks": training["num_blocks"],
        "team_embedding_dim": training["team_embedding_dim"],
        "scoreline_weight": weights["scoreline"],
        "outcome_weight": weights["outcome"],
        "quiniela_reward_weight": weights.get("quiniela_reward", 0.0),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _objective_value(folds: list[dict[str, Any]], objective: str, tuning_config: dict[str, Any]) -> float:
    if not folds:
        return 0.0
    if objective != "guarded_ev_points":
        return _mean(folds, objective)
    ev_points = _mean(folds, "ev_mean_points")
    exact_accuracy = _mean(folds, "exact_accuracy")
    matrix_outcome_accuracy = _mean(folds, "matrix_outcome_accuracy")
    ev_outcome_accuracy = _mean(folds, "ev_outcome_accuracy")
    guard = tuning_config.get("guardrails", {})
    min_matrix_outcome = float(guard.get("min_matrix_outcome_accuracy", 0.54))
    min_ev_outcome = float(guard.get("min_ev_outcome_accuracy", 0.54))
    matrix_penalty = max(0.0, min_matrix_outcome - matrix_outcome_accuracy)
    ev_penalty = max(0.0, min_ev_outcome - ev_outcome_accuracy)
    return (
        ev_points
        + float(guard.get("matrix_outcome_bonus", 0.45)) * matrix_outcome_accuracy
        + float(guard.get("exact_bonus", 0.15)) * exact_accuracy
        - float(guard.get("outcome_penalty", 3.5)) * (matrix_penalty + ev_penalty)
    )


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _compact_params(config: dict[str, Any]) -> str:
    training = config["training"]
    weights = training["loss_weights"]
    return (
        f"lr={training['learning_rate']} wd={training['weight_decay']} "
        f"drop={training['dropout']} hidden={training['hidden_dim']} blocks={training['num_blocks']} "
        f"emb={training['team_embedding_dim']} score_w={weights['scoreline']} "
        f"outcome_w={weights['outcome']} quiniela_w={weights.get('quiniela_reward', 0.0)}"
    )
