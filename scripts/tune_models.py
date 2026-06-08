#!/usr/bin/env python
"""
Grid/random search para tuning de hiperparametros de modelos de quiniela.

Pre-carga los datos de entrenamiento una vez y reutiliza por trial.
Soporta ejecucion paralela con --workers N para aprovechar multiples cores.

Uso:
  python scripts/tune_models.py --model elo_poisson
  python scripts/tune_models.py --model elo_dixon_coles --workers 8
  python scripts/tune_models.py --model draw_specialist --workers 8 --years 2014 2018 2022
  python scripts/tune_models.py --model bradley_terry_davidson --max-trials 500 --workers 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quiniela.models import (
    run_attack_defense_poisson,
    run_baseline_poisson,
    run_bradley_terry_davidson,
    run_draw_specialist,
    run_elo_dixon_coles,
    run_elo_poisson,
)
from quiniela.models.common import (
    ModelContext,
    PredictionMatch,
    TrainingMatch,
    load_json_config,
    normalize_team_name,
    outcome_1x2,
    parse_score,
)
from quiniela.storage.sqlite_store import SQLiteStore

MODEL_RUNNERS = {
    "attack_defense_poisson": run_attack_defense_poisson,
    "baseline_poisson": run_baseline_poisson,
    "bradley_terry_davidson": run_bradley_terry_davidson,
    "draw_specialist": run_draw_specialist,
    "elo_dixon_coles": run_elo_dixon_coles,
    "elo_poisson": run_elo_poisson,
}

EPSILON = 1e-12
PROJECT_ROOT = Path(__file__).parent.parent

# Espacios de busqueda expandidos para maquinas potentes.
# min_importance_for_rating: 0.0=todos, 0.7=excluye amistosos, 1.0=solo torneos importantes
SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "elo_poisson": {
        "k_factor":                  [8, 12, 16, 20, 24, 28, 32, 36, 40],
        "goal_scale":                [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "home_advantage":            [20, 30, 40, 50, 60, 70, 80, 90],
        "min_importance_for_rating": [0.0, 0.7, 1.0],
    },
    "elo_dixon_coles": {
        "k_factor":                  [8, 12, 16, 20, 24, 28, 32, 36, 40],
        "goal_scale":                [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "dixon_coles_rho":           [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05],
        "min_importance_for_rating": [0.0, 0.7, 1.0],
    },
    "draw_specialist": {
        "k_factor":                  [8, 12, 16, 20, 24, 28, 32, 36, 40],
        "goal_scale":                [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "max_draw_boost":            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "min_importance_for_rating": [0.0, 0.7],
    },
    "bradley_terry_davidson": {
        "k_factor":                  [8, 12, 16, 20, 24, 28, 32, 36, 40],
        "goal_scale":                [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "draw_parameter":            [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.50],
        "min_importance_for_rating": [0.0, 0.7],
    },
    "attack_defense_poisson": {
        "home_advantage":    [20, 30, 40, 45, 50, 55, 60, 65, 70, 80],
        "min_strength":      [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60],
        "max_strength":      [1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
        "fallback_matches":  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    },
}


# ---------------------------------------------------------------------------
# Worker (debe ser funcion de modulo para ser picklable en Windows)
# ---------------------------------------------------------------------------

def _worker(args: dict) -> list[dict[str, Any]]:
    """Carga sus propios datos del DB y evalua un lote de trials."""
    db_path = Path(args["db_path"])
    history_run_id = args["history_run_id"]
    years = args["years"]
    model_id = args["model_id"]
    base_model_config = args["base_model_config"]
    scoring_config = args["scoring_config"]
    trial_chunk: list[dict[str, Any]] = args["trial_chunk"]

    store = SQLiteStore(db_path)
    store.initialize()
    conn = store.conn
    try:
        wc_matches = _load_wc_matches(conn, history_run_id, years)
        date_groups = _group_by_date(wc_matches)
        training_by_date = {
            date: _load_training_for_date(conn, history_run_id, date)
            for date in sorted(date_groups.keys())
        }
    finally:
        store.close()

    runner = MODEL_RUNNERS[model_id]
    results = []
    for trial_params in trial_chunk:
        model_config = {**base_model_config, **trial_params}
        metrics = _evaluate(runner, model_config, scoring_config, date_groups, training_by_date, history_run_id, db_path)
        results.append({"params": trial_params, "metrics": metrics})
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tuning de hiperparametros para modelos de quiniela")
    parser.add_argument("--model",      required=True, choices=list(SEARCH_SPACES.keys()))
    parser.add_argument("--years",      nargs="+", type=int, default=[2014, 2018, 2022])
    parser.add_argument("--max-trials", type=int, default=None, help="Limite de trials en random search (default: grid completo)")
    parser.add_argument("--workers",    type=int, default=1,    help="Numero de procesos paralelos (default: 1)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--db",         type=Path, default=PROJECT_ROOT / "data" / "quiniela.db")
    parser.add_argument("--output",     type=Path, default=None)
    parser.add_argument("--top",        type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)

    models_config  = load_json_config(PROJECT_ROOT / "configs" / "models.yaml")
    scoring_config = load_json_config(PROJECT_ROOT / "configs" / "scoring.yaml")
    base_model_config = next(
        (m for m in models_config.get("models", []) if m["model_id"] == args.model),
        {"model_id": args.model, "model_version": "tuning", "max_goals": 8},
    )

    store = SQLiteStore(args.db)
    store.initialize()
    conn = store.conn
    try:
        history_run_id = _latest_history_run_id(conn)
        wc_matches     = _load_wc_matches(conn, history_run_id, args.years)
    finally:
        store.close()

    if not wc_matches:
        print(f"ERROR: No hay partidos de Mundial para los anos {args.years}")
        return

    n_total = len(wc_matches)
    space   = SEARCH_SPACES[args.model]
    keys    = list(space.keys())
    all_combos = list(itertools.product(*space.values()))

    if args.max_trials and args.max_trials < len(all_combos):
        selected = [dict(zip(keys, c)) for c in random.sample(all_combos, args.max_trials)]
        mode_str = f"random search: {len(selected)} de {len(all_combos)} posibles"
    else:
        selected = [dict(zip(keys, c)) for c in all_combos]
        mode_str = f"grid completo: {len(selected)} trials"

    n_workers = max(1, args.workers)
    print(f"Modelo: {args.model}  |  Anos: {args.years}  |  Partidos: {n_total}", flush=True)
    print(f"Modo: {mode_str}  |  Workers: {n_workers}", flush=True)

    t0 = time.time()
    if n_workers == 1:
        results = _run_sequential(selected, base_model_config, scoring_config, wc_matches,
                                  history_run_id, args.db, args.model, years=args.years)
    else:
        results = _run_parallel(n_workers, selected, base_model_config, scoring_config,
                                args.years, history_run_id, args.db, args.model)
    elapsed = time.time() - t0

    results.sort(key=lambda r: r["metrics"]["points_efficiency"], reverse=True)

    # Baseline con params actuales
    store2 = SQLiteStore(args.db)
    store2.initialize()
    conn2 = store2.conn
    try:
        dg = _group_by_date(wc_matches)
        tbd = {d: _load_training_for_date(conn2, history_run_id, d) for d in sorted(dg.keys())}
    finally:
        store2.close()

    runner   = MODEL_RUNNERS[args.model]
    baseline = _evaluate(runner, base_model_config, scoring_config, dg, tbd, history_run_id, args.db)
    baseline_params = {k: base_model_config.get(k) for k in keys}

    print(f"\nTiempo total: {elapsed/60:.1f} min  ({elapsed/len(selected):.2f}s/trial)")
    _print_report(args.model, args.years, n_total, results, baseline, baseline_params, args.top)

    output_path = args.output
    if not output_path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = PROJECT_ROOT / "data" / "backtests" / f"tuning_{args.model}_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id":        args.model,
        "years":           args.years,
        "n_matches":       n_total,
        "n_trials":        len(selected),
        "n_workers":       n_workers,
        "elapsed_seconds": round(elapsed, 1),
        "baseline_params": baseline_params,
        "baseline_metrics": baseline,
        "best_params":     results[0]["params"],
        "best_metrics":    results[0]["metrics"],
        "top_results":     results[:50],
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"\nResultados guardados: {output_path}")


# ---------------------------------------------------------------------------
# Sequential / parallel runners
# ---------------------------------------------------------------------------

def _checkpoint_path(model_id: str, years: list[int]) -> Path:
    years_str = "_".join(str(y) for y in sorted(years))
    return PROJECT_ROOT / "data" / "backtests" / f"tuning_{model_id}_{years_str}_checkpoint.json"


def _save_checkpoint(path: Path, model_id: str, years: list[int], results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"model_id": model_id, "years": years,
                    "n_completed": len(results), "completed_trials": results,
                    "saved_at": datetime.now(timezone.utc).isoformat()},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)  # escritura atomica


def _load_checkpoint(path: Path, model_id: str, years: list[int]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model_id") == model_id and data.get("years") == years:
            done = data.get("completed_trials", [])
            if done:
                print(f"Checkpoint encontrado: retomando desde {len(done)} trials ya completados.", flush=True)
            return done
    except Exception:
        pass
    return []


def _run_sequential(
    trials: list[dict[str, Any]],
    base_model_config: dict[str, Any],
    scoring_config: dict[str, Any],
    wc_matches: list[dict[str, Any]],
    history_run_id: str,
    db_path: Path,
    model_id: str,
    years: list[int] | None = None,
) -> list[dict[str, Any]]:
    cp_path = _checkpoint_path(model_id, years or [])
    done_results = _load_checkpoint(cp_path, model_id, years or [])
    done_keys = {json.dumps(r["params"], sort_keys=True) for r in done_results}
    pending = [t for t in trials if json.dumps(t, sort_keys=True) not in done_keys]

    store = SQLiteStore(db_path)
    store.initialize()
    conn = store.conn
    try:
        date_groups = _group_by_date(wc_matches)
        print(f"Precargando datos para {len(date_groups)} fechas...")
        training_by_date = {d: _load_training_for_date(conn, history_run_id, d) for d in sorted(date_groups.keys())}
    finally:
        store.close()

    runner  = MODEL_RUNNERS[model_id]
    results = list(done_results)
    t0 = time.time()
    for i, trial_params in enumerate(pending):
        model_config = {**base_model_config, **trial_params}
        metrics = _evaluate(runner, model_config, scoring_config, date_groups, training_by_date, history_run_id, db_path)
        results.append({"params": trial_params, "metrics": metrics})
        if (i + 1) % 100 == 0 or i + 1 == len(pending):
            best = max(r["metrics"]["points_efficiency"] for r in results)
            print(f"  Trial {i+1:>5}/{len(pending)} — {time.time()-t0:.0f}s — mejor eff: {best*100:.2f}%", flush=True)
            _save_checkpoint(cp_path, model_id, years or [], results)
    return results


def _run_parallel(
    n_workers: int,
    trials: list[dict[str, Any]],
    base_model_config: dict[str, Any],
    scoring_config: dict[str, Any],
    years: list[int],
    history_run_id: str,
    db_path: Path,
    model_id: str,
) -> list[dict[str, Any]]:
    cp_path = _checkpoint_path(model_id, years)
    done_results = _load_checkpoint(cp_path, model_id, years)
    done_keys = {json.dumps(r["params"], sort_keys=True) for r in done_results}
    pending = [t for t in trials if json.dumps(t, sort_keys=True) not in done_keys]

    if not pending:
        print("Todos los trials ya estaban en el checkpoint.", flush=True)
        return list(done_results)

    # Chunk size: lo suficientemente grande para amortizar el costo de spawn en Windows
    # (~15-30s por worker) pero pequeño para checkpoints frecuentes.
    # Objetivo: cada chunk tarda ~2-4 min → ~60 trials a 2s/trial.
    chunk_size = max(60, len(pending) // (n_workers * 2))
    chunks = [pending[i : i + chunk_size] for i in range(0, len(pending), chunk_size)]
    total  = len(trials)
    print(f"Trials pendientes: {len(pending)}/{total}  |  Lotes: {len(chunks)} ({chunk_size} trials/lote)  |  Workers: {n_workers}", flush=True)
    print(f"Spawn overhead Windows ~15-30s/worker — chunks grandes lo amortizan.", flush=True)

    worker_args = [
        {
            "db_path":           str(db_path),
            "history_run_id":    history_run_id,
            "years":             years,
            "model_id":          model_id,
            "base_model_config": base_model_config,
            "scoring_config":    scoring_config,
            "trial_chunk":       chunk,
        }
        for chunk in chunks
    ]

    all_results: list[dict[str, Any]] = list(done_results)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, wargs): i for i, wargs in enumerate(worker_args)}
        for future in as_completed(futures):
            batch = future.result()
            all_results.extend(batch)
            best = max(r["metrics"]["points_efficiency"] for r in all_results)
            pct  = len(all_results) / total * 100
            elapsed = time.time() - t0
            print(f"  Completado: {len(all_results):>5}/{total} ({pct:.0f}%) — "
                  f"{elapsed:.0f}s — mejor eff: {best*100:.2f}%", flush=True)
            _save_checkpoint(cp_path, model_id, years, all_results)
    return all_results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    runner: Any,
    model_config: dict[str, Any],
    scoring_config: dict[str, Any],
    date_groups: dict[str, list[dict[str, Any]]],
    training_by_date: dict[str, list[TrainingMatch]],
    history_run_id: str,
    db_path: Path,
) -> dict[str, float]:
    run_id = f"t_{uuid.uuid4().hex[:8]}"
    total_points = max_points = brier_sum = logloss_sum = 0.0
    exact_hits = winner_hits = margin_hits = n = 0
    exact_pts = float(scoring_config.get("exact_score", 5))

    for date, day_matches in sorted(date_groups.items()):
        context = ModelContext(
            db_path=db_path,
            as_of_utc=f"{date}T00:00:00Z",
            prediction_run_id=run_id,
            tournament_state_id=f"tuning_{date}",
            input_snapshot_id=history_run_id,
            training_data_version=history_run_id,
            training_matches=training_by_date[date],
            prediction_matches=[_to_prediction_match(m) for m in day_matches],
        )
        actual_by_id = {m["match_id"]: m for m in day_matches}

        for pred in runner(context, model_config, scoring_config):
            if pred.status != "ok":
                continue
            actual   = actual_by_id[pred.match_id]
            selected = pred.selected_score or pred.top_score
            if not selected:
                continue
            pts, exact, margin, winner = _score_pick(
                selected, f"{actual['home_score']}-{actual['away_score']}", scoring_config
            )
            total_points += pts
            exact_hits   += exact
            margin_hits  += margin
            winner_hits  += winner
            max_points   += exact_pts
            n            += 1
            ao    = outcome_1x2(actual["home_score"], actual["away_score"])
            probs = {"1": float(pred.p_team_a_win or 0), "X": float(pred.p_draw or 0), "2": float(pred.p_team_b_win or 0)}
            brier_sum   += sum((probs[k] - (1.0 if k == ao else 0.0)) ** 2 for k in ("1", "X", "2"))
            logloss_sum += -math.log(max(EPSILON, probs.get(ao, 0.0)))

    if n == 0:
        return {"points_efficiency": 0.0, "total_points": 0.0, "exact_score_accuracy": 0.0,
                "winner_accuracy": 0.0, "margin_accuracy": 0.0, "brier": 1.0, "log_loss": 10.0, "n": 0}
    return {
        "total_points":         total_points,
        "max_possible":         max_points,
        "points_efficiency":    total_points / max_points,
        "exact_score_accuracy": exact_hits / n,
        "winner_accuracy":      winner_hits / n,
        "margin_accuracy":      margin_hits / n,
        "brier":                brier_sum / n,
        "log_loss":             logloss_sum / n,
        "n":                    n,
    }


def _score_pick(candidate: str, actual: str, scoring: dict[str, Any]) -> tuple[float, int, int, int]:
    ca, cb = parse_score(candidate)
    aa, ab = parse_score(actual)
    ep = float(scoring.get("exact_score", 5))
    mp = float(scoring.get("same_margin_or_draw", scoring.get("margin_or_draw", 3)))
    wp = float(scoring.get("winner", 1))
    exact  = int(ca == aa and cb == ab)
    co, ao = outcome_1x2(ca, cb), outcome_1x2(aa, ab)
    margin = int((co == "X" and ao == "X") or ((ca - cb) == (aa - ab)))
    winner = int(co == ao)
    if exact:  return ep, exact, margin, winner
    if margin: return mp, exact, margin, winner
    if winner: return wp, exact, margin, winner
    return 0.0, exact, margin, winner


def _to_prediction_match(m: dict[str, Any]) -> PredictionMatch:
    return PredictionMatch(
        match_id=m["match_id"], source_match_id=m["match_id"],
        match_number=m.get("match_number"), stage=m.get("stage"), group_name=None,
        team_a_key=m["team_a_key"], team_b_key=m["team_b_key"],
        team_a_name=m["team_a_name"], team_b_name=m["team_b_name"],
        kickoff_utc=f"{m['match_date']}T00:00:00Z",
        stadium_country=m.get("country"), status="historical_backtest",
    )


def _print_report(
    model_id: str, years: list[int], n_total: int,
    results: list[dict[str, Any]], baseline: dict[str, float],
    baseline_params: dict[str, Any], top_n: int,
) -> None:
    sep = "=" * 95
    print(f"\n{sep}")
    print(f"TUNING — {model_id} — {n_total} partidos WC {years}")
    print(sep)
    print(f"{'rk':>3} {'eff%':>6} {'pts':>5} {'exact%':>7} {'win%':>6} {'brier':>6}  parametros")
    print("-" * 95)
    for rank, r in enumerate(results[:top_n], 1):
        m, p = r["metrics"], r["params"]
        print(f"{rank:>3} {m['points_efficiency']*100:>6.2f} {m['total_points']:>5.0f} "
              f"{m['exact_score_accuracy']*100:>7.2f} {m['winner_accuracy']*100:>6.2f} "
              f"{m['brier']:>6.3f}  " + "  ".join(f"{k}={v}" for k, v in p.items()))
    print()
    bp = "  ".join(f"{k}={v}" for k, v in baseline_params.items() if v is not None)
    print(f"Baseline:  eff={baseline['points_efficiency']*100:.2f}%  exact={baseline['exact_score_accuracy']*100:.2f}%  brier={baseline['brier']:.3f}")
    print(f"  params: {bp}")
    best = results[0]
    bm   = best["metrics"]
    print(f"\nMejor:     eff={bm['points_efficiency']*100:.2f}%  exact={bm['exact_score_accuracy']*100:.2f}%  brier={bm['brier']:.3f}")
    print(f"  Mejora eff: {(bm['points_efficiency']-baseline['points_efficiency'])*100:+.2f} pp  "
          f"|  Mejora brier: {bm['brier']-baseline['brier']:+.3f}")
    print(f"\nMejores parametros:")
    for k, v in best["params"].items():
        marker = " <-- cambio" if v != baseline_params.get(k) else ""
        print(f"  {k}: {v}{marker}")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _latest_history_run_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT history_run_id FROM v_latest_history_run").fetchone()
    if row is None:
        raise RuntimeError("No hay historico vigente. Ejecuta scripts/build_history.py primero.")
    return str(row["history_run_id"])


def _load_wc_matches(conn: sqlite3.Connection, history_run_id: str, years: list[int]) -> list[dict[str, Any]]:
    start, end = f"{min(years)}-01-01", f"{max(years)+1}-01-01"
    rows = conn.execute(
        "SELECT * FROM canonical_historical_matches "
        "WHERE history_run_id=? AND is_world_cup=1 AND match_date>=? AND match_date<? "
        "ORDER BY match_date, historical_match_id",
        (history_run_id, start, end),
    ).fetchall()
    year_set = set(years)
    counters: dict[int, int] = defaultdict(int)
    matches = []
    for row in rows:
        year = int(str(row["match_date"])[:4])
        if year not in year_set:
            continue
        counters[year] += 1
        n = counters[year]
        matches.append({
            "match_id": f"wc{year}_{n:02d}", "year": year, "match_number": n,
            "match_date": str(row["match_date"]), "stage": _infer_stage(n),
            "team_a_key": row["team_a_canonical_id"] or normalize_team_name(row["team_a_name"]),
            "team_b_key": row["team_b_canonical_id"] or normalize_team_name(row["team_b_name"]),
            "team_a_name": str(row["team_a_name"]), "team_b_name": str(row["team_b_name"]),
            "home_score": int(row["home_score"]), "away_score": int(row["away_score"]),
            "country": row["country"],
        })
    return matches


def _load_training_for_date(conn: sqlite3.Connection, history_run_id: str, cutoff_date: str) -> list[TrainingMatch]:
    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    rows   = conn.execute(
        "SELECT * FROM canonical_historical_matches WHERE history_run_id=? AND match_date<? "
        "ORDER BY match_date, historical_match_id",
        (history_run_id, cutoff_date),
    ).fetchall()
    matches = []
    for row in rows:
        md      = datetime.strptime(str(row["match_date"]), "%Y-%m-%d").date()
        recency = round(max(0.05, math.exp(-(cutoff - md).days / 365.25 / 8.0)), 6)
        matches.append(TrainingMatch(
            historical_match_id=str(row["historical_match_id"]),
            match_date=str(row["match_date"]),
            team_a_key=row["team_a_canonical_id"] or normalize_team_name(row["team_a_name"]),
            team_b_key=row["team_b_canonical_id"] or normalize_team_name(row["team_b_name"]),
            team_a_name=str(row["team_a_name"]), team_b_name=str(row["team_b_name"]),
            home_score=int(row["home_score"]), away_score=int(row["away_score"]),
            neutral=row["neutral"], importance_weight=float(row["importance_weight"] or 1.0),
            recency_weight=recency,
        ))
    return matches


def _group_by_date(matches: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in matches:
        grouped[str(m["match_date"])].append(m)
    return dict(sorted(grouped.items()))


def _infer_stage(n: int) -> str:
    if n <= 48: return "group"
    if n <= 56: return "r16"
    if n <= 60: return "qf"
    if n <= 62: return "sf"
    if n == 63: return "third_place"
    return "final"


if __name__ == "__main__":
    main()
