#!/usr/bin/env python
"""
GPU-accelerated hyperparameter tuning using PyTorch.
Corre TODOS los trials en paralelo en la GPU simultaneamente.

Velocidad vs CPU: 50-200x dependiendo de la GPU.
Soporta: elo_poisson, elo_dixon_coles, draw_specialist

Instalacion PyTorch (una sola vez en Anaconda Prompt):
  # Primero verifica tu version de CUDA: nvidia-smi
  # Para CUDA 12.x:
  conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
  # Para CUDA 11.8:
  conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

Uso:
  python scripts/tune_models_gpu.py --model elo_poisson
  python scripts/tune_models_gpu.py --model elo_dixon_coles --years 2014 2018 2022
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quiniela.models.common import (
    TrainingMatch,
    load_json_config,
    normalize_team_name,
)
from quiniela.scoring.quiniela import resolve_scoring_profile
from quiniela.storage.sqlite_store import SQLiteStore

PROJECT_ROOT = Path(__file__).parent.parent

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
}

GPU_UNSUPPORTED = {"bradley_terry_davidson", "attack_defense_poisson"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=list(SEARCH_SPACES.keys()) + list(GPU_UNSUPPORTED))
    parser.add_argument("--years",   nargs="+", type=int, default=[2014, 2018, 2022])
    parser.add_argument("--db",      type=Path, default=PROJECT_ROOT / "data" / "quiniela.db")
    parser.add_argument("--top",     type=int, default=20)
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--output",  type=Path, default=None)
    parser.add_argument("--scoring-profile", default=None, help="Perfil de scoring (ej: 3-1-0).")
    args = parser.parse_args()

    if args.model in GPU_UNSUPPORTED:
        print(f"ERROR: {args.model} no usa Elo — usa scripts/tune_models.py para este modelo.")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("AVISO: CUDA no disponible, usando CPU. Instala PyTorch con CUDA para aceleracion GPU.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    models_config  = load_json_config(PROJECT_ROOT / "configs" / "models.yaml")
    scoring_config = resolve_scoring_profile(load_json_config(PROJECT_ROOT / "configs" / "scoring.yaml"), args.scoring_profile)
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
        wc_date_groups = _group_by_date(wc_matches)
        print(f"Precargando datos para {len(wc_date_groups)} fechas...", flush=True)
        training_by_date = {
            date: _load_training_for_date(conn, history_run_id, date)
            for date in sorted(wc_date_groups.keys())
        }
    finally:
        store.close()

    if not wc_matches:
        print(f"ERROR: No hay partidos de Mundial para {args.years}")
        return

    space  = SEARCH_SPACES[args.model]
    keys   = list(space.keys())
    trials = [dict(zip(keys, c)) for c in itertools.product(*space.values())]

    print(f"Modelo: {args.model}  |  Anos: {args.years}  |  Partidos WC: {len(wc_matches)}", flush=True)
    print(f"Grid: {len(trials)} trials  |  Device: {device}", flush=True)

    t0      = time.time()
    results = _gpu_tune(trials, training_by_date, wc_date_groups,
                        scoring_config, base_model_config, args.model, device)
    elapsed = time.time() - t0

    results.sort(key=lambda r: r["metrics"]["points_efficiency"], reverse=True)
    print(f"\nTiempo total: {elapsed:.1f}s  ({elapsed / len(trials) * 1000:.1f} ms/trial)", flush=True)

    baseline_params = {k: base_model_config.get(k) for k in keys}
    _print_report(args.model, args.years, len(wc_matches), results, baseline_params, args.top)

    output_path = args.output or (
        PROJECT_ROOT / "data" / "backtests" /
        f"tuning_{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id":        args.model,
        "years":           args.years,
        "n_matches":       len(wc_matches),
        "n_trials":        len(trials),
        "elapsed_seconds": round(elapsed, 1),
        "device":          str(device),
        "baseline_params": baseline_params,
        "best_params":     results[0]["params"],
        "best_metrics":    results[0]["metrics"],
        "top_results":     results[:50],
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"Resultados guardados: {output_path}")


# ---------------------------------------------------------------------------
# GPU core
# ---------------------------------------------------------------------------

def _gpu_tune(
    trials: list[dict[str, Any]],
    training_by_date: dict[str, list[TrainingMatch]],
    wc_date_groups: dict[str, list[dict[str, Any]]],
    scoring_config: dict[str, Any],
    base_model_config: dict[str, Any],
    model_id: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    N          = len(trials)
    max_goals  = int(base_model_config.get("max_goals", 8))
    G          = max_goals + 1
    init_r     = float(base_model_config.get("initial_rating", 1500.0))
    exact_pts  = float(scoring_config.get("exact_score", 5))

    # Team vocabulary
    all_teams: set[str] = set()
    for ms in training_by_date.values():
        for m in ms:
            all_teams.update([m.team_a_key, m.team_b_key])
    for ms in wc_date_groups.values():
        for m in ms:
            all_teams.update([m["team_a_key"], m["team_b_key"]])
    team_vocab = {t: i for i, t in enumerate(sorted(all_teams))}
    T = len(team_vocab)

    # Trial parameter tensors [N]
    k_factors       = torch.tensor([t.get("k_factor", 22.0)                  for t in trials], dtype=torch.float32, device=device)
    home_advs       = torch.tensor([t.get("home_advantage", 55.0)             for t in trials], dtype=torch.float32, device=device)
    goal_scales     = torch.tensor([t.get("goal_scale", 0.55)                 for t in trials], dtype=torch.float32, device=device)
    min_importances = torch.tensor([t.get("min_importance_for_rating", 0.0)   for t in trials], dtype=torch.float32, device=device)

    rhos        = torch.tensor([t.get("dixon_coles_rho", -0.1)  for t in trials], dtype=torch.float32, device=device) if model_id == "elo_dixon_coles"  else None
    draw_boosts = torch.tensor([t.get("max_draw_boost",   0.22) for t in trials], dtype=torch.float32, device=device) if model_id == "draw_specialist"   else None

    # Precomputed quiniela points matrix [G, G, G, G] and flat [G^2, G^2]
    points_matrix = _build_points_matrix(max_goals, scoring_config, device)   # [G, G, G, G]
    pts_flat      = points_matrix.reshape(G * G, G * G)                       # [G^2(ci,cj), G^2(ai,aj)]

    # Outcome masks for Brier score
    gi, gj = torch.meshgrid(torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij")
    home_mask = (gi > gj).float()   # [G, G]
    draw_mask = (gi == gj).float()
    away_mask = (gi < gj).float()

    # Accumulators [N]
    total_pts  = torch.zeros(N, device=device)
    n_exact    = torch.zeros(N, device=device)
    n_winner   = torch.zeros(N, device=device)
    brier_sum  = torch.zeros(N, device=device)
    total_n    = 0

    sorted_dates = sorted(wc_date_groups.keys())

    for d_idx, date in enumerate(sorted_dates):
        training  = training_by_date[date]
        day_wc    = wc_date_groups[date]
        base_lam  = _base_goals(training)

        # Fit Elo for this date (all N trials simultaneously)
        ratings = torch.full((N, T), init_r, device=device)

        if training:
            imp_w   = [m.importance_weight for m in training]
            rec_w   = [m.recency_weight     for m in training]
            comb_w  = [max(0.05, i * r) for i, r in zip(imp_w, rec_w)]
            act_a   = [1.0 if m.home_score > m.away_score else (0.5 if m.home_score == m.away_score else 0.0) for m in training]
            gd_s    = [math.log1p(abs(m.home_score - m.away_score)) if m.home_score != m.away_score else 1.0 for m in training]
            ha_f    = [0.0 if m.neutral == 1 else 1.0 for m in training]
            ia_list = [team_vocab.get(m.team_a_key, -1) for m in training]
            ib_list = [team_vocab.get(m.team_b_key, -1) for m in training]

            # include_mask[n, m] = True if trial n includes training match m
            imp_tensor   = torch.tensor(imp_w, dtype=torch.float32, device=device)   # [M]
            include_mask = (min_importances < 1e-9).unsqueeze(1) | (imp_tensor.unsqueeze(0) >= min_importances.unsqueeze(1))  # [N, M]

            for m_idx in range(len(training)):
                ia = ia_list[m_idx]
                ib = ib_list[m_idx]
                if ia < 0 or ib < 0:
                    continue
                ra = ratings[:, ia].clone()   # [N]
                rb = ratings[:, ib].clone()   # [N]
                eff_ha   = home_advs * ha_f[m_idx]                                         # [N]
                exp_a    = 1.0 / (1.0 + torch.pow(10.0, -((ra + eff_ha) - rb) / 400.0))   # [N]
                delta    = k_factors * comb_w[m_idx] * gd_s[m_idx] * (act_a[m_idx] - exp_a)  # [N]
                delta    = delta * include_mask[:, m_idx].float()
                ratings[:, ia] = ra + delta
                ratings[:, ib] = rb - delta

        print(f"  [{d_idx+1:>2}/{len(sorted_dates)}] {date}  train={len(training)}  wc={len(day_wc)}", flush=True)

        # Predict WC matches
        for wc_m in day_wc:
            ia = team_vocab.get(wc_m["team_a_key"], -1)
            ib = team_vocab.get(wc_m["team_b_key"], -1)
            if ia < 0 or ib < 0:
                continue

            ra = ratings[:, ia]  # [N]
            rb = ratings[:, ib]  # [N]

            # Neutral venue: no home advantage for prediction
            rd       = ra - rb                                                            # [N]
            lambda_a = (base_lam * torch.exp( goal_scales * rd / 400.0)).clamp(0.2, 4.5) # [N]
            lambda_b = (base_lam * torch.exp(-goal_scales * rd / 400.0)).clamp(0.2, 4.5) # [N]

            sm = _score_matrix_batch(lambda_a, lambda_b, G, device)                      # [N, G, G]

            if model_id == "elo_dixon_coles":
                sm = _dixon_coles_correction(sm, lambda_a, lambda_b, rhos)
            elif model_id == "draw_specialist":
                sm = _draw_boost(sm, draw_boosts)

            # Best pick: maximise expected quiniela points
            sm_flat      = sm.reshape(N, G * G)                  # [N, G^2]
            exp_pts_flat = sm_flat @ pts_flat.T                   # [N, G^2]
            best_idx     = exp_pts_flat.argmax(dim=1)             # [N]
            best_i       = best_idx // G                          # [N]
            best_j       = best_idx % G                           # [N]

            ah = int(wc_m["home_score"])
            ab = int(wc_m["away_score"])

            # Points earned
            actual_pts = points_matrix[best_i, best_j, ah, ab]   # [N]
            total_pts += actual_pts
            n_exact   += (actual_pts >= exact_pts - 1e-6).float()

            # Winner accuracy
            if ah > ab:
                n_winner += (best_i > best_j).float()
            elif ah == ab:
                n_winner += (best_i == best_j).float()
            else:
                n_winner += (best_i < best_j).float()

            # Brier score
            p_home = (sm * home_mask).sum(dim=(1, 2))
            p_draw = (sm * draw_mask).sum(dim=(1, 2))
            p_away = (sm * away_mask).sum(dim=(1, 2))
            if ah > ab:
                brier_sum += (p_home - 1.0) ** 2 + p_draw ** 2 + p_away ** 2
            elif ah == ab:
                brier_sum += p_home ** 2 + (p_draw - 1.0) ** 2 + p_away ** 2
            else:
                brier_sum += p_home ** 2 + p_draw ** 2 + (p_away - 1.0) ** 2

            total_n += 1

    max_pts = exact_pts * total_n
    eff     = (total_pts / max(max_pts, 1.0)).cpu().tolist()
    tpts    = total_pts.cpu().tolist()
    exc     = (n_exact   / max(total_n, 1)).cpu().tolist()
    win     = (n_winner  / max(total_n, 1)).cpu().tolist()
    bri     = (brier_sum / max(total_n, 1)).cpu().tolist()

    return [
        {
            "params": trials[n],
            "metrics": {
                "points_efficiency":    eff[n],
                "total_points":         tpts[n],
                "exact_score_accuracy": exc[n],
                "winner_accuracy":      win[n],
                "brier":                bri[n],
                "n":                    total_n,
            },
        }
        for n in range(N)
    ]


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _build_points_matrix(max_goals: int, scoring: dict[str, Any], device: torch.device) -> torch.Tensor:
    G     = max_goals + 1
    exact = float(scoring.get("exact_score", 5))
    margin = float(scoring.get("same_margin_or_draw", scoring.get("margin_or_draw", 3)))
    winner = float(scoring.get("winner", 1))
    pts = torch.zeros(G, G, G, G, dtype=torch.float32, device=device)
    for ci in range(G):
        for cj in range(G):
            for ai in range(G):
                for aj in range(G):
                    if ci == ai and cj == aj:
                        pts[ci, cj, ai, aj] = exact
                    elif (ci - cj) == (ai - aj):
                        pts[ci, cj, ai, aj] = margin
                    elif (ci > cj) == (ai > aj) and not (ci == cj or ai == aj):
                        pts[ci, cj, ai, aj] = winner
                    elif ci == cj and ai == aj:
                        pass  # exact already handled; this can't happen here
    return pts


def _score_matrix_batch(lambda_a: torch.Tensor, lambda_b: torch.Tensor, G: int, device: torch.device) -> torch.Tensor:
    k       = torch.arange(G, dtype=torch.float32, device=device)  # [G]
    lgamma  = torch.lgamma(k + 1)                                   # [G]
    log_pa  = k * torch.log(lambda_a.unsqueeze(1).clamp(1e-10)) - lambda_a.unsqueeze(1) - lgamma  # [N, G]
    log_pb  = k * torch.log(lambda_b.unsqueeze(1).clamp(1e-10)) - lambda_b.unsqueeze(1) - lgamma  # [N, G]
    pa      = torch.exp(log_pa)   # [N, G]
    pb      = torch.exp(log_pb)   # [N, G]
    return pa.unsqueeze(2) * pb.unsqueeze(1)                        # [N, G, G]


def _dixon_coles_correction(sm: torch.Tensor, la: torch.Tensor, lb: torch.Tensor, rhos: torch.Tensor) -> torch.Tensor:
    sm = sm.clone()
    sm[:, 0, 0] = sm[:, 0, 0] * (1.0 - la * lb * rhos)
    sm[:, 1, 0] = sm[:, 1, 0] * (1.0 + lb * rhos)
    sm[:, 0, 1] = sm[:, 0, 1] * (1.0 + la * rhos)
    sm[:, 1, 1] = sm[:, 1, 1] * (1.0 - rhos)
    return sm / sm.sum(dim=(1, 2), keepdim=True).clamp(min=1e-10)


def _draw_boost(sm: torch.Tensor, boosts: torch.Tensor) -> torch.Tensor:
    sm = sm.clone()
    G  = sm.shape[1]
    for i in range(G):
        sm[:, i, i] = sm[:, i, i] * (1.0 + boosts)
    return sm / sm.sum(dim=(1, 2), keepdim=True).clamp(min=1e-10)


def _base_goals(training: list[TrainingMatch]) -> float:
    wg = wm = 0.0
    for m in training:
        w   = m.importance_weight * m.recency_weight
        wg += (m.home_score + m.away_score) * w
        wm += 2.0 * w
    return wg / wm if wm > 0 else 1.25


# ---------------------------------------------------------------------------
# DB helpers (identical to tune_models.py)
# ---------------------------------------------------------------------------

def _latest_history_run_id(conn) -> str:
    row = conn.execute("SELECT history_run_id FROM v_latest_history_run").fetchone()
    if row is None:
        raise RuntimeError("No hay historico vigente. Ejecuta scripts/build_history.py primero.")
    return str(row["history_run_id"])


def _load_wc_matches(conn, history_run_id: str, years: list[int]) -> list[dict[str, Any]]:
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
            "match_id": f"wc{year}_{n:02d}", "year": year,
            "match_date": str(row["match_date"]),
            "team_a_key": row["team_a_canonical_id"] or normalize_team_name(row["team_a_name"]),
            "team_b_key": row["team_b_canonical_id"] or normalize_team_name(row["team_b_name"]),
            "team_a_name": str(row["team_a_name"]), "team_b_name": str(row["team_b_name"]),
            "home_score": int(row["home_score"]), "away_score": int(row["away_score"]),
            "country": row["country"],
        })
    return matches


def _load_training_for_date(conn, history_run_id: str, cutoff_date: str) -> list[TrainingMatch]:
    from datetime import date as date_type
    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    rows   = conn.execute(
        "SELECT * FROM canonical_historical_matches WHERE history_run_id=? AND match_date<? "
        "ORDER BY match_date, historical_match_id",
        (history_run_id, cutoff_date),
    ).fetchall()
    out = []
    for row in rows:
        md      = datetime.strptime(str(row["match_date"]), "%Y-%m-%d").date()
        recency = round(max(0.05, math.exp(-(cutoff - md).days / 365.25 / 8.0)), 6)
        out.append(TrainingMatch(
            historical_match_id=str(row["historical_match_id"]),
            match_date=str(row["match_date"]),
            team_a_key=row["team_a_canonical_id"] or normalize_team_name(row["team_a_name"]),
            team_b_key=row["team_b_canonical_id"] or normalize_team_name(row["team_b_name"]),
            team_a_name=str(row["team_a_name"]), team_b_name=str(row["team_b_name"]),
            home_score=int(row["home_score"]), away_score=int(row["away_score"]),
            neutral=row["neutral"], importance_weight=float(row["importance_weight"] or 1.0),
            recency_weight=recency,
        ))
    return out


def _group_by_date(matches: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in matches:
        grouped[str(m["match_date"])].append(m)
    return dict(sorted(grouped.items()))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(model_id, years, n_total, results, baseline_params, top_n):
    sep = "=" * 95
    print(f"\n{sep}")
    print(f"TUNING GPU — {model_id} — {n_total} partidos WC {years}")
    print(sep)
    print(f"{'rk':>3} {'eff%':>6} {'pts':>5} {'exact%':>7} {'win%':>6} {'brier':>6}  parametros")
    print("-" * 95)
    for rank, r in enumerate(results[:top_n], 1):
        m, p = r["metrics"], r["params"]
        print(f"{rank:>3} {m['points_efficiency']*100:>6.2f} {m['total_points']:>5.0f} "
              f"{m['exact_score_accuracy']*100:>7.2f} {m['winner_accuracy']*100:>6.2f} "
              f"{m['brier']:>6.3f}  " + "  ".join(f"{k}={v}" for k, v in p.items()))
    print()
    bp   = "  ".join(f"{k}={v}" for k, v in baseline_params.items() if v is not None)
    best = results[0]
    bm   = best["metrics"]
    print(f"Mejor:  eff={bm['points_efficiency']*100:.2f}%  exact={bm['exact_score_accuracy']*100:.2f}%  brier={bm['brier']:.3f}")
    print(f"  Mejores parametros:")
    for k, v in best["params"].items():
        marker = " <-- cambio" if v != baseline_params.get(k) else ""
        print(f"    {k}: {v}{marker}")


if __name__ == "__main__":
    main()
