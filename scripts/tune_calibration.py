#!/usr/bin/env python
"""
tune_calibration.py — Optimización conjunta de pesos de blending + parámetros de
calibración histórica para calibrated_scoreline_ensemble usando Optuna (TPE).

Qué optimiza de forma CONJUNTA:
  Fase 1 — Pesos de blending de cada modelo base (log-espacio → softmax)
            confidence_power (ajuste de peso por confianza del modelo)
  Fase 2 — prior_power    : cuánto pesa el prior histórico vs el ensemble
            model_power   : exponente del ensemble en el blend geométrico
            low_score_penalty : descuento a marcadores conservadores (1-0, 1-1, 0-1)
            high_goal_bonus   : bonus a partidos de muchos goles
            high_goal_min_total: mínimo de goles para aplicar el bonus
            smoothing     : suavizado Laplace del prior histórico

Datos de entrada: v_latest_backtest_predictions (score_matrix_json ya calculadas)
Prior histórico : canonical_historical_matches donde is_world_cup=1 y year >= min_year
Objetivo        : maximizar points_efficiency sobre el backtest vigente

Instalación de Optuna (solo la primera vez, en Anaconda Prompt):
  pip install optuna

Uso básico:
  python scripts/tune_calibration.py --n-trials 50000

Run resumible (puede interrumpirse y continuar):
  python scripts/tune_calibration.py --n-trials 50000 \\
      --study-storage sqlite:///data/backtests/calib_study.db

Solo ver resultado sin modificar models.yaml:
  python scripts/tune_calibration.py --n-trials 50000 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Any

# Evitar UnicodeEncodeError en consola Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.models.common import (
    load_json_config,
    normalize_score_matrix,
    outcome_1x2,
    parse_score,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

TARGET_MODEL_ID = "calibrated_scoreline_ensemble"

ENSEMBLE_IDS = frozenset({
    "weighted_ensemble", "weighted_points_ensemble", "weighted_1x2_ensemble",
    "weighted_exact_ensemble", "calibrated_scoreline_ensemble",
})

# Espacio de búsqueda para parámetros de calibración
CALIB_BOUNDS: dict[str, tuple[float, float]] = {
    "prior_power":         (0.10, 0.80),
    "model_power":         (0.40, 1.20),
    "low_score_penalty":   (0.00, 0.50),
    "high_goal_bonus":     (0.00, 0.15),
    "smoothing":           (0.05, 1.00),
    "confidence_power":    (0.00, 0.50),
}
HIGH_MIN_BOUNDS = (2, 4)  # high_goal_min_total (int)

# Marcadores que reciben penalización por ser conservadores
CENTRAL_LOW_SCORES = frozenset({"1-0", "1-1", "0-1"})


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimización conjunta pesos + calibración para calibrated_scoreline_ensemble."
    )
    parser.add_argument("--db",           default=str(PROJECT_ROOT / "data" / "quiniela.db"))
    parser.add_argument("--models-config",default=str(PROJECT_ROOT / "configs" / "models.yaml"))
    parser.add_argument("--scoring-config",default=str(PROJECT_ROOT / "configs" / "scoring.yaml"))
    parser.add_argument("--n-trials",     type=int, default=20_000,
                        help="Número de trials Optuna. Recomendado: 50000 para producción.")
    parser.add_argument("--seed",         type=int, default=20260608)
    parser.add_argument("--output",       default=str(
        PROJECT_ROOT / "data" / "backtests" / "calibration_tuning_latest.json"))
    parser.add_argument("--study-storage", default=None,
                        help="URI SQLite para run resumible. Ej: sqlite:///data/backtests/calib_study.db")
    parser.add_argument("--study-name",   default="tune_calibration_v1")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Muestra resultados sin modificar models.yaml")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if not HAS_OPTUNA:
        print("ERROR: Optuna no está instalado.")
        print("  Instálalo con:  pip install optuna")
        return 1

    models_config  = load_json_config(Path(args.models_config))
    scoring_config = load_json_config(Path(args.scoring_config))

    target_config = next(
        (m for m in models_config.get("models", []) if m.get("model_id") == TARGET_MODEL_ID),
        None,
    )
    if target_config is None:
        print(f"ERROR: {TARGET_MODEL_ID} no encontrado en models.yaml")
        return 1

    excluded  = ENSEMBLE_IDS | set(target_config.get("exclude_models", []))
    max_goals = int(target_config.get("max_goals", 8))
    calib_cfg = dict(target_config.get("scoreline_calibration", {}))
    min_year  = int(calib_cfg.get("min_year", 1974))

    # ── Carga de datos ──────────────────────────────────────────────────────
    print("Cargando matrices del backtest...", flush=True)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        bt_data = _load_backtest_data(conn, excluded, max_goals)
        if bt_data is None:
            print("ERROR: Sin datos de backtest. Ejecuta run_backtest.py primero.")
            return 1
        matrices, confidences, actual_idx, match_ids, base_model_ids, bt_run_id = bt_data
        print(f"  backtest_run_id : {bt_run_id}")
        print(f"  partidos        : {len(match_ids)}")
        print(f"  modelos base    : {len(base_model_ids)} → {base_model_ids}")

        print("Calculando prior histórico WC...", flush=True)
        prior_raw, prior_n = _compute_historical_prior(conn, max_goals, min_year)
        print(f"  prior WC 1974+  : {prior_n} partidos")
    finally:
        conn.close()

    # ── Arrays estáticos ────────────────────────────────────────────────────
    scores = [f"{a}-{b}" for a in range(max_goals + 1) for b in range(max_goals + 1)]
    points_matrix, outcome_idx, central_low_mask, total_goals_arr = _build_static_arrays(
        scores, scoring_config
    )

    n_matches    = len(match_ids)
    max_possible = n_matches * float(scoring_config.get("exact_score", 5))

    existing_weights = dict(
        target_config.get("optimized_weights") or target_config.get("fallback_weights") or {}
    )
    n_dims = len(base_model_ids) + len(CALIB_BOUNDS) + 1  # +1 for high_goal_min_total

    # ── Función objetivo ────────────────────────────────────────────────────
    def objective(trial: "optuna.Trial") -> float:
        # Pesos de blending (log-espacio → softmax)
        log_ws  = np.array([trial.suggest_float(f"log_w_{m}", -4.0, 0.0) for m in base_model_ids])
        weights = np.exp(log_ws); weights /= weights.sum()

        # Parámetros de calibración
        conf_power  = trial.suggest_float("confidence_power",   *CALIB_BOUNDS["confidence_power"])
        prior_power = trial.suggest_float("prior_power",        *CALIB_BOUNDS["prior_power"])
        model_power = trial.suggest_float("model_power",        *CALIB_BOUNDS["model_power"])
        low_pen     = trial.suggest_float("low_score_penalty",  *CALIB_BOUNDS["low_score_penalty"])
        high_bonus  = trial.suggest_float("high_goal_bonus",    *CALIB_BOUNDS["high_goal_bonus"])
        high_min    = trial.suggest_int("high_goal_min_total",  *HIGH_MIN_BOUNDS)
        smoothing   = trial.suggest_float("smoothing",          *CALIB_BOUNDS["smoothing"])

        pts = _evaluate(
            weights=weights, conf_power=conf_power,
            prior_power=prior_power, model_power=model_power,
            low_pen=low_pen, high_bonus=high_bonus, high_min=high_min,
            smoothing=smoothing,
            matrices=matrices, confidences=confidences, actual_idx=actual_idx,
            points_matrix=points_matrix, prior_raw=prior_raw,
            central_low_mask=central_low_mask, total_goals_arr=total_goals_arr,
            outcome_idx=outcome_idx,
        )
        return pts / max_possible

    # ── Estudio Optuna ───────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=min(200, args.n_trials // 10))
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=args.study_storage,
        study_name=args.study_name,
        load_if_exists=True,
    )
    _add_warmstart_trial(study, base_model_ids, target_config, existing_weights)

    print(f"\nOptimización Optuna TPE", flush=True)
    print(f"  trials          : {args.n_trials}")
    print(f"  dimensiones     : {n_dims}  ({len(base_model_ids)} pesos + 7 calibración)")
    print(f"  seed            : {args.seed}")
    if args.study_storage:
        print(f"  storage         : {args.study_storage}  (resumible)")
    print()

    cb = _ProgressCallback(args.n_trials, max_possible)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=args.n_trials, callbacks=[cb], show_progress_bar=False)

    # ── Resultados ───────────────────────────────────────────────────────────
    best = study.best_trial
    bp   = best.params

    best_weights = np.exp(np.array([bp[f"log_w_{m}"] for m in base_model_ids]))
    best_weights /= best_weights.sum()
    best_weight_dict = {m: round(float(w), 6) for m, w in zip(base_model_ids, best_weights)}

    best_pts = best.value * max_possible
    extra    = _final_metrics(
        weights=best_weights,
        conf_power=float(bp["confidence_power"]),
        prior_power=float(bp["prior_power"]),
        model_power=float(bp["model_power"]),
        low_pen=float(bp["low_score_penalty"]),
        high_bonus=float(bp["high_goal_bonus"]),
        high_min=int(bp["high_goal_min_total"]),
        smoothing=float(bp["smoothing"]),
        matrices=matrices, confidences=confidences, actual_idx=actual_idx,
        points_matrix=points_matrix, prior_raw=prior_raw,
        central_low_mask=central_low_mask, total_goals_arr=total_goals_arr,
        outcome_idx=outcome_idx, scoring_config=scoring_config,
    )

    print(f"\n{'='*65}")
    print(f"RESULTADO FINAL — {TARGET_MODEL_ID}")
    print(f"{'='*65}")
    print(f"  Eficiencia : {best.value*100:.2f}%   "
          f"({best_pts:.0f} / {max_possible:.0f} pts)")
    print(f"  Exactos    : {extra['exact']}   "
          f"Márgenes: {extra['margin']}   Ganadores: {extra['winner']}")

    print("\nPesos de blending:")
    for m, w in sorted(best_weight_dict.items(), key=lambda x: -x[1]):
        mark = " ←" if w > 0.10 else ""
        print(f"    {m:43s}  {w:.4f}{mark}")

    print("\nParámetros de calibración:")
    calib_keys = [
        ("confidence_power",   target_config.get("confidence_power", "—")),
        ("prior_power",        calib_cfg.get("prior_power", "—")),
        ("model_power",        calib_cfg.get("model_power", "—")),
        ("low_score_penalty",  calib_cfg.get("low_score_penalty", "—")),
        ("high_goal_bonus",    calib_cfg.get("high_goal_bonus", "—")),
        ("high_goal_min_total",calib_cfg.get("high_goal_min_total", "—")),
        ("smoothing",          calib_cfg.get("smoothing", "—")),
    ]
    for key, old in calib_keys:
        new = bp.get(key, "—")
        new_str = f"{new:.4f}" if isinstance(new, float) else str(new)
        print(f"    {key:30s}  {str(old):>8}  →  {new_str}")

    # ── Guardar JSON ─────────────────────────────────────────────────────────
    output_payload = {
        "backtest_run_id"   : bt_run_id,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(study.trials),
        "seed"              : args.seed,
        "base_models"       : base_model_ids,
        "n_matches"         : n_matches,
        "best_efficiency"   : round(best.value, 8),
        "best_points"       : round(best_pts, 2),
        "max_possible_points": max_possible,
        "exact_hits"        : extra["exact"],
        "margin_or_draw_hits": extra["margin"],
        "winner_hits"       : extra["winner"],
        "best_weights"      : best_weight_dict,
        "best_calib_params" : {k: v for k, v in bp.items() if not k.startswith("log_w_")},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nResultados guardados : {out_path}")

    # ── Actualizar models.yaml ────────────────────────────────────────────────
    if not args.dry_run:
        _update_models_config(
            models_path=Path(args.models_config),
            models_config=models_config,
            best_weight_dict=best_weight_dict,
            bp=bp,
            bt_run_id=bt_run_id,
            best_pts=best_pts,
            max_possible=max_possible,
            n_trials=args.n_trials,
            seed=args.seed,
            extra=extra,
        )
        print(f"Config actualizada   : {args.models_config}")
    else:
        print("--dry-run activo: models.yaml no fue modificado")

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────────────────────

def _load_backtest_data(
    conn: sqlite3.Connection,
    excluded: frozenset[str],
    max_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], str] | None:
    """Carga matrices del backtest como arrays numpy.

    Returns:
        matrices      [M, N, S]  — score matrices por modelo, partido, marcador
        confidences   [M, N]     — max(p1, px, p2) por modelo y partido
        actual_idx    [N]        — índice del marcador real por partido
        match_ids     list[str]  — IDs de los N partidos
        base_model_ids list[str] — IDs de los M modelos base
        backtest_run_id str
    """
    run_row = conn.execute("SELECT backtest_run_id FROM v_latest_backtest_run").fetchone()
    if run_row is None:
        return None
    bt_run_id = str(run_row["backtest_run_id"])

    rows = conn.execute(
        """SELECT match_id, model_id, score_matrix_json, actual_score,
                  p_team_a_win, p_draw, p_team_b_win
           FROM v_latest_backtest_predictions
           WHERE status = 'ok' AND score_matrix_json IS NOT NULL"""
    ).fetchall()
    if not rows:
        return None

    scores    = [f"{a}-{b}" for a in range(max_goals + 1) for b in range(max_goals + 1)]
    score_idx = {s: i for i, s in enumerate(scores)}
    n_scores  = len(scores)

    # Solo modelos completos (predicción en TODOS los partidos)
    all_match_ids = sorted({str(r["match_id"]) for r in rows})
    n_all         = len(all_match_ids)
    model_counts: dict[str, int] = {}
    for r in rows:
        mid = str(r["model_id"])
        if mid not in excluded:
            model_counts[mid] = model_counts.get(mid, 0) + 1
    base_model_ids = sorted(m for m, cnt in model_counts.items() if cnt == n_all)
    if not base_model_ids:
        return None

    M = len(base_model_ids); N = n_all
    model_pos = {m: i for i, m in enumerate(base_model_ids)}
    match_pos  = {m: i for i, m in enumerate(all_match_ids)}

    matrices    = np.zeros((M, N, n_scores), dtype=np.float64)
    confidences = np.ones((M, N), dtype=np.float64)
    actual_by_match: dict[str, str] = {}

    for r in rows:
        mid = str(r["model_id"])
        if mid not in model_pos:
            continue
        mi = model_pos[mid]
        ni = match_pos[str(r["match_id"])]

        mat = normalize_score_matrix(json.loads(str(r["score_matrix_json"])))
        for s, p in mat["scores"].items():
            if s in score_idx:
                matrices[mi, ni, score_idx[s]] = float(p)
        row_sum = matrices[mi, ni].sum()
        if row_sum > 0:
            matrices[mi, ni] /= row_sum

        conf = max(
            float(r["p_team_a_win"] or 0.0),
            float(r["p_draw"]       or 0.0),
            float(r["p_team_b_win"] or 0.0),
        )
        confidences[mi, ni] = max(0.001, conf)
        actual_by_match[str(r["match_id"])] = str(r["actual_score"])

    actual_idx = np.array([
        score_idx["{}-{}".format(
            min(parse_score(actual_by_match.get(mid, "0-0"))[0], max_goals),
            min(parse_score(actual_by_match.get(mid, "0-0"))[1], max_goals),
        )]
        for mid in all_match_ids
    ], dtype=np.int64)

    return matrices, confidences, actual_idx, all_match_ids, base_model_ids, bt_run_id


def _compute_historical_prior(
    conn: sqlite3.Connection,
    max_goals: int,
    min_year: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Calcula conteos históricos de marcadores WC para el prior (array completo de 81)."""
    scores    = [f"{a}-{b}" for a in range(max_goals + 1) for b in range(max_goals + 1)]
    score_idx = {s: i for i, s in enumerate(scores)}
    counts    = {o: np.zeros(len(scores), dtype=np.float64) for o in ("1", "X", "2")}
    n = 0

    rows = conn.execute(
        "SELECT home_score, away_score, match_date FROM canonical_historical_matches "
        "WHERE home_score IS NOT NULL AND away_score IS NOT NULL AND is_world_cup = 1"
    ).fetchall()
    for r in rows:
        try:
            year = int(str(r["match_date"])[:4])
        except (ValueError, TypeError):
            continue
        if year < min_year:
            continue
        a = min(int(r["home_score"]), max_goals)
        b = min(int(r["away_score"]), max_goals)
        s = f"{a}-{b}"
        o = outcome_1x2(a, b)
        counts[o][score_idx[s]] += 1.0
        n += 1

    return counts, n


# ─────────────────────────────────────────────────────────────────────────────
# Arrays estáticos
# ─────────────────────────────────────────────────────────────────────────────

def _build_static_arrays(
    scores: list[str],
    scoring_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Construye matrices/máscaras que no cambian entre trials."""
    n         = len(scores)
    exact_pts = float(scoring_config.get("exact_score", 5))
    marg_pts  = float(scoring_config.get("same_margin_or_draw", scoring_config.get("margin_or_draw", 3)))
    win_pts   = float(scoring_config.get("winner", 1))

    parsed   = [parse_score(s) for s in scores]
    outcomes = [outcome_1x2(a, b) for a, b in parsed]

    # Points matrix [n_scores, n_scores]
    pts = np.zeros((n, n), dtype=np.float64)
    for pi, (pa, pb) in enumerate(parsed):
        for ai, (aa, ab) in enumerate(parsed):
            if pa == aa and pb == ab:
                pts[pi, ai] = exact_pts
            elif (pa - pb) == (aa - ab):
                pts[pi, ai] = marg_pts
            elif outcomes[pi] == outcomes[ai]:
                pts[pi, ai] = win_pts

    # Índices por outcome
    outcome_idx = {
        o: np.array([i for i, oo in enumerate(outcomes) if oo == o], dtype=np.int64)
        for o in ("1", "X", "2")
    }

    central_low_mask = np.array([s in CENTRAL_LOW_SCORES for s in scores], dtype=bool)
    total_goals_arr  = np.array([a + b for a, b in parsed], dtype=np.float64)

    return pts, outcome_idx, central_low_mask, total_goals_arr


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación vectorizada (núcleo del loop de optimización)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(
    weights: np.ndarray,          # [M]
    conf_power: float,
    prior_power: float,
    model_power: float,
    low_pen: float,
    high_bonus: float,
    high_min: int,
    smoothing: float,
    matrices: np.ndarray,         # [M, N, S]
    confidences: np.ndarray,      # [M, N]
    actual_idx: np.ndarray,       # [N]
    points_matrix: np.ndarray,    # [S, S]
    prior_raw: dict[str, np.ndarray],  # outcome → [S]
    central_low_mask: np.ndarray, # [S]
    total_goals_arr: np.ndarray,  # [S]
    outcome_idx: dict[str, np.ndarray],
) -> float:
    """Blend → calibrar → seleccionar pick → puntos totales.  Completamente vectorizado."""

    # 1. Pesos efectivos por confianza [M, N]
    if conf_power > 1e-6:
        eff_w = weights[:, None] * (confidences ** conf_power)
    else:
        eff_w = weights[:, None] * np.ones_like(confidences)

    # 2. Blend de matrices [N, S]
    num     = np.einsum("mn,mns->ns", eff_w, matrices)
    den     = eff_w.sum(axis=0)[:, None]                       # [N, 1]
    blended = num / np.clip(den, 1e-10, None)

    # 3. Shape factor [S] (igual para todos los partidos)
    sf = np.ones(len(central_low_mask), dtype=np.float64)
    if low_pen > 0:
        sf[central_low_mask] *= (1.0 - low_pen)
    if high_bonus > 0:
        hg_mask     = total_goals_arr >= high_min
        extra_goals = (total_goals_arr[hg_mask] - high_min + 1).clip(min=0)
        sf[hg_mask] *= (1.0 + high_bonus * extra_goals)

    # 4. Calibración por outcome [N, S]
    calibrated = np.zeros_like(blended)
    for o, idx in outcome_idx.items():
        raw_counts = prior_raw[o][idx]                         # [K_o]
        prior_cond = (raw_counts + smoothing)
        prior_cond /= prior_cond.sum()                         # [K_o]

        blended_o = blended[:, idx]                            # [N, K_o]
        mass_o    = blended_o.sum(axis=1, keepdims=True)       # [N, 1]
        model_cond = blended_o / np.clip(mass_o, 1e-10, None) # [N, K_o]

        raw_cal = (
            np.clip(model_cond, 1e-12, None) ** model_power
            * np.clip(prior_cond,  1e-12, None) ** prior_power
            * np.clip(sf[idx],     1e-12, None)
        )                                                       # [N, K_o]
        raw_sum = raw_cal.sum(axis=1, keepdims=True)
        calibrated[:, idx] = mass_o * raw_cal / np.clip(raw_sum, 1e-10, None)

    # 5. Seleccionar pick con máximo expected_points [N]
    exp_pts  = calibrated @ points_matrix.T                    # [N, S]
    best_idx = exp_pts.argmax(axis=1)                          # [N]

    # 6. Puntos obtenidos
    return float(points_matrix[best_idx, actual_idx].sum())


def _final_metrics(
    weights: np.ndarray, conf_power: float, prior_power: float, model_power: float,
    low_pen: float, high_bonus: float, high_min: int, smoothing: float,
    matrices: np.ndarray, confidences: np.ndarray, actual_idx: np.ndarray,
    points_matrix: np.ndarray, prior_raw: dict[str, np.ndarray],
    central_low_mask: np.ndarray, total_goals_arr: np.ndarray,
    outcome_idx: dict[str, np.ndarray], scoring_config: dict[str, Any],
) -> dict[str, int]:
    """Evaluación final: calcula exact/margin/winner hits además de puntos."""
    exact_pts = float(scoring_config.get("exact_score", 5))
    marg_pts  = float(scoring_config.get("same_margin_or_draw", scoring_config.get("margin_or_draw", 3)))

    # Reutilizar _evaluate pero capturar los picks individuales
    if conf_power > 1e-6:
        eff_w = weights[:, None] * (confidences ** conf_power)
    else:
        eff_w = weights[:, None] * np.ones_like(confidences)
    num     = np.einsum("mn,mns->ns", eff_w, matrices)
    blended = num / np.clip(eff_w.sum(axis=0)[:, None], 1e-10, None)

    sf = np.ones(len(central_low_mask), dtype=np.float64)
    if low_pen > 0:
        sf[central_low_mask] *= (1.0 - low_pen)
    if high_bonus > 0:
        hg_mask = total_goals_arr >= high_min
        sf[hg_mask] *= (1.0 + high_bonus * (total_goals_arr[hg_mask] - high_min + 1).clip(min=0))

    calibrated = np.zeros_like(blended)
    for o, idx in outcome_idx.items():
        raw_counts = prior_raw[o][idx]
        prior_cond = (raw_counts + smoothing); prior_cond /= prior_cond.sum()
        blended_o  = blended[:, idx]
        mass_o     = blended_o.sum(axis=1, keepdims=True)
        model_cond = blended_o / np.clip(mass_o, 1e-10, None)
        raw_cal    = (np.clip(model_cond, 1e-12, None) ** model_power
                      * np.clip(prior_cond, 1e-12, None) ** prior_power
                      * np.clip(sf[idx], 1e-12, None))
        calibrated[:, idx] = mass_o * raw_cal / np.clip(raw_cal.sum(axis=1, keepdims=True), 1e-10, None)

    best_idx = (calibrated @ points_matrix.T).argmax(axis=1)
    earned   = points_matrix[best_idx, actual_idx]

    return {
        "exact":  int((earned >= exact_pts - 1e-6).sum()),
        "margin": int(((earned >= marg_pts - 1e-6) & (earned < exact_pts - 1e-6)).sum()),
        "winner": int(((earned >= 1 - 1e-6) & (earned < marg_pts - 1e-6)).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start y progreso
# ─────────────────────────────────────────────────────────────────────────────

def _add_warmstart_trial(
    study: "optuna.Study",
    base_model_ids: list[str],
    target_config: dict[str, Any],
    existing_weights: dict[str, float],
) -> None:
    """Encola la configuración actual como punto de partida."""
    default_w = float(target_config.get("default_weight", 1.0))
    calib     = dict(target_config.get("scoreline_calibration", {}))

    params: dict[str, Any] = {}
    for mid in base_model_ids:
        w = float(existing_weights.get(mid, default_w))
        params[f"log_w_{mid}"] = float(np.log(max(w, 1e-4)))

    params["confidence_power"]    = float(target_config.get("confidence_power", 0.25))
    params["prior_power"]         = float(calib.get("prior_power",        0.34))
    params["model_power"]         = float(calib.get("model_power",        0.74))
    params["low_score_penalty"]   = float(calib.get("low_score_penalty",  0.18))
    params["high_goal_bonus"]     = float(calib.get("high_goal_bonus",    0.04))
    params["high_goal_min_total"] = int(calib.get("high_goal_min_total",  3))
    params["smoothing"]           = float(calib.get("smoothing",          0.35))

    # Clamp a los bounds del search space
    for k, (lo, hi) in CALIB_BOUNDS.items():
        params[k] = float(np.clip(params[k], lo, hi))
    params["high_goal_min_total"] = int(np.clip(params["high_goal_min_total"], *HIGH_MIN_BOUNDS))
    for mid in base_model_ids:
        params[f"log_w_{mid}"] = float(np.clip(params[f"log_w_{mid}"], -4.0, 0.0))

    try:
        study.enqueue_trial(params)
    except Exception:
        pass  # Si ya existe en storage, ignorar


class _ProgressCallback:
    def __init__(self, n_trials: int, max_possible: float) -> None:
        self.n_trials    = n_trials
        self.max_poss    = max_possible
        self.report_step = max(1, n_trials // 20)  # 20 actualizaciones

    def __call__(self, study: "optuna.Study", trial: "optuna.Trial") -> None:
        t = trial.number + 1
        if t % self.report_step == 0 or t == self.n_trials:
            bv = study.best_value
            print(
                f"  trial {t:>6}/{self.n_trials}  "
                f"mejor_eff={bv*100:.2f}%  "
                f"mejor_pts={bv*self.max_poss:.0f}/{self.max_poss:.0f}",
                flush=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Escritura de resultados
# ─────────────────────────────────────────────────────────────────────────────

def _update_models_config(
    models_path: Path,
    models_config: dict[str, Any],
    best_weight_dict: dict[str, float],
    bp: dict[str, Any],
    bt_run_id: str,
    best_pts: float,
    max_possible: float,
    n_trials: int,
    seed: int,
    extra: dict[str, int],
) -> None:
    for model in models_config.get("models", []):
        if model.get("model_id") != TARGET_MODEL_ID:
            continue

        model["weight_source"]    = "optimized_backtest"
        model["optimized_weights"]= best_weight_dict
        model["confidence_power"] = round(float(bp["confidence_power"]), 4)

        if "scoreline_calibration" not in model:
            model["scoreline_calibration"] = {}
        sc = model["scoreline_calibration"]
        sc["prior_power"]         = round(float(bp["prior_power"]),        4)
        sc["model_power"]         = round(float(bp["model_power"]),        4)
        sc["low_score_penalty"]   = round(float(bp["low_score_penalty"]),  4)
        sc["high_goal_bonus"]     = round(float(bp["high_goal_bonus"]),    4)
        sc["high_goal_min_total"] = int(bp["high_goal_min_total"])
        sc["smoothing"]           = round(float(bp["smoothing"]),          4)

        model["optimization"] = {
            "backtest_run_id"   : bt_run_id,
            "exact_hits"        : extra["exact"],
            "margin_or_draw_hits": extra["margin"],
            "max_possible_points": round(max_possible, 4),
            "method"            : "optuna_tpe_joint_calib",
            "n_trials"          : n_trials,
            "objective"         : "points",
            "score"             : round(best_pts / max_possible, 8),
            "seed"              : seed,
            "total_points"      : round(best_pts, 2),
            "winner_hits"       : extra["winner"],
        }
        break

    models_path.write_text(
        json.dumps(models_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raise SystemExit(main())
