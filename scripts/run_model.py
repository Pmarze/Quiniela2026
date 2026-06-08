from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from quiniela.ensemble import build_weighted_ensemble_predictions
from quiniela.models import (
    run_attack_defense_poisson,
    run_baseline_poisson,
    run_bayesian_monte_carlo_scoreline,
    run_bradley_terry_davidson,
    run_draw_specialist,
    run_elo_dixon_coles,
    run_elo_poisson,
    run_opta_power_poisson,
)
from quiniela.models.common import (
    ModelContext,
    ModelPrediction,
    load_json_config,
    load_model_context,
    store_predictions_in_sqlite,
    summarize_score_matrix,
    utc_now,
    write_prediction_artifacts,
)
from quiniela.models.neural_scoreline_mlp import run_neural_scoreline_mlp
from quiniela.models.neural_hybrid_v2 import run_neural_hybrid_v2


MODEL_RUNNERS = {
    "attack_defense_poisson": run_attack_defense_poisson,
    "baseline_poisson": run_baseline_poisson,
    "bayesian_monte_carlo_scoreline": run_bayesian_monte_carlo_scoreline,
    "bradley_terry_davidson": run_bradley_terry_davidson,
    "draw_specialist": run_draw_specialist,
    "elo_dixon_coles": run_elo_dixon_coles,
    "elo_poisson": run_elo_poisson,
    "neural_hybrid_v2": run_neural_hybrid_v2,
    "neural_scoreline_mlp": run_neural_scoreline_mlp,
    "opta_power_poisson": run_opta_power_poisson,
}


MODEL_FAMILIES = {
    "baseline_poisson": "control",
    "elo_poisson": "fuerza+goles",
    "elo_dixon_coles": "marcadores bajos",
    "attack_defense_poisson": "ataque/defensa",
    "bayesian_monte_carlo_scoreline": "Monte Carlo limpio",
    "draw_specialist": "empates",
    "bradley_terry_davidson": "1X2+empate",
    "neural_hybrid_v2": "red neuronal hibrida",
    "neural_scoreline_mlp": "red neuronal",
    "opta_power_poisson": "Opta externo",
    "weighted_ensemble": "ponderador",
    "weighted_points_ensemble": "ponderador puntos",
    "weighted_1x2_ensemble": "ponderador 1X2",
    "weighted_exact_ensemble": "ponderador exacto",
    "calibrated_scoreline_ensemble": "ponderador calibrado",
}

ENSEMBLE_MODEL_IDS = {
    "weighted_ensemble",
    "weighted_points_ensemble",
    "weighted_1x2_ensemble",
    "weighted_exact_ensemble",
    "calibrated_scoreline_ensemble",
}

# Modelos marcados como "referencia" en backtest: incluyen data leakage de redes neuronales
# entrenadas con el dataset completo. No se usan como Tier-1 en la selección dinámica,
# pero sí como fallback (Tier-2) si no hay modelos base disponibles.
_BACKTEST_REFERENCE_MODELS: frozenset[str] = frozenset({
    "baseline_poisson",
    "neural_scoreline_mlp",
    "neural_hybrid_v2",
    "opta_power_poisson",
    "weighted_ensemble",
    "weighted_points_ensemble",
    "weighted_1x2_ensemble",
    "weighted_exact_ensemble",
    "calibrated_scoreline_ensemble",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta modelos activos y publica predicciones de quiniela."
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "quiniela.db"),
        help="Ruta de la base SQLite.",
    )
    parser.add_argument(
        "--models-config",
        default=str(PROJECT_ROOT / "configs" / "models.yaml"),
        help="Configuracion de modelos activos.",
    )
    parser.add_argument(
        "--scoring-config",
        default=str(PROJECT_ROOT / "configs" / "scoring.yaml"),
        help="Reglas de puntaje de quiniela.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="model_id especifico a ejecutar. Puede repetirse. Si se omite, usa modelos activos.",
    )
    parser.add_argument(
        "--as-of-utc",
        default=None,
        help="Corte temporal ISO-8601. Si se omite, usa el as_of_utc del ultimo estado.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "predictions"),
        help="Carpeta raiz de artefactos de prediccion.",
    )
    parser.add_argument(
        "--ui-overrides",
        default=str(PROJECT_ROOT / "data" / "ui" / "prediction_overrides.json"),
        help="JSON que consume el dashboard local.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models_config = load_json_config(Path(args.models_config))
    scoring_config = load_json_config(Path(args.scoring_config))
    selected_models = _select_models(models_config, args.model)
    base_model_configs = [model for model in selected_models if not _is_ensemble_model(model)]
    ensemble_model_configs = [model for model in selected_models if _is_ensemble_model(model)]
    prediction_run_id = f"pred_{utc_now().replace('-', '').replace(':', '').replace('+00:00', 'Z')}_{uuid.uuid4().hex[:8]}"
    context = load_model_context(Path(args.db), prediction_run_id=prediction_run_id, as_of_utc=args.as_of_utc)
    output_dir = Path(args.output_root) / prediction_run_id

    print(f"prediction_run_id: {prediction_run_id}")
    print(f"as_of_utc: {context.as_of_utc}")
    print(f"training_data_version: {context.training_data_version}")
    print(f"tournament_state_id: {context.tournament_state_id}")
    print(f"training_matches: {len(context.training_matches)}")
    print(f"prediction_matches: {len(context.prediction_matches)}")

    predictions_by_model: dict[str, list[ModelPrediction]] = {}
    for model_config in base_model_configs:
        model_id = model_config["model_id"]
        runner = MODEL_RUNNERS.get(model_id)
        if runner is None:
            if model_config.get("required"):
                raise RuntimeError(f"Modelo requerido no implementado: {model_id}")
            print(f"{model_id}: skipped (no implementado)")
            continue
        predictions = runner(context, model_config, scoring_config)
        model_version = str(model_config.get("model_version", "0.1.0"))
        ok = sum(1 for prediction in predictions if prediction.status == "ok")
        masked = sum(1 for prediction in predictions if prediction.status == "masked")
        failed = len(predictions) - ok - masked
        json_path, csv_path = write_prediction_artifacts(
            output_dir=output_dir,
            model_id=model_id,
            model_version=model_version,
            context=context,
            predictions=predictions,
            notes=f"{model_id} generated by scripts/run_model.py",
        )
        store_predictions_in_sqlite(
            db_path=Path(args.db),
            model_id=model_id,
            model_version=model_version,
            context=context,
            predictions=predictions,
            json_path=json_path,
            csv_path=csv_path,
            notes=f"{model_id} generated by scripts/run_model.py",
        )
        predictions_by_model[model_id] = predictions
        print(f"{model_id}: ok={ok} masked={masked} failed={failed} json={json_path}")

    for model_config in ensemble_model_configs:
        model_id = str(model_config["model_id"])
        predictions = build_weighted_ensemble_predictions(
            context=context,
            predictions_by_model=predictions_by_model,
            model_config=model_config,
            scoring_config=scoring_config,
        )
        model_version = str(model_config.get("model_version", "0.1.0"))
        ok = sum(1 for prediction in predictions if prediction.status == "ok")
        masked = sum(1 for prediction in predictions if prediction.status == "masked")
        failed = len(predictions) - ok - masked
        json_path, csv_path = write_prediction_artifacts(
            output_dir=output_dir,
            model_id=model_id,
            model_version=model_version,
            context=context,
            predictions=predictions,
            notes=f"{model_id} generated by scripts/run_model.py",
        )
        store_predictions_in_sqlite(
            db_path=Path(args.db),
            model_id=model_id,
            model_version=model_version,
            context=context,
            predictions=predictions,
            json_path=json_path,
            csv_path=csv_path,
            notes=f"{model_id} generated by scripts/run_model.py",
        )
        predictions_by_model[model_id] = predictions
        print(f"{model_id}: ok={ok} masked={masked} failed={failed} json={json_path}")

    if predictions_by_model:
        ui_path = Path(args.ui_overrides)
        preferred_model_id = _select_preferred_model_id(
            db_path=Path(args.db),
            predictions_by_model=predictions_by_model,
            fallback_model_id=str(models_config.get("default_quiniela_model_id", "")),
        )
        write_ui_overrides(
            ui_path=ui_path,
            context=context,
            predictions_by_model=predictions_by_model,
            preferred_model_id=preferred_model_id,
        )
        print(f"ui_overrides: {ui_path}")
    else:
        print("no se generaron predicciones")
        return 1

    return 0


def _select_models(models_config: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    models = list(models_config.get("models", []))
    if requested:
        requested_set = set(requested)
        selected = [model for model in models if model.get("model_id") in requested_set]
        missing = requested_set - {model.get("model_id") for model in selected}
        if missing:
            raise RuntimeError(f"Modelos no definidos en configs/models.yaml: {', '.join(sorted(missing))}")
        return selected
    return [model for model in models if model.get("active")]


def _is_ensemble_model(model_config: dict[str, Any]) -> bool:
    return bool(model_config.get("ensemble")) or str(model_config.get("model_id")) in ENSEMBLE_MODEL_IDS


def _select_preferred_model_id(
    db_path: Path,
    predictions_by_model: dict[str, list[ModelPrediction]],
    fallback_model_id: str,
) -> str:
    """Elige dinámicamente el modelo preferido según el backtest más reciente.

    Lógica de selección:
      Tier 1 — modelos base con validación limpia (sin data leakage de redes neuronales),
               ordenados por total_quiniela_points DESC, brier_1x2 ASC.
      Tier 2 — todos los modelos disponibles (incluye ensembles y neuronales),
               misma métrica. Se usa si ningún modelo Tier-1 corrió en esta ejecución.
      Fallback — default_quiniela_model_id del config, si el backtest no está disponible
                 o ningún modelo del ranking está en predictions_by_model.
    """
    import sqlite3 as _sqlite3

    available = set(predictions_by_model.keys())
    if not available:
        return fallback_model_id

    try:
        conn = _sqlite3.connect(str(db_path))
        conn.row_factory = _sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT model_id, total_quiniela_points, brier_1x2
            FROM v_latest_backtest_model_metrics
            WHERE year = 'all'
            ORDER BY total_quiniela_points DESC, brier_1x2 ASC
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
    except Exception as exc:
        print(f"[preferred_model] backtest query failed ({exc}), usando fallback: {fallback_model_id}")
        return fallback_model_id

    if not rows:
        print(f"[preferred_model] sin métricas en backtest, usando fallback: {fallback_model_id}")
        return fallback_model_id

    tiers = [
        ("clean", [r for r in rows if r["model_id"] not in _BACKTEST_REFERENCE_MODELS]),
        ("all",   rows),
    ]
    for tier_label, candidates in tiers:
        for r in candidates:
            mid = r["model_id"]
            if mid in available:
                pts = r["total_quiniela_points"]
                brier = r["brier_1x2"]
                print(f"[preferred_model] elegido={mid}  pts={pts:.0f}  brier={brier:.4f}  tier={tier_label}")
                return mid

    print(f"[preferred_model] ningún modelo con backtest disponible, usando fallback: {fallback_model_id}")
    return fallback_model_id


def write_ui_overrides(
    ui_path: Path,
    context: ModelContext,
    predictions_by_model: dict[str, list[ModelPrediction]],
    preferred_model_id: str,
) -> None:
    existing = _load_existing_ui_overrides(ui_path)
    existing_matches = existing.get("matches", {})
    matches: dict[str, dict[str, Any]] = {}
    model_order = list(predictions_by_model)
    preferred = preferred_model_id if preferred_model_id in predictions_by_model else model_order[-1]

    by_source_match: dict[str, dict[str, ModelPrediction]] = {}
    for model_id, predictions in predictions_by_model.items():
        for prediction in predictions:
            by_source_match.setdefault(prediction.source_match_id, {})[model_id] = prediction

    for source_match_id, model_predictions in by_source_match.items():
        prior = existing_matches.get(source_match_id, {})
        frozen = bool(prior.get("frozen_pick"))
        preferred_prediction = model_predictions.get(preferred)
        if frozen and prior.get("quiniela_pick"):
            quiniela_pick = prior["quiniela_pick"]
        elif preferred_prediction and preferred_prediction.status == "ok":
            quiniela_pick = {
                "model_id": preferred_prediction.model_id,
                "score": preferred_prediction.selected_score,
                "expected_points": preferred_prediction.selected_expected_points,
                "top_score": preferred_prediction.top_score,
                "top_score_probability": preferred_prediction.top_score_probability,
            }
        else:
            quiniela_pick = None

        matches[source_match_id] = {
            "quiniela_pick": quiniela_pick,
            "frozen_pick": frozen,
            "model_predictions": [
                _dashboard_model_prediction(prediction)
                for prediction in model_predictions.values()
                if prediction.status == "ok"
            ],
            "notes": f"prediction_run_id={context.prediction_run_id}",
        }

    payload = {
        "generated_at_utc": utc_now(),
        "prediction_run_id": context.prediction_run_id,
        "as_of_utc": context.as_of_utc,
        "matches": matches,
    }
    ui_path.parent.mkdir(parents=True, exist_ok=True)
    ui_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _dashboard_model_prediction(prediction: ModelPrediction) -> dict[str, Any]:
    p_values = [
        ("1", prediction.p_team_a_win or 0.0),
        ("X", prediction.p_draw or 0.0),
        ("2", prediction.p_team_b_win or 0.0),
    ]
    outcome, confidence = max(p_values, key=lambda item: item[1])
    matrix_summary = summarize_score_matrix(prediction.score_matrix or {"scores": {}})
    return {
        "model_id": prediction.model_id,
        "family": MODEL_FAMILIES.get(prediction.model_id, "modelo"),
        "score": prediction.selected_score,
        "top_score": prediction.top_score,
        "outcome": outcome,
        "confidence": round(confidence, 4),
        "expected_goals": f"{prediction.expected_goals_a:.2f}-{prediction.expected_goals_b:.2f}",
        "p_team_a_win": prediction.p_team_a_win,
        "p_draw": prediction.p_draw,
        "p_team_b_win": prediction.p_team_b_win,
        "top_score_probability": matrix_summary.get("top_score_probability"),
        "expected_points": prediction.selected_expected_points,
        "notes": "\n".join(prediction.warnings),
    }


def _load_existing_ui_overrides(ui_path: Path) -> dict[str, Any]:
    if not ui_path.exists():
        return {"matches": {}}
    return json.loads(ui_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
