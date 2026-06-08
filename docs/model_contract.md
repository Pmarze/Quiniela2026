# Contrato Modular de Modelos

## Objetivo

Cada modelo debe poder cambiar sin romper el resto del proyecto. Para lograrlo, todos publican el mismo contrato de salida.

## Identidad del Modelo

Cada modelo debe tener:

- `model_id`: identificador estable, por ejemplo `elo_dixon_coles`.
- `model_version`: version semantica o hash.
- `run_id`: identificador de corrida.
- `as_of_utc`: corte temporal de la informacion usada.
- `created_at`: timestamp.
- `training_data_version`: version/snapshot de datos.
- `input_snapshot_id`: identificador del snapshot de datos.
- `tournament_state_id`: identificador del estado del torneo usado.

## Entrada Minima

Tabla de partidos:

```text
match_id
date
team_a
team_b
stage
neutral
venue
host_team
```

Features opcionales:

```text
elo_a
elo_b
fifa_rank_a
fifa_rank_b
market_p_a_win
market_p_draw
market_p_b_win
market_over_under_lines
recent_form_a
recent_form_b
```

## Salida Obligatoria

Cada modelo debe producir una fila por partido:

```text
run_id
as_of_utc
model_id
match_id
team_a
team_b
kickoff_utc
input_snapshot_id
tournament_state_id
expected_goals_a
expected_goals_b
p_team_a_win
p_draw
p_team_b_win
score_matrix_json
top_score               <- marcador de mayor probabilidad marginal (most_probable)
top_score_probability
selected_score          <- marcador que maximiza puntos esperados de quiniela (max_points)
selected_expected_points
status
warnings
```

### Diferencia entre top_score y selected_score

- `top_score` (`most_probable`): argmax de la distribución de probabilidad. El 0-0
  puede ser el más probable aun cuando 1-0 tenga mayor valor esperado.
- `selected_score` (`max_points`): argmax de `EV(score) = sum_k P(k) * quiniela_points(pred, k)`.
  Es la predicción operativa recomendada para la quiniela.

El dashboard y el backtest reportan métricas para ambas estrategias.

`score_matrix_json` debe representar una matriz de marcadores. Formato sugerido:

```json
{
  "max_goals": 8,
  "scores": {
    "0-0": 0.083,
    "1-0": 0.112,
    "0-1": 0.074
  }
}
```

## Reglas de Validez

- `p_team_a_win + p_draw + p_team_b_win` debe sumar aproximadamente 1.
- La matriz de marcadores debe sumar aproximadamente 1.
- Las probabilidades no pueden ser negativas.
- `as_of_utc` debe ser anterior al kickoff del partido pronosticado, salvo reportes post-partido marcados explicitamente como backtest.
- Si un modelo no puede estimar un partido, debe devolver `status=failed` y no romper el pipeline.
- Si un partido no tiene equipos asignados, por ejemplo eliminatorias con placeholders, debe devolver `status=masked` y `is_evaluation_candidate=0`.
- Las evaluaciones deben usar solo predicciones con `status=ok` e `is_evaluation_candidate=1`.

## Consumo por Ensemble

El ensemble no debe asumir que todos los modelos existen. Debe leer `configs/models.yaml`, filtrar modelos activos y aplicar estas reglas:

- Si `required=true` y falta el modelo, fallar con error claro.
- Si `required=false` y falta el modelo, continuar.
- Si un modelo tiene `status=failed`, ignorar esa prediccion para ese partido.
- Si solo queda un modelo valido, usarlo como fallback.
- `weighted_ensemble` mezcla matrices de marcadores normalizadas y publica su propia prediccion bajo el mismo contrato.
- En backtest, el ponderador debe evitar usar metricas historicas calculadas con los mismos partidos evaluados.
- `opta_power_poisson` queda excluido de los ponderadores activos por defecto para conservar una linea principal backtesteable sin Opta.

## Notebooks por Modelo

Cada modelo tendra su propio notebook:

```text
notebooks/01_model_elo_dixon_coles.ipynb
notebooks/02_model_market_calibrated_poisson.ipynb
notebooks/03_model_xg_or_stats_features.ipynb
notebooks/04_model_ml_1x2.ipynb
```

Regla importante:

Los notebooks no se llaman entre si. Cada notebook genera artefactos. El ensemble consume artefactos.

## Implementacion Actual

Modelos activos implementados (todos en `configs/models.yaml`):

```text
baseline_poisson        v0.1.0  — defaults sin tuning
elo_poisson             v0.2.2  — k=32, gs=0.35, ha=80 (GPU grid 1944t WC2018+2022, 2026-06-06)
elo_dixon_coles         v0.2.1  — k=36, gs=0.4, ha=80, rho=0.0 (GPU grid 1944t WC2018+2022)
draw_specialist         v0.2.1  — k=32, gs=0.7, ha=80, boost=0.05 (GPU grid 1440t WC2018+2022)
bradley_terry_davidson  v0.2.1  — k=8, gs=0.35, ha=80, draw_param=0.4 (CPU grid 1584t WC2018+2022)
attack_defense_poisson  v0.1.1  — ha=80, min_str=0.6, max_str=1.5, fallback=2.0 (CPU grid 4480t WC2018+2022)
bayesian_monte_carlo_scoreline v0.1.0 — Monte Carlo limpio sin Opta, 20k simulaciones diarias / 5k backtest
opta_power_poisson      v0.1.0  — senal externa 2026 con Opta Power Ratings/Rankings publicos + fallback Elo interno
neural_hybrid_v2        v0.1.0  — segundo modelo neural experimental con features historicas + estado del torneo
neural_scoreline_mlp    v0.1.0  — modelo neuronal independiente entrenado localmente con PyTorch
weighted_ensemble       v0.1.0  — ponderador balanceado
weighted_points_ensemble v0.1.0 — ponderador default para maximizar puntos de quiniela
weighted_1x2_ensemble   v0.1.0  — ponderador orientado a 1X2
weighted_exact_ensemble v0.1.0  — ponderador orientado a marcador exacto
calibrated_scoreline_ensemble v0.1.0 — ponderador default calibrado con prior historico mundialista
```

Los modelos v0.2.x comparten `_fit_elo_ratings` de `elo_poisson.py` con `combined_weight`
que incorpora `recency_weight` (cambio Ola 1). Parámetros tuneados con GPU/CPU full-grid
sobre WC2018+2022 (128 partidos). Ver notas 017 y 018.

Modelo pendiente y desactivado:

```text
market_calibrated_poisson  — requiere fuente de odds externas
```

Comando de predicción:

```powershell
python scripts\run_model.py
```

Artefactos:

```text
data/predictions/{prediction_run_id}/{model_id}.json
data/predictions/{prediction_run_id}/{model_id}.csv
data/ui/prediction_overrides.json
```

Vistas SQL:

```text
v_latest_model_prediction_runs
v_latest_model_predictions
v_latest_evaluable_model_predictions
```

El modelo default para propuesta de quiniela se define en:

```text
configs/models.yaml -> default_quiniela_model_id  (actualmente: calibrated_scoreline_ensemble)
```

Nota actual: `calibrated_scoreline_ensemble` esta activo como ponderador principal y `default_quiniela_model_id` debe apuntar a `calibrated_scoreline_ensemble`. Conserva las probabilidades 1X2 del ensemble y redistribuye marcador exacto con prior historico mundialista para reducir la concentracion excesiva en `1-1`, `1-0` y `0-1`. Los ponderadores consumen predicciones de modelos base activos y publican una prediccion propia bajo este mismo contrato.
