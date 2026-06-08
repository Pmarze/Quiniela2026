# Validacion Historica y Backtesting

## Objetivo

Validar los modelos actuales simulando que se hubieran usado antes de cada partido de los Mundiales 2014, 2018 y 2022.

La metrica principal es la misma de la quiniela:

```text
exact_score: 5
same_margin_or_draw: 3
winner: 1
```

## Regla Temporal

El backtest base usa una regla conservadora:

```text
match_date < fecha_del_partido
```

Esto significa que para un partido de un dia determinado, los modelos solo pueden entrenar con partidos historicos de fechas anteriores. No usan partidos posteriores ni partidos del mismo dia, porque la fuente historica actual no contiene kickoff exacto de Mundiales anteriores.

Cuando se incorpore una fuente con horarios historicos confiables, esta regla puede refinarse a:

```text
kickoff_utc < kickoff_del_partido
```

## Implementacion

Archivos:

```text
configs/backtest.yaml
src/quiniela/backtest/runner.py
src/quiniela/backtest/dashboard.py
scripts/run_backtest.py
scripts/generate_validation_dashboard.py
```

Comandos:

```powershell
python scripts\run_backtest.py
python scripts\generate_validation_dashboard.py
```

Con Python completo del entorno Conda:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\run_backtest.py
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\generate_validation_dashboard.py
```

## Artefactos

```text
data/backtests/{backtest_run_id}/backtest_results.json
data/backtests/{backtest_run_id}/backtest_predictions.csv
outputs/validation_dashboard/index.html
```

## Tablas SQLite

```text
backtest_runs
backtest_matches
backtest_predictions
backtest_model_metrics
backtest_parameter_trials
```

Vistas:

```text
v_latest_backtest_run
v_latest_backtest_model_metrics
v_latest_backtest_predictions
```

## Metricas

Por modelo y por año:

```text
matches_evaluated
exact_hits
margin_or_draw_hits
winner_hits
total_quiniela_points
max_possible_points
points_efficiency
mean_quiniela_points
exact_score_accuracy
margin_or_draw_accuracy
winner_accuracy
brier_1x2
log_loss_1x2
scoreline_log_loss
draw_predictions
draw_precision
draw_recall
```

## Estrategias de Seleccion de Marcador

El backtest evalúa dos estrategias distintas:

- **`max_points`** (`selected_score`): el marcador que maximiza los puntos esperados de quiniela
  ponderados por la distribución de probabilidad del modelo. Es la predicción recomendada operativamente.
- **`most_probable`** (`top_score`): el marcador con mayor probabilidad marginal sin ponderar
  por el esquema de puntos. Útil para comparar calibración de la distribución.

Ambas estrategias se reportan en el dashboard y en las métricas. La columna principal de eficiencia
usa `max_points`.

## Corridas Históricas de Referencia

### Corrida pre-Ola 1 (2018+2022, estrategia inicial)

```text
backtest_run_id: backtest_wc2018_2022_base_20260605T184106Z_eea438c7
years: 2018, 2022
matches: 128  |  max_possible_points: 640
```

| Modelo | Puntos | Eficiencia |
|---|---:|---:|
| `elo_dixon_coles` | 160 | 25.0% |
| `bradley_terry_davidson` | 156 | 24.4% |
| `draw_specialist` | 150 | 23.4% |
| `elo_poisson` | 149 | 23.3% |
| `baseline_poisson` | 108 | 16.9% |
| `attack_defense_poisson` | 97 | 15.2% |

### Corrida post-Ola 1 (2014+2018+2022, modelos v0.2.0 tuneados)

```text
years: 2014, 2018, 2022
matches: 192  |  max_possible_points: 960
```

| Modelo | Pts | Eficiencia | Brier |
|---|---:|---:|---:|
| `bradley_terry_davidson` | 504 | 26.3% | 0.610 |
| `draw_specialist` | 502 | 26.2% | 0.592 |
| `elo_poisson` (v0.2.0, k=32/gs=0.6/ha=80) | ~243* | 25.3% | 0.603 |

\* Solo elo_poisson tuneado individualmente; resto de modelos aún con tuning pendiente.

Referencia aleatoria: Brier ≈ 0.667. Objetivo Ola 1: > 28% de eficiencia.

## Proxima Etapa

1. Completar tuning exhaustivo de todos los modelos (GPU + CPU en curso).
2. Aplicar parámetros óptimos en `configs/models.yaml`.
3. Correr backtest final con todos los modelos v0.2.0 tuneados sobre 2014+2018+2022.
4. Evaluar si el ensemble supera al mejor modelo individual en hold-out 2022.
