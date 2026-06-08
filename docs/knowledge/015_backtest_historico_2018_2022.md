# 015 - Backtest historico 2018/2022

## Conocimiento

Se implemento una fase de validacion historica antes del ponderador de modelos.

La validacion simula los Mundiales 2018 y 2022 con los modelos activos y aplica la regla:

```text
match_date < fecha_del_partido
```

Esto evita usar informacion futura. Como la fuente historica actual no tiene kickoff exacto de esos Mundiales, la regla es conservadora y no usa partidos del mismo dia.

## Archivos

```text
configs/backtest.yaml
src/quiniela/backtest/runner.py
src/quiniela/backtest/dashboard.py
scripts/run_backtest.py
scripts/generate_validation_dashboard.py
docs/backtesting_validation.md
```

## Tablas y Vistas

```text
backtest_runs
backtest_matches
backtest_predictions
backtest_model_metrics
backtest_parameter_trials
v_latest_backtest_run
v_latest_backtest_model_metrics
v_latest_backtest_predictions
```

## Primera Corrida

```text
backtest_run_id: backtest_wc2018_2022_base_20260605T180324Z_73745264
years: 2018, 2022
matches: 128
predictions: 768
models: 6
```

Ranking total:

```text
elo_dixon_coles: 160 puntos, 15 exactos, 35 margen/empate, 60 ganador
bradley_terry_davidson: 156 puntos, 14 exactos, 30 margen/empate, 68 ganador
draw_specialist: 150 puntos, 13 exactos, 31 margen/empate, 62 ganador
elo_poisson: 149 puntos, 13 exactos, 29 margen/empate, 65 ganador
baseline_poisson: 108 puntos, 12 exactos, 28 margen/empate, 28 ganador
attack_defense_poisson: 97 puntos, 5 exactos, 21 margen/empate, 45 ganador
```

## Dashboard

Se genero una segunda pagina local:

```text
outputs/validation_dashboard/index.html
```

Muestra:

```text
ranking de modelos por puntos
filtros por año, modelo y fase
metricas agregadas
tabla partido por partido con resultado real, pick, top score, puntos, 1X2 y xG
```

## Decision

No se optimizaron parametros todavia. Esta corrida es el diagnostico base.

La siguiente fase debe usar:

```text
2018 como calibracion
2022 como validacion
```

Luego se pueden elegir pesos para el ensemble con base en puntos, exactos y estabilidad por año.

## Estado

Activo. Amplia la nota 014 y prepara el ponderador/ensemble.
