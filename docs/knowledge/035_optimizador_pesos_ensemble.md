# 035 - Optimizador de pesos por ensemble

## Contexto

Los ponderadores anteriores convertian metricas agregadas de backtest en pesos. Eso era util como primera aproximacion, pero no optimizaba directamente el comportamiento del ensemble como modelo de quiniela.

Ahora el backtest guarda `score_matrix_json` por modelo y partido. Con eso se puede reconstruir la mezcla de probabilidades de marcador, seleccionar el marcador que maximiza puntos esperados y evaluar cuantos puntos habria obtenido el ensemble en 2018 y 2022.

## Decision

Se agrego `scripts/optimize_ensemble_weights.py`.

El script:

- Lee el ultimo backtest disponible desde SQLite.
- Usa solo modelos base con matriz completa por partido.
- Respeta `exclude_models` de cada ensemble.
- Simula el mismo `confidence_power` que usa el ensemble en prediccion real.
- Busca pesos con muestreo Dirichlet aleatorio reproducible.
- Escribe `weight_source = optimized_backtest` y `optimized_weights` en `configs/models.yaml`.

Los objetivos quedan separados:

- `weighted_points_ensemble`: maximiza puntos de quiniela.
- `weighted_1x2_ensemble`: maximiza acierto 1X2 con desempates por puntos.
- `weighted_exact_ensemble`: maximiza marcador exacto con desempates por puntos.
- `weighted_ensemble`: combina puntos, exactos, margen/empate y ganador.
- `calibrated_scoreline_ensemble`: conserva calibracion historica de marcador, pero sus pesos se optimizan sobre la matriz base.

## Resultado

Backtest final:

```text
backtest_run_id = backtest_wc2018_2022_base_20260608T182120Z_8f2d0d44
```

Metricas principales:

```text
weighted_1x2_ensemble          195 / 640  exact=18  margin=41  winner=77
weighted_ensemble              195 / 640  exact=18  margin=41  winner=77
weighted_points_ensemble       195 / 640  exact=18  margin=41  winner=77
weighted_exact_ensemble        184 / 640  exact=20  margin=38  winner=68
calibrated_scoreline_ensemble  184 / 640  exact=14  margin=40  winner=76
```

`weighted_points_ensemble` quedo como `default_quiniela_model_id` porque, antes de que existan resultados reales del Mundial 2026, el criterio operativo sera usar el modelo orientado a maximizar puntos.

## Dashboard

El hover y el modal ahora muestran dos referencias:

- Modelo de quiniela: el modelo activo por defecto para generar la recomendacion principal.
- Mejor actual: el modelo con mayor puntaje acumulado en partidos ya jugados del torneo actual.

Mientras no haya partidos con resultado real, `Mejor actual` usa `weighted_points_ensemble` como fallback.

## Estado

Activo. Complementa la nota 034: el peso `0.9` del Monte Carlo sigue siendo fallback, pero el flujo normal usa `optimized_weights` cuando existe backtest optimizado.
