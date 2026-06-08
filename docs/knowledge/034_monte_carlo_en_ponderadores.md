# 034 - Monte Carlo en ponderadores

## Contexto

`bayesian_monte_carlo_scoreline` ya estaba incluido tecnicamente en los ponderadores, pero su `fallback_weight` seguia en `0.25`.

Ese peso era provisional, de la etapa en que el Monte Carlo aun no tenia backtest. En prediccion diaria los ensembles pueden usar `latest_backtest`, pero en backtest los ponderadores usan fallback para evitar fuga temporal. Por eso, en la evaluacion historica el Monte Carlo estaba entrando con peso artificialmente bajo.

## Decision

Se actualizo `bayesian_monte_carlo_scoreline` de `0.25` a `0.9` en los fallback weights de:

- `weighted_ensemble`
- `weighted_points_ensemble`
- `calibrated_scoreline_ensemble`
- `weighted_1x2_ensemble`
- `weighted_exact_ensemble`

El valor `0.9` se eligio porque el rendimiento relativo del Monte Carlo esta cerca de 85% a 92% de los mejores modelos base segun metrica, dependiendo de si se compara por puntos, exactos o 1X2. Ya no debe tratarse como modelo marginal, aunque tampoco debe dominar el ensemble.

## Resultado recalibrado

Backtest:

```text
backtest_run_id = backtest_wc2018_2022_base_20260608T174820Z_641cec28
```

Metricas principales:

```text
weighted_1x2_ensemble             159 / 640
weighted_exact_ensemble           156 / 640
weighted_ensemble                 153 / 640
weighted_points_ensemble          153 / 640
calibrated_scoreline_ensemble     152 / 640
bayesian_monte_carlo_scoreline    143 / 640
```

El aumento de peso del Monte Carlo redujo algo el rendimiento historico bruto de los ponderadores, pero ahora la comparacion refleja mejor lo que sucede cuando el Monte Carlo participa de verdad en el ensemble.

Corrida diaria:

```text
prediction_run_id = pred_20260608T175318Z_449e75bf
```

Peso promedio diario del Monte Carlo en los ponderadores:

```text
calibrated_scoreline_ensemble   ~0.127
weighted_points_ensemble        ~0.127
weighted_1x2_ensemble           ~0.131
weighted_exact_ensemble         ~0.126
```

## Estado

Activo. Complementa la nota 032. No contradice la decision de mantener `opta_power_poisson` fuera de los ponderadores limpios.
