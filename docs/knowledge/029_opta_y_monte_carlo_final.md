# 029 - Opta como senal diaria y Monte Carlo final

## Estado

Parcialmente reemplazada por `030_monte_carlo_final_sin_opta.md`.

La parte reemplazada es la idea de que el Monte Carlo final use Opta como prior. La decision actual es que el Monte Carlo final no debe consumir Opta para poder backtestearse limpiamente contra 2018/2022.

## Conocimiento

Se agrego `opta_power_poisson` como modelo activo de referencia externa para el Mundial 2026. Usa informacion publica de Opta Analyst/Stats Perform cuando existe, fallback Elo interno cuando falta cobertura publica y resultados reales del torneo solo si ocurrieron antes del corte `as_of_utc`.

Este modelo no debe considerarse backtest limpio 2018/2022 mientras no tengamos ratings Opta historicos archivados por fecha. Por defecto queda excluido de `configs/backtest.yaml`, aunque puede aparecer como referencia si se decide incluirlo manualmente.

En los ponderadores se permite como senal secundaria con `fallback_weight=0.20`. La razon es que todavia no tiene metrica historica comparable y no debe dominar la quiniela antes de acumular evaluacion diaria.

## Decision operativa

La evaluacion honesta de `opta_power_poisson` sera diaria durante el Mundial:

```text
ingesta resultado real -> congelar pick anterior -> actualizar estado del torneo -> recalcular predicciones futuras
```

Tambien se definio la propuesta del ultimo modelo robusto a implementar:

```text
bayesian_monte_carlo_scoreline
```

Este modelo debe ser Monte Carlo real: muestrear incertidumbre de fuerza/equipo/lambdas, simular marcadores y simular el torneo completo muchas veces. Opta debe entrar como prior con varianza, no como verdad dura.

## Documentos relacionados

- `docs/opta_power_poisson.md`
- `docs/monte_carlo_final_model_proposal.md`
- `docs/data_sources.md`
