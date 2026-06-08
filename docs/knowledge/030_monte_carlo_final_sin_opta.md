# 030 - Monte Carlo final sin Opta

## Conocimiento

El modelo final `bayesian_monte_carlo_scoreline` no debe considerar datos de Opta en su version principal.

La razon es que se quiere hacer backtest limpio contra los Mundiales 2018 y 2022. Como la informacion publica actual de Opta corresponde a 2026 y no tenemos snapshots historicos equivalentes por fecha, incluir Opta contaminaria la comparacion historica.

## Decision operativa

Solo `opta_power_poisson` debe depender de datos Opta.

Los ponderadores activos tambien excluyen `opta_power_poisson` por defecto para evitar dependencia indirecta de Opta en la propuesta principal de quiniela.

Si mas adelante se quiere evaluar una version que use Opta, debe crearse como modelo separado, por ejemplo:

```text
bayesian_monte_carlo_scoreline_opta
weighted_points_ensemble_with_opta
```

Esto permite comparar:

- modelos limpios backtesteables;
- modelos externos de referencia 2026;
- variantes experimentales con Opta sin mezclar sus metricas.

