# 011 - Mascara de evaluacion para partidos sin asignar

## Conocimiento

Los partidos sin equipos definidos, especialmente eliminatorias con placeholders, no deben contarse como errores del modelo ni afectar metricas futuras.

## Cambio Implementado

Las predicciones de partidos no asignados ahora se publican como:

```text
status: masked
is_evaluation_candidate: 0
mask_reason: unassigned_knockout_placeholder
```

Esto reemplaza la interpretacion previa de la nota 010, donde esos 32 partidos aparecian como `failed`. La nota 010 se conserva por historia; para evaluacion y reportes tiene prioridad esta nota 011.

## Tablas y Vistas

Columnas nuevas:

```text
model_prediction_runs.masked_predictions
model_predictions.is_evaluation_candidate
model_predictions.mask_reason
```

Vista nueva:

```text
v_latest_evaluable_model_predictions
```

Esta vista incluye solo:

```text
status = 'ok'
is_evaluation_candidate = 1
```

## Nueva Corrida Validada

```text
prediction_run_id: pred_20260605T171613Z_27e7869a
baseline_poisson: ok=72 masked=32 failed=0
elo_poisson: ok=72 masked=32 failed=0
```

Conteo evaluable:

```text
baseline_poisson: 72
elo_poisson: 72
```

## Regla Practica

Para cualquier backtest, reporte o metrica de modelos, usar:

```sql
SELECT *
FROM v_latest_evaluable_model_predictions;
```

No usar todos los registros de `v_latest_model_predictions` sin filtrar.

## Estado

Activo. Aclara y reemplaza, para evaluacion, la interpretacion de `failed=32` registrada en la nota 010.
