# 027 - Ponderador weighted_ensemble

## Conocimiento

Se implementa `weighted_ensemble` como capa modular de ponderacion. El ponderador consume las matrices de marcadores de los modelos base activos y publica una prediccion con el mismo contrato que cualquier modelo.

Se agregan tres variantes adicionales:

- `weighted_points_ensemble`: optimiza pesos hacia puntos de quiniela.
- `weighted_1x2_ensemble`: optimiza pesos hacia acierto ganador/empate/perdedor.
- `weighted_exact_ensemble`: optimiza pesos hacia acierto de marcador exacto.

## Regla

El ponderador no debe depender de detalles internos de los modelos. Solo consume `ModelPrediction`: matriz de marcador, probabilidades 1X2, goles esperados y estado `ok/masked/failed`.

## Pesos

En produccion diaria puede usar pesos desde el ultimo backtest disponible. Si no hay metricas, cae a `fallback_weights` configurados en `configs/models.yaml`.

En backtest se fuerza `weight_source=fallback` por defecto para evitar fuga temporal desde metricas historicas calculadas con los mismos partidos evaluados.

Si el ponderador consume modelos neuronales evaluados con artefactos finales (`neural_scoreline_mlp` o `neural_hybrid_v2`), el resultado del ponderador en el dashboard historico tambien debe tratarse como referencia visual, no como validacion limpia.

## Implementacion

- `src/quiniela/ensemble/weighted.py`
- `scripts/run_model.py`
- `src/quiniela/backtest/runner.py`
- `configs/models.yaml`
- `docs/weighted_ensemble.md`

El dashboard diario usa tres columnas en el hover para que los modelos ponderados y los modelos base quepan mejor. Las notas de pesos se muestran con salto de linea antes de `ensemble_weights`.

## Uso

```powershell
python scripts\run_model.py
python scripts\generate_dashboard.py
python scripts\run_backtest.py
python scripts\generate_validation_dashboard.py
```
