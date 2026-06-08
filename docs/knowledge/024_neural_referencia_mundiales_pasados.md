# 024 - Neural como referencia en dashboard historico

Fecha: 2026-06-05

## Decision

El dashboard de mundiales pasados debe mostrar tambien resultados de `neural_scoreline_mlp` usando el artefacto final entrenado, aunque no sea un backtest limpio. Se presenta como referencia visual para entender comportamiento del modelo con la data disponible.

## Cuidado

`neural_scoreline_mlp` queda marcado como `reference_model` en `configs/backtest.yaml`. Esto significa que puede haber fuga temporal porque el artefacto final pudo ver informacion posterior a los mundiales evaluados. No debe compararse como validacion estricta contra los modelos walk-forward.

## Implementacion

- `src/quiniela/backtest/runner.py` incluye el runner `neural_scoreline_mlp`.
- `configs/backtest.yaml` agrega `reference_models`.
- `src/quiniela/backtest/dashboard.py` muestra una nota y una etiqueta `Referencia` para esos modelos.

## Comando

Con el entorno `quiniela2026` activado:

```powershell
python scripts\run_backtest.py
python scripts\generate_validation_dashboard.py
```
