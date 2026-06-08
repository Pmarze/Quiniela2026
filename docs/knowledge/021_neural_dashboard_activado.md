# 021 - Modelo neural activado para dashboard diario

Fecha: 2026-06-05

## Decision

El modelo `neural_scoreline_mlp` ya tiene artefactos entrenados en `data/models/neural_scoreline/latest`, por lo que queda activo en `configs/models.yaml` para que `scripts/run_model.py` lo incluya en el dashboard diario de 2026.

## Cuidado importante

No se debe usar el artefacto final entrenado para comparar contra los mundiales 2018/2022 como si fuera backtest limpio, porque ese artefacto ya vio informacion posterior a esas fechas. Para una comparacion historica justa se deben usar folds walk-forward o entrenar artefactos por corte temporal.

## Archivos esperados

- `data/models/neural_scoreline/latest/model.pt`
- `data/models/neural_scoreline/latest/metadata.json`
- `data/models/neural_scoreline/latest/metrics.json`
- `data/models/neural_scoreline/training_summary.json`

## Comando diario

Con el entorno `quiniela2026` activado:

```powershell
python scripts\run_model.py
python scripts\generate_dashboard.py
```
