# 019 - Modelo neural independiente

Fecha: 2026-06-05

## Decision

Se implementa `neural_scoreline_mlp` como un modelo independiente, no como residual de los modelos Poisson/Elo actuales. La razon es que los priors existentes tienen rendimiento aproximado de 25%, por lo que conviene que la primera version neuronal aprenda desde datos historicos y features propias antes de mezclar predicciones externas.

## Implicaciones

- El modelo queda `active: false` hasta entrenar y validar artefactos.
- El entrenamiento usa PyTorch y puede usar GPU con `--device cuda`.
- Las features se construyen sin informacion futura: cada partido historico solo ve resultados anteriores.
- La salida principal es una matriz de marcadores `0-0` a `8-8`, desde la cual se obtiene el marcador mas probable y el marcador que maximiza puntos esperados.

## Archivos relevantes

- `configs/neural_scoreline.yaml`
- `src/quiniela/features/neural_features.py`
- `src/quiniela/models/neural_scoreline_mlp.py`
- `src/quiniela/training/neural_trainer.py`
- `scripts/train_neural_scoreline.py`
- `docs/neural_scoreline_model.md`

## Ejecucion

Con el entorno conda `quiniela2026` ya activado y estando en `D:\Quiniela2026`:

```powershell
python scripts\train_neural_scoreline.py --device cuda
```

No se debe lanzar el entrenamiento mientras exista otro proceso pesado corriendo en la laptop. Primero se deja el codigo listo y luego se confirma con el usuario antes de ejecutar.
