# 020 - Progreso y checkpoints del entrenamiento neural

Fecha: 2026-06-05

## Decision

El entrenamiento de `neural_scoreline_mlp` debe imprimir pasos intermedios y guardar estado por epoca para que el usuario pueda confirmar avance y no pierda todo si se corta el proceso, se satura la RAM o se apaga la computadora.

## Implementacion

- `scripts/train_neural_scoreline.py` acepta `--fresh` para ignorar checkpoints previos.
- Por defecto reanuda desde `checkpoint_last.pt` si existe.
- Cada entrenamiento/fold guarda:
  - `checkpoint_last.pt`: ultimo estado.
  - `checkpoint_best.pt`: mejor estado por `valid_loss`.
  - `metrics_live.json`: metricas de la ultima epoca.
  - `training_log.csv`: historial por epoca.
- La terminal imprime carga de datos, construccion de ejemplos, inicio de folds, epoca, train loss, valid loss, accuracy exacta, accuracy 1X2, paciencia y tiempo por epoca.

## Comando recomendado

Con el entorno conda `quiniela2026` activado y estando en `D:\Quiniela2026`:

```powershell
python scripts\train_neural_scoreline.py --device cuda
```

Si se quiere reiniciar desde cero:

```powershell
python scripts\train_neural_scoreline.py --device cuda --fresh
```
