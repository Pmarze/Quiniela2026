# 022 - Ola 2 del modelo neural: tuning y loss de quiniela

Fecha: 2026-06-05

## Decision

Antes de avanzar a otro modelo mas robusto, se mejora `neural_scoreline_mlp` porque el entrenamiento corre rapido en la laptop con GPU. La prioridad es exprimir el modelo neural actual con validacion limpia y una funcion objetivo mas cercana a la quiniela.

## Cambios implementados

- `configs/neural_scoreline.yaml` sube a `model_version` `0.2.0`.
- Los folds de validacion pasan a 2014, 2018 y 2022.
- `max_epochs` baja a 120 y `patience` a 14 porque el modelo anterior encontraba su mejor epoca temprano.
- La loss agrega `quiniela_reward`, que premia masa de probabilidad en marcadores que conservan exacto, empate/margen o ganador.
- La evaluacion agrega:
  - `top_mean_points`: puntos promedio usando el marcador mas probable.
  - `ev_mean_points`: puntos promedio usando el marcador que maximiza puntos esperados.
  - `ev_exact_accuracy`.
  - `ev_outcome_accuracy`.
- El tuning usa el objetivo `guarded_ev_points`, que prioriza puntos esperados pero penaliza configuraciones que degraden el 1X2 de la matriz o el 1X2 del pick de quiniela.
- El artefacto final guarda calibracion por temperatura en `metadata.json`.
- Se agrega `scripts/tune_neural_scoreline.py` para correr busqueda de hiperparametros sin sobrescribir el modelo final actual.

## Comandos

Con el entorno conda `quiniela2026` activado y estando en `D:\Quiniela2026`:

```powershell
python scripts\tune_neural_scoreline.py --device cuda --max-trials 24
```

Para entrenar el modelo final con el mejor config:

```powershell
python scripts\train_neural_scoreline.py --config data\models\neural_scoreline_tuning\best_config.json --output-root data\models\neural_scoreline --device cuda --fresh
```

Despues de entrenar el artefacto final, regenerar predicciones y dashboard:

```powershell
python scripts\run_model.py
python scripts\generate_dashboard.py
python scripts\generate_validation_dashboard.py
```

## Nota

El tuning escribe resultados en `data/models/neural_scoreline_tuning` y no reemplaza `data/models/neural_scoreline/latest` hasta que se ejecute explicitamente el entrenamiento final con `best_config.json`.
