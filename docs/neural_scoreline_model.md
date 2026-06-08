# Modelo neural_scoreline_mlp

Este modelo agrega una red neuronal independiente a la capa de prediccion. No usa como entrada los resultados de los modelos Poisson/Elo actuales, porque los priors existentes aun tienen rendimiento limitado. La red aprende directamente desde historico, forma reciente, ratings internos, contexto del partido y embeddings de equipos.

## Objetivo

Predecir una matriz completa de marcadores de `0-0` a `8-8`, probabilidades 1X2 y goles esperados. Con esa matriz se calcula:

- marcador mas probable;
- marcador que maximiza puntos esperados de quiniela;
- ganador/empate/perdedor.

## Estructura

- `configs/neural_scoreline.yaml`: hiperparametros de entrenamiento y ubicacion del artefacto.
- `src/quiniela/features/neural_features.py`: features temporales sin fuga de informacion futura.
- `src/quiniela/models/neural_scoreline_mlp.py`: arquitectura PyTorch y runner de prediccion.
- `src/quiniela/training/neural_dataset.py`: dataset PyTorch.
- `src/quiniela/training/neural_trainer.py`: entrenamiento, validacion por mundiales y guardado de artefactos.
- `scripts/train_neural_scoreline.py`: CLI de entrenamiento.

## Control temporal

Para cada ejemplo historico, las features se calculan con partidos anteriores a la fecha del partido. Los partidos del mismo dia se agregan al estado despues de construir sus ejemplos, para evitar que un resultado alimente otro encuentro del mismo dia.

## Artefactos esperados

El entrenamiento final guarda:

- `data/models/neural_scoreline/latest/model.pt`
- `data/models/neural_scoreline/latest/metadata.json`
- `data/models/neural_scoreline/latest/metrics.json`
- `data/models/neural_scoreline/latest/training_log.csv`

Durante el entrenamiento tambien guarda archivos intermedios:

- `checkpoint_last.pt`: ultimo estado entrenado, usado para reanudar si el proceso se corta.
- `checkpoint_best.pt`: mejor estado observado segun `valid_loss`.
- `metrics_live.json`: ultima metrica disponible para seguimiento mientras corre.
- `training_log.csv`: log acumulado por epoca.

Los folds historicos usan carpetas como `data/models/neural_scoreline/fold_2018` y `data/models/neural_scoreline/fold_2022`. El artefacto final queda en `data/models/neural_scoreline/latest`.

El runner no predice si esos artefactos no existen. En ese caso reporta estado `failed` para ese modelo, y el modelo permanece inactivo por defecto.

## Ejecucion

Con el entorno `quiniela2026` activado y desde `D:\Quiniela2026`:

```powershell
python scripts\train_neural_scoreline.py --device cuda
```

Para validar sin entrenar artefacto final:

```powershell
python scripts\train_neural_scoreline.py --device cuda --folds-only
```

Por defecto el entrenamiento reanuda desde `checkpoint_last.pt` si existe. Para empezar desde cero:

```powershell
python scripts\train_neural_scoreline.py --device cuda --fresh
```

## Tuning ola 2

La version `0.2.0` agrega:

- folds limpios 2014/2018/2022;
- loss adicional orientada a puntos de quiniela;
- metricas `top_mean_points` y `ev_mean_points`;
- guardrails para no degradar el 1X2 que sale de la matriz de marcadores;
- calibracion por temperatura guardada en `metadata.json`;
- script de busqueda de hiperparametros.

Para ejecutar tuning sin sobrescribir el modelo final actual:

```powershell
python scripts\tune_neural_scoreline.py --device cuda --max-trials 24
```

El tuning escribe:

- `data/models/neural_scoreline_tuning/tuning_results.csv`
- `data/models/neural_scoreline_tuning/tuning_summary.json`
- `data/models/neural_scoreline_tuning/best_config.json`
- carpetas `trial_001`, `trial_002`, etc. con checkpoints por fold.

Cuando se quiera entrenar el artefacto final con la mejor configuracion:

```powershell
python scripts\train_neural_scoreline.py --config data\models\neural_scoreline_tuning\best_config.json --output-root data\models\neural_scoreline --device cuda --fresh
```

Despues de entrenar, se puede ejecutar solo este modelo:

```powershell
python scripts\run_model.py --model neural_scoreline_mlp
```
