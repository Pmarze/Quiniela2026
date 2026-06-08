# Modelo neural_hybrid_v2

`neural_hybrid_v2` es el segundo candidato de red neuronal del proyecto. No sustituye a `neural_scoreline_mlp`: vive en artefactos, configuracion, trainer y tuner separados.

## Objetivo

El modelo busca un salto de capacidad frente al MLP base usando:

- embeddings de equipos con interacciones A/B;
- bloques residuales densos;
- features historicas ya usadas por `neural_scoreline_mlp`;
- features de estado del torneo disponibles al cierre del dia anterior;
- una cabeza 1X2 separada que calibra la matriz de marcadores;
- seleccion de marcador por puntos esperados de quiniela.

## Regla temporal

La quiniela se llena un dia antes. Por eso el corte oficial de este modelo es `previous_day`:

- Para partidos del dia `D`, solo se usan resultados con `match_date < D`.
- No se usan partidos del mismo dia, aunque hayan ocurrido antes.
- En produccion 2026, los resultados ya completados del torneo entran como contexto solo desde el dia siguiente.

Esta regla aplica tanto al entrenamiento walk-forward como a la generacion diaria de features.

## Componentes

- `configs/neural_hybrid_v2.yaml`: hiperparametros base y espacio de tuning.
- `src/quiniela/features/hybrid_features.py`: features historicas + estado del torneo con corte `previous_day`.
- `src/quiniela/models/neural_hybrid_v2.py`: arquitectura PyTorch y runner de prediccion.
- `src/quiniela/training/neural_hybrid_trainer.py`: entrenamiento, checkpoints, logs y artefacto final.
- `src/quiniela/training/neural_hybrid_tuner.py`: busqueda de hiperparametros con guardrail 1X2.
- `scripts/train_neural_hybrid_v2.py`: CLI de entrenamiento final o por folds.
- `scripts/tune_neural_hybrid_v2.py`: CLI de tuning.

## Salidas del entrenamiento

El entrenamiento guarda progreso incremental en:

```text
data/models/neural_hybrid_v2/
  fold_2014/
  fold_2018/
  fold_2022/
  latest/
    checkpoint_last.pt
    checkpoint_best.pt
    metrics_live.json
    metrics.json
    training_log.csv
    model.pt
    metadata.json
```

Durante tuning, cada trial vive en:

```text
data/models/neural_hybrid_v2_tuning/trial_001/
```

El mejor resultado se resume en:

```text
data/models/neural_hybrid_v2_tuning/best_config.json
data/models/neural_hybrid_v2_tuning/tuning_results.csv
data/models/neural_hybrid_v2_tuning/tuning_summary.json
```

## Comandos

Con el entorno conda `quiniela2026` ya activo:

```powershell
python scripts\tune_neural_hybrid_v2.py --device cuda --max-trials 6
python scripts\tune_neural_hybrid_v2.py --device cuda --max-trials 48
python scripts\train_neural_hybrid_v2.py --config data\models\neural_hybrid_v2_tuning\best_config.json --output-root data\models\neural_hybrid_v2 --device cuda --fresh
```

Para una prueba sin tuning:

```powershell
python scripts\train_neural_hybrid_v2.py --device cuda --folds-only
```

## Activacion

`neural_hybrid_v2` queda registrado en `configs/models.yaml` con `active: false`. Debe activarse solo despues de generar:

```text
data/models/neural_hybrid_v2/latest/model.pt
data/models/neural_hybrid_v2/latest/metadata.json
```

Una vez activo, `scripts/run_model.py` y el backtest pueden usarlo como cualquier otro modelo.
