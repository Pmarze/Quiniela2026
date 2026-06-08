# 010 - Modelos iniciales

## Conocimiento

Ya existe una primera implementacion funcional de modelos modulares para la quiniela.

## Modelos Implementados

```text
baseline_poisson
elo_poisson
```

`baseline_poisson` usa el promedio historico ponderado de goles por equipo como benchmark minimo.

`elo_poisson` entrena ratings Elo secuenciales con el historico importado, usa peso por importancia de torneo, escala por diferencia de goles y convierte diferencia Elo en goles esperados mediante una matriz Poisson.

## Archivos Principales

```text
configs/scoring.yaml
configs/models.yaml
src/quiniela/models/__init__.py
src/quiniela/models/common.py
src/quiniela/models/baseline_poisson.py
src/quiniela/models/elo_poisson.py
src/quiniela/scoring/__init__.py
src/quiniela/scoring/quiniela.py
scripts/run_model.py
notebooks/01_model_elo_dixon_coles.ipynb
```

## Tablas y Vistas Nuevas

```text
model_prediction_runs
model_predictions
v_latest_prediction_batch
v_latest_model_prediction_runs
v_latest_model_predictions
```

## Comando Principal

Con Anaconda activado y estando en la carpeta del proyecto:

```powershell
python scripts\run_model.py
```

Luego regenerar dashboard:

```powershell
python scripts\generate_dashboard.py
```

## Primera Corrida Exitosa

```text
prediction_run_id: pred_20260605T164137Z_5bcdbc9c
as_of_utc: 2026-06-05T16:11:38Z
training_data_version: history_20260605T065321Z_1846474b
tournament_state_id: state_20260605T161138Z_e529121f
training_matches: 49318
prediction_matches: 104
baseline_poisson: ok=72 failed=32
elo_poisson: ok=72 failed=32
```

Los 32 fallos corresponden a partidos de eliminatoria con placeholders; eso es esperado antes de conocer clasificados.

## Artefactos Generados

```text
data/predictions/pred_20260605T164137Z_5bcdbc9c/baseline_poisson.json
data/predictions/pred_20260605T164137Z_5bcdbc9c/baseline_poisson.csv
data/predictions/pred_20260605T164137Z_5bcdbc9c/elo_poisson.json
data/predictions/pred_20260605T164137Z_5bcdbc9c/elo_poisson.csv
data/ui/prediction_overrides.json
outputs/dashboard/index.html
```

## Criterio de Quiniela

El selector inicial usa:

```text
exact_score: 5
same_margin_or_draw: 3
winner: 1
```

El pick recomendado por ahora sale del modelo configurado como:

```text
default_quiniela_model_id: elo_poisson
```

## Estado

Activo. No contradice conocimientos anteriores; implementa el primer bloque propuesto en la nota 009.
