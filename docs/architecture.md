# Arquitectura

## Capas

```text
snapshots -> canonicalizacion -> historicos/estado del torneo -> features -> modelos -> calibracion -> ensemble -> scoring -> reportes
                                        |
                                   [tuning layer]
                                 grid/random search
                                 sobre backtest WC
                                 configs/models.yaml
```

## Responsabilidades

### `src/quiniela/data/`

Ingesta y normalizacion:

- Fixtures.
- Resultados historicos.
- Rankings.
- Odds/mercados.
- Catalogo canonico de equipos.

### `src/quiniela/orchestration/`

Control de corridas:

- Crear `run_id`.
- Definir `as_of_utc`.
- Ejecutar pasos en orden.
- Registrar metadata.
- Evitar que un paso use informacion posterior al corte.

### `src/quiniela/state/`

Estado vivo del torneo:

- Partidos completados.
- Partidos pendientes.
- Tablas de grupo.
- Bracket real.
- Ratings actualizados.
- Forma reciente.
- Dias de descanso y fatiga.

Artefactos actuales:

- `tournament_state_runs`
- `state_matches`
- `state_group_tables`
- `state_team_form`
- `v_latest_state_matches`
- `v_latest_state_group_tables`
- `v_latest_state_team_form`

### `src/quiniela/history/`

Capa historica de entrenamiento:

- Descarga CSVs historicos versionados por corrida.
- Importa partidos internacionales a SQLite.
- Mapea equipos historicos contra equipos canonicos del Mundial 2026 cuando aplica.
- Calcula etiquetas de resultado, goles, diferencia y pesos iniciales por torneo/recencia.

Artefactos actuales:

- `history_ingestion_runs`
- `history_source_files`
- `canonical_historical_matches`
- `v_latest_history_run`
- `v_model_training_matches`
- `v_team_rating_inputs`

### `src/quiniela/features/`

Construccion de variables:

- Elo/rating difference.
- Forma reciente.
- Descanso y sede.
- Ranking FIFA.
- Goles a favor/en contra.
- Senales de mercado.

### `src/quiniela/models/`

Modelos independientes. Cada modelo debe poder correr solo y publicar un artefacto estandar.

Ejemplos:

- `baseline_poisson.py`
- `elo_poisson.py`
- `elo_dixon_coles.py`
- `market_calibrated_poisson.py`
- `ml_1x2.py`
- `xg_features_poisson.py`

Artefactos actuales:

- `data/predictions/{prediction_run_id}/{model_id}.json`
- `data/predictions/{prediction_run_id}/{model_id}.csv`
- `model_prediction_runs`
- `model_predictions`
- `v_latest_model_predictions`

### `src/quiniela/calibration/`

Ajustes de probabilidad:

- Normalizacion de odds.
- Remocion de vig si aplica.
- Calibracion isotonic/Platt si aplica.
- Reponderacion de matriz de marcadores para igualar probabilidades 1X2 externas.

### `src/quiniela/ensemble/`

Combinacion de modelos:

- Promedio ponderado.
- Stacking.
- Seleccion por desempeno historico.
- Fallback si un modelo no existe o falla.

### `src/quiniela/scoring/`

Reglas de quiniela:

- Resultado exacto.
- Empate o diferencia de goles.
- Ganador.
- Puntos esperados por marcador candidato.

Configuracion actual:

```text
configs/scoring.yaml
```

### `src/quiniela/simulation/`

Monte Carlo:

- Simulacion de partido.
- Simulacion de fase de grupos.
- Simulacion de eliminatorias.
- Propagacion de incertidumbre.

### `src/quiniela/evaluation/`

Metricas:

- Accuracy de marcador exacto.
- Accuracy de diferencia/empate.
- Accuracy 1X2.
- Brier.
- Log-loss.
- ECE.
- Puntos de quiniela reales y esperados.

### `scripts/tune_models*.py` — Capa de Tuning

Búsqueda de hiperparámetros sobre el backtest walk-forward. No es parte del pipeline
de producción diario; se corre a demanda para calibrar `configs/models.yaml`.

- `scripts/tune_models.py`: grid/random search en CPU con `ProcessPoolExecutor`.
  Soporta todos los modelos. Resultados en `data/backtests/tuning_<model>_<ts>.json`.
- `scripts/tune_models_gpu.py`: todos los trials simultáneos en GPU (PyTorch).
  50-200x más rápido para modelos Elo. Soporta elo_poisson, elo_dixon_coles, draw_specialist.
- `scripts/tune_all_gpu.bat` / `scripts/tune_all_cpu_rest.bat`: orquestación del tuning completo.

Los parámetros óptimos encontrados se aplican manualmente en `configs/models.yaml`
y se incrementa el `model_version` (actualmente v0.2.0 para modelos Elo).

### `src/quiniela/backtest/`

Validacion historica walk-forward:

- Reconstruye partidos de Mundiales anteriores desde `canonical_historical_matches`.
- Ejecuta modelos con corte temporal anterior a cada fecha.
- Calcula puntos reales de quiniela, aciertos y metricas probabilisticas.
- Guarda resultados en SQLite y artefactos JSON/CSV.
- Genera dashboard local de comparacion de modelos.

Artefactos actuales:

- `backtest_runs`
- `backtest_matches`
- `backtest_predictions`
- `backtest_model_metrics`
- `outputs/validation_dashboard/index.html`

## Flujo de Artefactos

Cada modelo escribe predicciones en:

```text
data/predictions/{run_id}/{model_id}.json
data/predictions/{run_id}/{model_id}.csv
```

El ensemble lee todos los artefactos activos declarados en `configs/models.yaml`.

Si se elimina un modelo:

- Se marca como inactivo en `configs/models.yaml`, o
- Se borra su artefacto y el ensemble lo omite si `required=false`.

El pipeline no debe importar notebooks. Los notebooks llaman codigo de `src/` y guardan artefactos.

## Flujo Diario

Cada corrida diaria crea artefactos independientes:

```text
data/raw/snapshots/{source}/{as_of_utc}/
data/state/{state_id}/
data/predictions/{run_id}/
outputs/{run_id}/
```

Esto permite comparar predicciones a traves del tiempo y auditar que datos existian antes de cada partido.
