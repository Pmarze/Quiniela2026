# Almacenamiento de Datos

## Objetivo

Guardar datos de forma eficiente, reproducible y facil de consultar por los siguientes pasos del pipeline.

## Capas

```text
data/raw/snapshots/   payloads originales comprimidos
data/raw/history/     CSVs historicos descargados por corrida
data/quiniela.db      base SQLite normalizada
data/state/           estado derivado por corrida
data/predictions/     artefactos de modelos
data/backtests/       artefactos de validacion historica
outputs/              recomendaciones finales
```

## Snapshots Crudos

Cada descarga se guarda como archivo comprimido:

```text
data/raw/snapshots/{source}/{as_of_utc}/{resource}.json.gz
data/raw/snapshots/{source}/{as_of_utc}/{resource}.metadata.json
```

Esto permite reproducir una corrida aunque la fuente externa cambie.

## SQLite

La base principal esta en:

```text
data/quiniela.db
```

La primera descarga real ya fue ejecutada y dejo datos base en esa ruta. Ver detalle historico en:

```text
docs/knowledge/002_primera_descarga_datos.md
```

Tablas iniciales:

- `ingestion_runs`: corridas de descarga.
- `data_snapshots`: metadata de cada snapshot.
- `canonical_build_runs`: corridas de canonicalizacion.
- `canonical_teams`: equipos canonicos vigentes.
- `canonical_matches`: partidos canonicos vigentes.
- `reconciliation_runs`: corridas de reconciliacion.
- `reconciliation_issues`: avisos/discrepancias entre fuentes.
- `history_ingestion_runs`: corridas de ingesta historica para modelos.
- `history_source_files`: metadata de CSVs historicos descargados.
- `canonical_historical_matches`: partidos historicos canonicos por corrida historica.
- `model_prediction_runs`: corridas por modelo.
- `model_predictions`: predicciones por partido, modelo y corrida.
- `teams`: equipos normalizados por fuente.
- `stadiums`: sedes normalizadas por fuente.
- `matches`: partidos, resultados y estado.
- `group_standings`: tablas de grupo.
- `tournament_state_runs`: corridas de estado del torneo.
- `state_matches`: partidos del estado construido.
- `state_group_tables`: standings recalculados por estado.
- `state_team_form`: forma y acumulados por equipo.
- `backtest_runs`: corridas de validacion historica.
- `backtest_matches`: partidos historicos evaluados.
- `backtest_predictions`: predicciones historicas por modelo/partido.
- `backtest_model_metrics`: metricas agregadas por modelo y año.
- `backtest_parameter_trials`: espacio reservado para optimizacion de parametros.

Vistas iniciales:

- `v_worldcup26_matches`: partidos de la fuente operativa principal enriquecidos con nombres de equipos y sedes.
- `v_worldcup26_group_standings`: standings enriquecidos con nombres de equipos.
- `v_latest_completed_run`: ultima corrida completada.
- `v_latest_canonical_run`: ultima canonicalizacion valida.
- `v_canonical_teams`: equipos canonicos vigentes.
- `v_canonical_matches`: partidos canonicos vigentes con horarios normalizados.
- `v_latest_reconciliation_run`: ultima reconciliacion.
- `v_latest_reconciliation_issues`: avisos de la ultima reconciliacion.
- `v_latest_history_run`: ultima ingesta historica completada.
- `v_canonical_historical_matches`: partidos historicos de la ultima ingesta.
- `v_model_training_matches`: dataset historico listo para modelos.
- `v_team_rating_inputs`: insumos historicos para Elo/rating.
- `v_latest_prediction_batch`: ultimo lote de predicciones completado.
- `v_latest_model_prediction_runs`: corridas de modelos del ultimo lote.
- `v_latest_model_predictions`: predicciones del ultimo lote.
- `v_latest_evaluable_model_predictions`: predicciones validas para evaluacion.
- `v_latest_tournament_state`: ultimo estado del torneo valido.
- `v_latest_state_matches`: partidos del ultimo estado valido.
- `v_latest_state_group_tables`: tablas del ultimo estado valido.
- `v_latest_state_team_form`: forma del ultimo estado valido.
- `v_latest_backtest_run`: ultima validacion historica.
- `v_latest_backtest_model_metrics`: metricas de la ultima validacion.
- `v_latest_backtest_predictions`: predicciones de la ultima validacion.

## Comando Inicial

Con el entorno Conda `quiniela2026`:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\download_data.py
```

Para ver un resumen:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\db_summary.py --samples
```

Para recrear tablas/vistas sin descargar datos:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\init_db.py
```

Para descargar e importar historicos de entrenamiento:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\build_history.py
```

Para ejecutar modelos activos:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\run_model.py
```

Para validar modelos contra Mundiales 2018 y 2022:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\run_backtest.py
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\generate_validation_dashboard.py
```

## Notas

SQLite es suficiente para empezar porque:

- No requiere instalar dependencias.
- Permite consultas SQL inmediatas.
- Es portable y versionable como artefacto local.
- Puede migrar luego a DuckDB/Parquet si el volumen crece.
