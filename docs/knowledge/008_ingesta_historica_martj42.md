# 008 - Ingesta historica martj42

## Conocimiento

Ya existe una implementacion ejecutable para descargar e importar la fuente historica principal `martj42/international_results`.

## Runtime

Se ejecuto usando el entorno Conda principal:

```text
C:\Users\pablo\.conda\envs\quiniela2026\python.exe
```

## Archivos creados o actualizados

```text
src/quiniela/history/pipeline.py
scripts/build_history.py
configs/history_sources.json
src/quiniela/storage/sqlite_store.py
scripts/db_summary.py
docs/history_layer.md
docs/data_storage.md
docs/daily_update_workflow.md
docs/architecture.md
docs/data_sources.md
README.md
PROJECT_CONTEXT.md
```

## Comando principal

Con Anaconda activado y estando en la carpeta del proyecto:

```powershell
python scripts\build_history.py
```

Con ruta completa al Python del entorno:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\build_history.py
```

## Primera corrida exitosa

```text
history_run_id: history_20260605T065321Z_1846474b
as_of_utc: 2026-06-05T06:53:21Z
sources_checked: 1
files_downloaded: 4
matches_imported: 49318
```

## Cobertura inicial validada

```text
total training matches: 49318
matches with at least one Mundial 2026 team mapped: 25890
matches with both teams mapped: 7517
world cup matches: 1024
qualifiers: 15928
friendlies: 18312
```

## Tablas y vistas nuevas

```text
history_ingestion_runs
history_source_files
canonical_historical_matches
v_latest_history_run
v_canonical_historical_matches
v_model_training_matches
v_team_rating_inputs
```

## Nota de permisos

La primera ejecucion sin permisos externos fallo por bloqueo de red del sandbox de Windows. Luego se ejecuto con permiso externo y la descarga fue exitosa.

## Estado

Activo. No contradice conocimientos anteriores; materializa la capa historica definida en la nota 007.
