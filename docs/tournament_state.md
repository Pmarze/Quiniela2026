# Estado del Torneo

## Objetivo

Construir un estado vivo del Mundial a partir de la base SQLite normalizada. Este estado sera consumido por modelos, calibradores y reportes diarios.

## Comando

Con el entorno Conda activo:

```powershell
python scripts\build_state.py
```

O usando la ruta directa validada:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\build_state.py
```

## Entradas

El builder lee:

```text
v_canonical_matches
v_latest_completed_run
v_worldcup26_matches
v_worldcup26_group_standings
```

Si existe canon vigente, `build_state.py` usa `v_canonical_matches` para traer IDs canonicos y horarios normalizados. Si no existe, cae al flujo anterior basado en `v_worldcup26_matches`.

Por ahora la fuente operativa principal es:

```text
worldcup26_ir
```

## Salidas SQLite

Tablas:

```text
tournament_state_runs
state_matches
state_group_tables
state_team_form
```

Vistas para consumo:

```text
v_latest_tournament_state
v_latest_state_matches
v_latest_state_group_tables
v_latest_state_team_form
```

Los modelos deben preferir las vistas `v_latest_state_*` salvo que necesiten un `state_id` especifico para backtesting o auditoria.

## Salidas en Archivos

Cada estado tambien exporta CSV y metadata:

```text
data/state/{state_id}/matches.csv
data/state/{state_id}/group_tables.csv
data/state/{state_id}/team_form.csv
data/state/{state_id}/metadata.json
```

## Estado Actual Validado

Ultimo estado valido:

```text
state_id: state_20260605T051108Z_fd9766fe
source_run_id: run_20260605T051108Z_ef8b558c
as_of_utc: 2026-06-05T05:11:08Z
total_matches: 104
completed_matches: 0
pending_matches: 104
group_matches_completed: 0
teams: 48
groups: 12
```

## Regla Importante

`worldcup26_ir` puede traer marcadores `0-0` en partidos programados. Por eso, el estado no considera un partido completado solo porque existan goles numericos.

La regla vigente:

- Si `finished=1`, el partido esta completado.
- Si `status` indica final/completed/finished, el partido esta completado.
- Si `status` indica scheduled/upcoming/not_started, el partido esta pendiente aunque tenga score `0-0`.
- Solo si no hay estado claro, se usa la presencia de scores como fallback.

## Invalidacion de Estados

Si una corrida de estado queda incorrecta, se marca como `invalidated`:

```powershell
python scripts\invalidate_state.py {state_id} --notes "motivo"
```

Los estados invalidados quedan en historial, pero no aparecen en `v_latest_tournament_state`.
