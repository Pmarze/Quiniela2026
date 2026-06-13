# 004 - Estado del torneo

## Conocimiento

Ya existe una capa implementada para construir el estado vivo del Mundial desde SQLite.

## Runtime

Se valido usando el entorno Conda principal:

```text
python
```

## Comando principal

Con Anaconda activado y estando en la carpeta del proyecto:

```powershell
python scripts\build_state.py
```

## Estado vigente

Ultimo estado valido construido:

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

## Vistas para siguientes pasos

Los modelos deben consultar preferentemente:

```text
v_latest_tournament_state
v_latest_state_matches
v_latest_state_group_tables
v_latest_state_team_form
```

## Leccion de datos

La fuente `worldcup26_ir` puede traer partidos programados con score `0-0`. No se debe usar la sola presencia de goles numericos como criterio de partido completado.

Regla vigente:

- `finished=1` o `status` final => completado.
- `scheduled/upcoming/not_started` => pendiente aunque el score sea `0-0`.
- Scores numericos solo son fallback si el estado no es claro.

## Estados invalidados

Durante la validacion se generaron dos estados de prueba y fueron invalidados:

```text
state_20260605T051108Z_b493f664
state_20260605T051108Z_94f6772d
```

No deben usarse para modelos.

## Estado

Activo. No contradice conocimientos anteriores; agrega una regla nueva sobre interpretacion de estados de partido.
