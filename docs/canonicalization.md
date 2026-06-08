# Canonicalizacion y Reconciliacion

## Objetivo

Crear una superficie estable para modelos y reportes, separada de los detalles de cada fuente externa.

La fuente operativa primaria actual es:

```text
worldcup26_ir
```

Las fuentes secundarias se usan para reconciliacion y alertas.

## Comando

Con Anaconda activado:

```powershell
python scripts\build_canonical.py
```

## Flujo Diario Unico

Comando recomendado:

```powershell
python scripts\run_daily.py
```

Para correr sin descargar datos nuevos:

```powershell
python scripts\run_daily.py --skip-download
```

## Salidas SQLite

Tablas:

```text
canonical_build_runs
canonical_teams
canonical_matches
reconciliation_runs
reconciliation_issues
```

Vistas:

```text
v_latest_canonical_run
v_canonical_teams
v_canonical_matches
v_latest_reconciliation_run
v_latest_reconciliation_issues
```

## IDs Canonicos

Equipos:

```text
team_mex
team_usa
team_bra
```

Partidos:

```text
wc2026_001
wc2026_002
...
wc2026_104
```

Los modelos deben preferir estos IDs en vez de IDs propios de cada fuente.

## Horarios

El sistema normaliza cada partido a:

```text
kickoff_local_raw
kickoff_local_iso
kickoff_utc
kickoff_timezone
kickoff_guatemala
```

El mapa de zonas horarias vive en:

```text
configs/stadium_timezones.json
```

Supuesto operativo actual:

```text
worldcup26_ir.kickoff_local representa la hora local del estadio.
```

Si una fuente oficial contradice este supuesto, se debe documentar como contradiccion y pedir aprobacion antes de cambiar la regla.

## Reconciliacion

La reconciliacion inicial compara:

- Conteos por fuente.
- Partidos de seleccion por firma de equipos y fecha UTC.
- Diferencias de kickoff UTC cuando hay match equivalente.

No intenta reconciliar placeholders de eliminatoria como:

```text
1A
3A/B/C/D/F
W95
L101
Winner Match 101
```

## Estado Actual Validado

Ultima corrida diaria sin descarga:

```text
canonical.teams: 48
canonical.matches: 104
state.completed: 0
state.pending: 104
reconciliation.issues: 2
```

Avisos vigentes:

```text
warning openfootball_worldcup_json kickoff_time_mismatch: 1
info rezarahiminia_static_csv no_matches_reconciled: 1
```

El warning de openfootball corresponde a una diferencia de 30 minutos en un kickoff respecto al canon.

