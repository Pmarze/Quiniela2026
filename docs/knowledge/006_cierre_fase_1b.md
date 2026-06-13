# 006 - Cierre de detalles Fase 1/1B

## Conocimiento

Se agrego una capa de canonicalizacion, normalizacion horaria, reconciliacion inicial y comando diario unico.

## Runtime

Validado con:

```text
python
```

## Comandos nuevos

Construir canon y reconciliacion:

```powershell
python scripts\build_canonical.py
```

Ejecutar flujo diario completo:

```powershell
python scripts\run_daily.py
```

Ejecutar flujo diario sin descarga:

```powershell
python scripts\run_daily.py --skip-download
```

## Orden diario vigente

```text
download_data -> build_canonical -> build_state -> generate_dashboard
```

## IDs canonicos

Equipos:

```text
team_mex
team_usa
team_bra
```

Partidos:

```text
wc2026_001 ... wc2026_104
```

## Horarios

El sistema normaliza horarios a:

```text
kickoff_local_iso
kickoff_utc
kickoff_timezone
kickoff_guatemala
```

El mapa editable esta en:

```text
configs/stadium_timezones.json
```

Supuesto actual:

```text
worldcup26_ir.kickoff_local es hora local del estadio.
```

## Reconciliacion vigente

La ultima validacion dejo:

```text
reconciliation_issues: 2
warning openfootball_worldcup_json kickoff_time_mismatch: 1
info rezarahiminia_static_csv no_matches_reconciled: 1
```

Los placeholders de eliminatoria no se tratan como errores de reconciliacion.

## Estado

Activo. No contradice conocimientos anteriores; agrega una capa nueva antes del estado vivo y modelos.
