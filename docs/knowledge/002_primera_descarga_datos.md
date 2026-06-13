# 002 - Primera descarga de datos

## Conocimiento

La primera descarga real de datos ya fue ejecutada por Codex. El usuario no necesita correr el pipeline inicial para tener datos base.

## Base creada

```text
D:\Quiniela2026\data\quiniela.db
```

## Resultado de la corrida exitosa

```text
run_id: run_20260605T051108Z_ef8b558c
as_of_utc: 2026-06-05T05:11:08Z
snapshots: 9
errors: 0
```

Resumen de tablas:

```text
ingestion_runs: 2
data_snapshots: 9
teams: 96
stadiums: 32
matches: 280
group_standings: 60
```

Por fuente:

```text
worldcup26_ir:
  teams: 48
  stadiums: 16
  matches: 104
  group_standings: 48

openfootball_worldcup_json:
  matches: 104

rezarahiminia_static_csv:
  teams: 48
  stadiums: 16
  matches: 72
  group_standings: 12
```

## Nota sobre la primera corrida fallida

Hubo una corrida previa fallida por restricciones de red del sandbox:

```text
run_id: run_20260605T051049Z_c777b65c
status: failed
```

Despues se ejecuto con permisos externos y la descarga fue exitosa.

## Comandos utiles

Revisar la base:

```powershell
python scripts\db_summary.py --samples
```

Actualizar datos otro dia:

```powershell
python scripts\download_data.py
```

## Estado

Activo. No contradice conocimientos anteriores.
