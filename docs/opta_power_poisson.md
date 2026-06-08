# Modelo opta_power_poisson

## Objetivo

`opta_power_poisson` es un modelo externo de referencia para el Mundial 2026. No intenta reproducir exactamente el Opta Supercomputer, porque Opta no publica todo su modelo ni todos sus insumos, pero usa la informacion publica disponible de Opta como prior de fuerza y la combina con el historico local.

Este modelo esta pensado para operacion diaria durante el torneo:

- antes de que empiece el Mundial usa ratings/rankings Opta publicos y fallback Elo interno;
- despues de cada jornada incorpora solo resultados reales ya registrados antes del corte `as_of_utc`;
- no debe usar resultados del mismo dia si la quiniela se llena un dia antes;
- no se valida como backtest limpio 2018/2022 mientras no tengamos ratings Opta historicos archivados por fecha.

## Fuentes Opta usadas

Archivo local:

```text
data/external/opta/opta_power_ratings_20260607.json
```

Fuentes publicas consultadas:

- https://theanalyst.com/articles/world-cup-groups-2026-easiest-hardest-opta-power-rankings
- https://theanalyst.com/articles/world-cup-2026-group-a-predictions-preview
- https://theanalyst.com/articles/who-will-win-2026-fifa-world-cup-predictions-opta-supercomputer
- https://www.statsperform.com/products/opta-data/
- https://www.statsperform.com/products/opta-vision/

Lo publico hoy permite registrar ratings exactos para algunos equipos, ranking para otros, probabilidades de Grupo A y probabilidades de campeon/camino para equipos principales. Cuando falta un equipo, el modelo usa fallback Elo derivado del historico local.

## Logica del modelo

1. Carga el snapshot Opta local.
2. Ajusta un Elo historico interno con `training_matches`.
3. Construye rating por equipo:
   - si hay `opta_power_rating`, lo transforma a escala Elo y lo mezcla con Elo interno;
   - si solo hay `opta_power_rank`, estima una calificacion aproximada desde el ranking y la mezcla con Elo interno;
   - si no hay informacion Opta, usa Elo interno.
4. Lee partidos reales completados en `v_latest_state_matches` con `kickoff_utc < as_of_utc`.
5. Aplica actualizaciones tipo Elo a esas fuerzas antes de pronosticar partidos futuros.
6. Convierte diferencia de ratings en goles esperados.
7. Genera matriz de marcadores Poisson.
8. Publica:
   - marcador mas probable (`top_score`);
   - marcador que maximiza puntos de quiniela (`selected_score`);
   - probabilidades 1X2;
   - notas de fuente y fallback.

## Integracion

Codigo:

```text
src/quiniela/models/opta_power_poisson.py
```

Runner diario:

```text
scripts/run_model.py
```

Configuracion:

```text
configs/models.yaml
```

Backtest:

```text
configs/backtest.yaml
```

`opta_power_poisson` esta activo en prediccion diaria y excluido del backtest limpio por defecto.

## Relacion con ponderadores

Los ponderadores activos no consumen `opta_power_poisson` por defecto.

Esto es intencional: queremos que solo `opta_power_poisson` dependa de Opta. Los ensembles y el futuro Monte Carlo final deben poder backtestearse contra 2018/2022 sin depender de una fuente publica 2026. Si mas adelante se quiere comparar una version con Opta, debe crearse un modelo separado, por ejemplo `weighted_points_ensemble_with_opta` o `bayesian_monte_carlo_scoreline_opta`.

## Comandos

Con el entorno conda `quiniela2026` activo:

```powershell
python scripts\run_model.py
python scripts\generate_dashboard.py
```

Para una comprobacion aislada del modelo sin sobrescribir el dashboard:

```powershell
python -c "from pathlib import Path; import sys; sys.path.insert(0, 'src'); from quiniela.models.common import load_model_context, load_json_config; from quiniela.models.opta_power_poisson import run_opta_power_poisson; ctx=load_model_context(Path('data/quiniela.db'), 'test_opta'); cfg=load_json_config(Path('configs/models.yaml')); sc=load_json_config(Path('configs/scoring.yaml')); m=[x for x in cfg['models'] if x['model_id']=='opta_power_poisson'][0]; preds=run_opta_power_poisson(ctx,m,sc); print(sum(p.status=='ok' for p in preds), sum(p.status=='masked' for p in preds), sum(p.status=='failed' for p in preds)); print(preds[0].model_id, preds[0].team_a, preds[0].team_b, preds[0].selected_score, preds[0].warnings[:3])"
```

## Limitaciones

- No es el Opta Supercomputer completo.
- La cobertura publica de ratings/rankings es parcial.
- Las probabilidades de avance publicadas por Opta no siempre se pueden convertir de forma limpia a marcadores por partido.
- La evaluacion honesta sera diaria durante el Mundial, comparando sus picks congelados contra resultados reales.
