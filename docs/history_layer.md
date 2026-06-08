# Capa Historica para Modelos

## Objetivo

Crear una capa historica separada del estado operativo del Mundial 2026. Esta capa alimentara modelos como Elo, Poisson/Dixon-Coles, forma reciente, fuerza ofensiva/defensiva y backtesting.

## Decision Inicial

Usar como base:

```text
martj42/international_results
```

Motivo:

- Es amplio: partidos internacionales masculinos desde 1872.
- Tiene columnas directamente utiles para modelos de goles.
- Tiene licencia CC0.
- Tiene mirror en GitHub, por lo que no requiere credenciales Kaggle.

## Tabla Comparativa

| Fuente | Rol recomendado | Ventajas | Riesgos / limites | Prioridad |
|---|---|---|---|---|
| martj42 / Kaggle International Results | Historico principal de entrenamiento | 49k+ partidos internacionales, scores, torneo, ciudad, pais, neutralidad, licencia CC0 | No trae xG, alineaciones ni odds; hay que limpiar nombres/equivalencias | 1 |
| Hicruben WC2026 model | Referencia reciente y metodologica | Elo + Dixon-Coles + Monte Carlo; backtest walk-forward; datos recientes 2023-2026 | Dataset mas pequeno; repo JS; no debe reemplazar historico amplio | 2 |
| jfjelstul/worldcup | Enriquecimiento mundialista | 27 datasets, granularidad Mundial, jugadores, arbitros, estadios, eventos; buen contexto de torneos | Solo Mundiales; licencia CC-BY-SA; no suficiente para entrenar fuerza global | 3 |
| Currybon30/fifa_wc_2026_datacamp | Referencia metodologica | Features tipo Elo, ataque/defensa, forma, Monte Carlo, notebooks | AGPL-3.0; proyecto chico; no usar codigo directo sin revisar licencia | 4 |
| EhteshamBahoo/Fifa-WorldCup-Data-Analysis-1930-2026 | Referencia pedagogica | Scraping, limpieza, Random Forest para goles, simulacion | Datos menos robustos; no se confirmo Poisson; licencia poco clara | 5 |
| API-Football | Enriquecimiento pagado/API | Fixtures, standings, eventos, lineups, stats, odds; free tier 100 req/dia | API key, cuotas, costo; cobertura historica depende plan | 6 |
| The Odds API | Calibracion mercado | H2H, totals, spreads, historical odds desde 2020, muchas casas | No es dataset de goles; historicos son de pago/creditos; coverage selecciones debe validarse | 7 |

## Fuentes Verificadas

- martj42/international_results: https://github.com/martj42/international_results
- Kaggle martj42: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- Hicruben/world-cup-2026-prediction-model: https://github.com/Hicruben/world-cup-2026-prediction-model
- jfjelstul/worldcup: https://github.com/jfjelstul/worldcup
- Currybon30/fifa_wc_2026_datacamp: https://github.com/Currybon30/fifa_wc_2026_datacamp
- EhteshamBahoo/Fifa-WorldCup-Data-Analysis-1930-2026: https://github.com/EhteshamBahoo/Fifa-WorldCup-Data-Analysis-1930-2026
- API-Football: https://www.api-football.com/
- The Odds API: https://the-odds-api.com/

## Contrato Esperado

Tabla historica canonica implementada:

```text
canonical_historical_matches
```

Columnas principales:

```text
historical_match_id
source_id
source_match_key
match_date
team_a_name
team_b_name
team_a_canonical_id
team_b_canonical_id
home_score
away_score
tournament
city
country
neutral
result_1x2
goal_difference
total_goals
is_world_cup
is_qualifier
is_friendly
importance_weight
recency_weight
history_run_id
created_at_utc
```

Vistas implementadas:

```text
v_latest_history_run
v_canonical_historical_matches
v_model_training_matches
v_team_rating_inputs
```

## Implementacion

Archivos creados:

```text
configs/history_sources.json
src/quiniela/history/pipeline.py
scripts/build_history.py
```

Comando:

```powershell
python scripts\build_history.py
```

Con el Python completo del entorno:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\build_history.py
```

## Primera Corrida

La primera ingesta historica exitosa fue:

```text
history_run_id: history_20260605T065321Z_1846474b
as_of_utc: 2026-06-05T06:53:21Z
sources_checked: 1
files_downloaded: 4
matches_imported: 49318
```

Cobertura inicial:

```text
total training matches: 49318
matches with at least one Mundial 2026 team mapped: 25890
matches with both teams mapped: 7517
world cup matches: 1024
qualifiers: 15928
friendlies: 18312
```

## Estrategia Recomendada

1. Descargar `martj42/international_results` desde GitHub raw.
2. Canonicalizar nombres de equipos contra `v_canonical_teams`.
3. Construir `canonical_historical_matches`.
4. Derivar pesos:
   - mayor peso a partidos recientes;
   - mayor peso a Mundial, eliminatorias continentales y torneos oficiales;
   - menor peso a amistosos.
5. Usar Hicruben para comparar Elo/backtest y validar enfoque.
6. Usar Fjelstul para enriquecer contexto mundialista.
7. Integrar The Odds API/API-Football solo cuando tengamos API keys o cuando el modelo base ya funcione.

## No Contradiccion

Esta capa no reemplaza la canonicalizacion ni el estado vivo del torneo. Se agrega antes de modelos y alimenta features/modelos.
