# Quiniela Mundial 2026 - Contexto General

## Objetivo

Construir un proyecto modular para estimar pronosticos de quiniela del Mundial 2026 usando simulacion Monte Carlo, modelos de goles y senales externas de mercado. El foco no es solo acertar ganador, sino maximizar puntos esperados segun la regla de quiniela.

El proyecto debe operar como un sistema vivo. Antes del torneo generara predicciones con datos historicos y mercados pre-torneo. Durante el Mundial, cada corrida diaria debe incorporar resultados ya ocurridos, standings actualizados, ratings recalculados y nuevos precios de mercado antes de pronosticar los siguientes partidos.

Prioridad de acierto:

1. Resultado exacto.
2. Empate o diferencia de goles.
3. Ganador del partido.

## Idea Central

El sistema debe separar tres decisiones:

1. Estimar una distribucion de marcadores por partido.
2. Calibrar esa distribucion con senales externas, como odds, Polymarket, Kalshi o rankings.
3. Elegir el marcador recomendado que maximiza puntos esperados bajo las reglas de la quiniela.

Esto evita que el proyecto dependa de un unico modelo. Cada modelo puede vivir en su propio notebook o modulo y producir un artefacto estandar. El siguiente paso del pipeline consume esos artefactos, no detalles internos del modelo.

## Principios de Diseno

- Modularidad: cada modelo se puede agregar, quitar o reemplazar sin romper el resto del pipeline.
- Contratos claros: cada modelo debe publicar predicciones en el mismo esquema.
- Reproducibilidad: cada corrida debe guardar configuracion, version de datos, fecha y metricas.
- Cortes temporales: cada pronostico debe declarar `as_of_utc` para saber que informacion estaba disponible.
- Evaluacion honesta: walk-forward cuando sea posible, evitando usar informacion posterior al kickoff.
- Separacion de notebooks y produccion: notebooks para exploracion/modelado; codigo en `src/` para logica reutilizable.
- Calibracion explicita: mercados externos son una senal, no una verdad absoluta.
- Memoria incremental: decisiones y lecciones se registran en `docs/knowledge/` con numeracion progresiva.

## Fuentes Investigadas

Base recomendada:

- Hicruben/world-cup-2026-prediction-model: Elo + Dixon-Coles + Monte Carlo para Mundial 2026.
- opisthokonta/goalmodel: modelos de goles, scoreline probabilities y scoring rules.
- pespila: API Python tipo scikit-learn para Dixon-Coles, Elo y distribuciones de goles.
- ChristianLG2/WorldCup2026-Match-Predictor: XGBoost con ranking FIFA, forma reciente e historicos.
- OddsMap: ejemplo conceptual de fusion de historicos, Polymarket, Kalshi y Monte Carlo.

Fuentes de datos candidatas:

- openfootball/worldcup.json para fixtures.
- StatsBomb open-data para xG/eventos cuando aplique.
- FIFA rankings / World Football Elo para fuerza de equipos.
- Polymarket/Kalshi/bookmakers para calibracion externa de probabilidades.

## Estructura Propuesta

```text
Quiniela2026/
  PROJECT_CONTEXT.md
  README.md
  pyproject.toml
  configs/
    project.yaml
    scoring.yaml
    models.yaml
  data/
    raw/
    interim/
    processed/
    predictions/
    external/
    state/
  docs/
    knowledge/
    implementation_plan.md
    architecture.md
    daily_update_workflow.md
    canonicalization.md
    history_layer.md
    tournament_state.md
    ui_dashboard.md
    model_contract.md
    data_sources.md
    evaluation_metrics.md
    notebook_workflow.md
  notebooks/
    00_data_audit.ipynb
    01_model_elo_dixon_coles.ipynb
    02_model_market_calibrated_poisson.ipynb
    03_model_xg_or_stats_features.ipynb
    04_model_ml_1x2.ipynb
    10_ensemble_and_selection.ipynb
    20_backtest_report.ipynb
  src/
    quiniela/
      __init__.py
      data/
      features/
      models/
      calibration/
      ensemble/
      scoring/
      simulation/
      evaluation/
      reporting/
      orchestration/
      canonical/
      history/
      state/
      ui/
  tests/
```

## Resultado Esperado del Sistema

Para cada partido:

- Probabilidad de cada marcador razonable, por ejemplo 0-0 a 8-8.
- Probabilidad 1X2: gana equipo A, empate, gana equipo B.
- Probabilidad de diferencia de goles.
- Marcador recomendado para quiniela.
- Puntos esperados del marcador recomendado.
- Explicacion corta de las senales que mas influyeron.
- `run_id` y `as_of_utc` para reproducir el pronostico.
- Dashboard local regenerable para revisar grupos, partidos, resultados y pronosticos.

## Estado Implementado

Ya existe una primera base operativa:

- Descarga de fixtures/resultados del Mundial 2026 a SQLite.
- Canonicalizacion de equipos, partidos y horarios.
- Estado vivo del torneo con vistas para partidos, grupos y forma.
- Dashboard local HTML.
- Capa historica con `martj42/international_results` para entrenamiento.
- Modelo neural base `neural_scoreline_mlp` activo como candidato diario.
- Segundo modelo neural experimental `neural_hybrid_v2` implementado pero inactivo hasta entrenar artefactos.
- Ponderador `weighted_ensemble` activo como propuesta principal de quiniela.

La capa historica se ejecuta con:

```powershell
python scripts\build_history.py
```

La primera corrida exitosa importo 49,318 partidos historicos y dejo disponibles:

```text
v_latest_history_run
v_model_training_matches
v_team_rating_inputs
```
