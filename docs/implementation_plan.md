# Plan de Implementacion

## Fase 0 - Base del Proyecto

Objetivo: crear una estructura limpia, reproducible y facil de extender.

Entregables:

- `pyproject.toml` con dependencias base.
- `configs/project.yaml` con rutas y parametros generales.
- `configs/scoring.yaml` con reglas de quiniela.
- `configs/models.yaml` con modelos activos.
- Estructura `src/quiniela/` y `notebooks/`.

Criterio de cierre:

- El proyecto puede importar `quiniela`.
- Existe un comando o script minimo para validar configuracion.

## Fase 1 - Datos Base

Objetivo: cargar fixtures, equipos y resultados historicos.

Entregables:

- Ingesta de fixtures 2026.
- Dataset historico de partidos internacionales.
- Normalizador de nombres de equipos.
- Tabla canonica de partidos.

Criterio de cierre:

- Todos los partidos tienen `match_id`, `date`, `team_a`, `team_b`, `neutral`, `competition`, `stage`.
- Se puede generar un snapshot versionado en `data/processed/`.

## Fase 1B - Operacion Diaria y Estado Vivo

Objetivo: permitir corridas repetidas antes de cada bloque de partidos.

Entregables:

- `run_id` y `as_of_utc` para cada corrida.
- Ingesta de resultados del Mundial 2026 desde fuentes configurables.
- Canonicalizacion de equipos, partidos y horarios.
- Reconciliacion de resultados si hay discrepancias entre fuentes.
- Reconstruccion de standings y estado del torneo.
- Actualizacion de ratings y forma usando partidos ya completados.
- Export de quiniela diaria.

Criterio de cierre:

- Una corrida posterior a un partido incorpora el resultado en ratings, standings y features.
- Una corrida anterior a un partido no puede usar informacion posterior a su `as_of_utc`.
- Existe un comando diario unico para ejecutar descarga, canon, estado y dashboard.

## Fase 2 - Modelo Base de Goles

Objetivo: implementar el primer modelo funcional de matriz de marcadores.

Modelo inicial recomendado:

- Elo + Poisson + ajuste Dixon-Coles.

Entregables:

- Notebook `notebooks/01_model_elo_dixon_coles.ipynb`.
- Modulo reutilizable en `src/quiniela/models/elo_dixon_coles.py`.
- Artefacto de predicciones en `data/predictions/`.

Criterio de cierre:

- El modelo produce `score_matrix`, `p_team_a_win`, `p_draw`, `p_team_b_win`, `expected_goals_a`, `expected_goals_b`.

## Fase 3 - Calibracion con Mercado

Objetivo: incorporar probabilidades externas sin destruir la matriz de goles.

Fuentes candidatas:

- Polymarket.
- Kalshi.
- Bookmakers con mercados 1X2 y over/under.

Entregables:

- Ingesta de precios externos en `data/external/`.
- Normalizacion de probabilidades.
- Ajuste de buckets 1X2 sobre matriz de marcadores.
- Notebook `notebooks/02_model_market_calibrated_poisson.ipynb`.

Criterio de cierre:

- Una prediccion calibrada conserva probabilidades por marcador y respeta los totales 1X2 calibrados.

## Fase 4 - Modelos Alternativos

Objetivo: agregar modelos independientes que puedan competir o integrarse al ensemble.

Modelos candidatos:

- Poisson con features estadisticas.
- ML 1X2 con XGBoost/LightGBM.
- Modelo basado en xG o shots.
- Modelo de mercado puro.
- Modelo de consenso simple.

Criterio de cierre:

- Cada modelo publica el mismo contrato de salida.
- El ensemble puede ignorar modelos ausentes o desactivados.

## Fase 5 - Ensemble y Selector de Quiniela

Objetivo: combinar N modelos y elegir marcadores por puntos esperados.

Entregables:

- `src/quiniela/ensemble/` para combinar matrices/probabilidades.
- `src/quiniela/scoring/` para reglas de quiniela.
- Notebook `notebooks/10_ensemble_and_selection.ipynb`.

Criterio de cierre:

- Dada una lista de predicciones, el sistema elige el marcador con mayor valor esperado.

## Fase 6 - Backtesting y Metricas

Objetivo: medir si los modelos mejoran bajo reglas reales.

Entregables:

- Backtest walk-forward.
- Reporte de metricas por modelo.
- Comparacion por etapa, confederacion, favorito/no favorito.

Criterio de cierre:

- Se reportan exact score accuracy, margin/draw accuracy, 1X2 accuracy, Brier, log-loss y puntos esperados.

## Fase 7 - Reportes Operativos

Objetivo: producir pronosticos utilizables antes del torneo y actualizables durante el torneo.

Entregables:

- CSV/JSON final de predicciones.
- Reporte Markdown o notebook.
- Opcional: dashboard Streamlit.

Criterio de cierre:

- Se puede regenerar el reporte con datos actualizados sin editar codigo manualmente.

## Fase 8 - Ola 1: Mejoras de Senal (Quick Wins)

Objetivo: mejorar la calidad de la senal de los modelos existentes sin agregar arquitectura nueva.

Diagnostico base: backtest 2014/2018/2022 muestra eficiencia 25-26% vs maximo posible.
El mejor modelo (bradley_terry_davidson) alcanza Brier=0.610 frente a aleatorio=0.667.

Entregables:

- `recency_weight` aplicado al delta de Elo en `elo_poisson.py` — partidos viejos
  dejan de distorsionar ratings actuales.
- Parametro `min_importance_for_rating` para excluir amistosos del ajuste de Elo.
- Script `scripts/tune_models.py` para busqueda de hiperparametros.

Criterio de cierre:

- El tuning encuentra configuracion con eficiencia > 28% sobre backtest 2014/2018/2022.
- Los parametros optimos se documentan y aplican en `configs/models.yaml`.

Estado: IMPLEMENTADO (2026-06-05). Ver notas 017 y 018.

Tuning exhaustivo en ejecución (GPU RTX 3050): elo_poisson 1944 trials,
elo_dixon_coles 1944 trials, draw_specialist 1440 trials. CPU después para
bradley_terry_davidson y attack_defense_poisson.

## Fase 9 - Ola 2: Modelos Nuevos

Objetivo: agregar modelos genuinamente independientes para diversidad real en el ensemble.

Modelos candidatos:

- Dixon-Coles MLE real: ajustar alpha_i, beta_i, rho por maxima verosimilitud sobre
  historico ponderado por recencia (scipy.optimize.minimize).
- XGBoost 1X2 con features: diferencia de Elo, forma reciente, goles for/against
  ponderados, indicador anfitrion, etapa del torneo.
- draw_specialist mejorado: regresion logistica sobre features explicativas en lugar
  del boost heuristico actual.

Criterio de cierre:

- Al menos un modelo nuevo con Brier < 0.57 sobre backtest 2014/2018/2022.
- Correlacion < 0.80 con elo_poisson en predicciones 1X2.

## Fase 10 - Calibracion con Mercado

Objetivo: activar `market_calibrated_poisson` con fuente de probabilidades externas.

Fuentes candidatas:

- Polymarket (API publica durante el torneo 2026).
- Pinnacle/OddsPortal para odds historicas 2014-2022 (validacion retrospectiva).

Entregables:

- Ingesta de odds en `data/external/`.
- Ajuste 1X2 sobre matriz Poisson usando `adjust_score_matrix_to_1x2` (ya implementada).
- Activacion de `market_calibrated_poisson` en `configs/models.yaml`.

Criterio de cierre:

- Si hay odds disponibles, el modelo las incorpora antes del kickoff.
- Si no hay odds, el modelo usa el fallback Elo-Poisson sin error.

## Fase 11 - Ensemble Formal

Objetivo: combinar matrices de marcadores de N modelos con pesos derivados del backtest.

Entregables:

- `src/quiniela/ensemble/` con combinacion ponderada usando `blend_score_matrices`
  (ya implementada en common.py).
- Pesos derivados del Brier inverso sobre backtest reciente (hold-out 2022).
- El selector de quiniela opera sobre la matriz combinada del ensemble.

Criterio de cierre:

- El ensemble supera al mejor modelo individual en backtest hold-out sobre 2022.
- La eficiencia del ensemble es >= 29% sobre los 3 Mundiales.
