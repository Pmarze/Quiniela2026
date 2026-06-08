# 009 - Opciones de modelizacion

## Conocimiento

Se investigaron repositorios y paquetes utiles para ampliar las opciones de modelos de prediccion de la quiniela. La decision vigente es construir modelos modulares que publiquen el contrato definido en:

```text
docs/model_contract.md
```

Cada modelo debe producir matriz de marcadores, probabilidades 1X2 y metadata de corrida. El ensemble o selector posterior debe poder ignorar modelos ausentes o desactivados.

## Repos y Paquetes de Referencia

| Fuente | Aporte | Uso recomendado |
|---|---|---|
| `Hicruben/world-cup-2026-prediction-model` | Elo + Dixon-Coles + Monte Carlo para Mundial 2026 | Benchmark directo para comparar nuestro primer modelo |
| `martineastwood/penaltyblog` | Poisson, bivariate Poisson, Dixon-Coles, modelos bayesianos, ratings y odds | Candidato Python para acelerar implementacion seria |
| `opisthokonta/goalmodel` | R: Poisson, Negative Binomial, Conway-Maxwell-Poisson y Dixon-Coles | Referencia metodologica para variantes de goles |
| `pespila` | Python: Dixon-Coles, Bradley-Terry, Elo y distribuciones de goles | Muy alineado con Python 3.11; revisar estabilidad porque esta en alpha |
| `leoegidi/footbayes` | R: modelos bayesianos/MLE, bivariate Poisson, Skellam, zero-inflated | Referencia avanzada para fase bayesiana posterior |
| `ChristianLG2/WorldCup2026-Match-Predictor` | XGBoost 1X2 con rankings FIFA, forma, H2H, host y features de squad | Inspiracion para modelo ML de ganador/calibrador |
| `Currybon30/fifa_wc_2026_datacamp` | Elo, ataque/defensa, Poisson/ML y Monte Carlo | Inspiracion conceptual; no copiar codigo por licencia AGPL-3.0 |
| `EhteshamBahoo/Fifa-WorldCup-Data-Analysis-1930-2026` | Random Forest para goles y simulacion de Mundial | Referencia pedagogica, no fuente principal productiva |
| `ML-KULeuven/soccer_xg` | Modelos xG con eventos StatsBomb/Wyscout/Opta | Util solo si despues tenemos datos de eventos/xG por seleccion |
| `tanjt107/football-prediction` | Rating multiplicativo ataque/defensa para goles esperados | Buena idea simple para `attack_defense_poisson` |

## Lineas de Modelizacion Propuestas

| Modelo | Prioridad | Rol |
|---|---:|---|
| `baseline_poisson` | 1 | Control minimo con promedios historicos de goles |
| `elo_poisson` | 2 | Primer modelo serio de goles esperados basado en fuerza relativa |
| `elo_dixon_coles` | 3 | Modelo principal inicial para marcador exacto, especialmente 0-0, 1-0, 1-1 |
| `attack_defense_poisson` | 4 | Estimar goles por fuerza ofensiva/defensiva, torneo y recencia |
| `negative_binomial_goals` | 5 | Alternativa si Poisson subestima la variabilidad de goles |
| `bradley_terry_davidson` | 6 | Calibrador 1X2 con manejo explicito de empates |
| `xgboost_1x2` o `catboost_1x2` | 7 | Especialista en ganador con rankings, forma, H2H, host y features contextuales |
| `market_calibrated_poisson` | 8 | Ajustar la matriz de marcadores a probabilidades de mercado/odds |
| `draw_specialist` | 9 | Capa enfocada en mejorar empates, importantes para quiniela |
| `tournament_state_adjuster` | 10 | Ajuste durante el Mundial por grupo, necesidad de puntos, descanso y bracket |
| `ensemble_selector` | 11 | Combina modelos activos y elige marcador por puntos esperados |

## Orden Recomendado de Implementacion

1. Crear configuraciones base:

```text
configs/scoring.yaml
configs/models.yaml
```

2. Crear estructura de modelos:

```text
src/quiniela/models/
src/quiniela/scoring/
scripts/run_model.py
notebooks/01_model_elo_dixon_coles.ipynb
```

3. Implementar primero:

```text
baseline_poisson
elo_poisson
```

4. Extender despues a:

```text
elo_dixon_coles
```

5. Cuando ya exista matriz de marcadores y primer backtest, agregar:

```text
market_calibrated_poisson
ensemble_selector
```

## Criterio de Evaluacion

Los modelos deben evaluarse con las metricas ya definidas:

```text
exact score accuracy
margin or draw accuracy
winner accuracy
brier 1X2
log-loss 1X2
scoreline log-loss
mean quiniela points
```

La prioridad de negocio sigue siendo:

```text
1. Resultado exacto
2. Empate o diferencia de goles
3. Ganador del partido
```

## Cautelas

- No copiar codigo de repos con licencias restrictivas sin revisar implicaciones.
- `Currybon30/fifa_wc_2026_datacamp` tiene licencia AGPL-3.0; usar solo como inspiracion conceptual salvo aprobacion explicita.
- Modelos R como `goalmodel` y `footbayes` sirven como referencia, pero el proyecto principal sigue siendo Python.
- Modelos xG requieren datos de eventos; no deben bloquear el primer modelo porque hoy la base historica principal tiene resultados, no tiros/eventos.
- El mercado externo debe calibrar probabilidades, no reemplazar directamente la matriz de goles.

## Estado

Activo. No contradice conocimientos anteriores; amplía la nota 007 y prepara la fase de modelos.
