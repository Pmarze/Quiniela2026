# 007 - Capa historica para modelos

## Conocimiento

Se definio la capa historica para entrenar modelos predictivos. Esta capa es distinta del estado vivo del Mundial 2026.

## Fuente principal recomendada

```text
martj42/international_results
```

Razones:

- Historico amplio de selecciones masculinas desde 1872.
- 49k+ partidos.
- Incluye score, torneo, ciudad, pais y neutralidad.
- Licencia CC0.
- Disponible en GitHub raw, sin credenciales Kaggle.

## Fuentes secundarias

```text
Hicruben/world-cup-2026-prediction-model
jfjelstul/worldcup
```

Hicruben sirve para validar enfoque Elo + Dixon-Coles + Monte Carlo y backtest reciente.

Fjelstul sirve para enriquecer contexto especifico de Mundiales.

## Fuentes no prioritarias para datos base

```text
Currybon30/fifa_wc_2026_datacamp
EhteshamBahoo/Fifa-WorldCup-Data-Analysis-1930-2026
API-Football
The Odds API
```

Currybon30 y EhteshamBahoo son referencias metodologicas, no fuentes canonicas.

API-Football y The Odds API pueden servir despues como enriquecimiento o calibracion de mercado, pero no como base inicial del modelo de goles.

## Configuracion creada

```text
configs/history_sources.json
src/quiniela/history/
docs/history_layer.md
```

## Estado

Activo. No contradice conocimientos anteriores; agrega la capa que alimentara modelos y backtesting.

