# Fuentes de Datos

## Fixtures y Calendario

### openfootball/worldcup.json

Uso:

- Fixtures del Mundial 2026.
- Equipos, grupos, fechas, sedes.

Ventajas:

- JSON publico.
- Sin API key.
- Licencia CC0.

Riesgos:

- Puede requerir actualizacion manual si cambia algun dato oficial.

URL:

- https://github.com/openfootball/worldcup.json

## Resultados y Standings Durante el Mundial

Estas fuentes son criticas para la operacion diaria. La arquitectura debe tratarlas como adaptadores intercambiables, porque disponibilidad, limites y calidad pueden cambiar durante el torneo.

### FIFA.com

Uso:

- Referencia oficial de calendario, fixtures y resultados.
- Validacion final cuando haya discrepancias.

Notas:

- Puede no ser la API mas comoda para automatizar, pero debe ser la referencia oficial de reconciliacion.

URL:

- https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums

### worldcup26.ir / rezarahiminia/worldcup2026

Uso:

- API REST gratuita, sin API key segun su documentacion.
- Match schedule, live results, group standings, teams y stadiums.

Ventajas:

- Open-source.
- Facil de consultar desde scripts.
- Buena candidata para fuente primaria operativa si responde bien durante el torneo.

Riesgos:

- Proyecto no oficial.
- Debe validarse contra una fuente secundaria.

URLs:

- https://worldcup26.ir/?lang=en
- https://github.com/rezarahiminia/worldcup2026

### WC2026 API

Uso:

- Real-time scores, fixtures, standings y stadium data.
- API key requerida.

Notas:

- Tiene limite de requests por plan.
- Candidata como fuente secundaria o fallback.

URL:

- https://www.wc2026api.com/

### Zafronix World Cup API

Uso:

- API historica y live.
- Webhooks y SSE para score updates/resultados.
- Bracket y standings con tiebreakers.

Notas:

- Free tier con API key.
- Reporta retraso best-effort de unos minutos durante partidos.

URL:

- https://api.zafronix.com/

### TheStatsAPI

Uso:

- Fixtures, results, match events, standings, xG, lineups, player stats y odds.
- Muy util si decidimos pagar una fuente mas completa.

Notas:

- Servicio pago con trial.
- Aporta odds y xG, que pueden mejorar calibracion de goles.

URL:

- https://www.thestatsapi.com/world-cup

## Resultados Historicos

### Hicruben/world-cup-2026-prediction-model

Uso:

- Dataset reciente 2023-2026.
- Elo calibrado para equipos mundialistas.
- Backtest de modelo base.

URL:

- https://github.com/Hicruben/world-cup-2026-prediction-model

### martj42/international_results

Uso:

- Historico principal para modelos de goles, Elo, forma reciente y backtesting.
- Partidos internacionales masculinos desde 1872.

Ventajas:

- Licencia CC0.
- Disponible en GitHub raw y Kaggle.
- Incluye score, torneo, sede y neutralidad.

URLs:

- https://github.com/martj42/international_results
- https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

### Fjelstul World Cup Database

Uso:

- Historial estructurado de Mundiales.
- Validacion contra torneos pasados.

URL:

- https://github.com/jfjelstul/worldcup

### API-Football / API-Sports

Uso:

- Enriquecimiento con fixtures, standings, eventos, lineups, estadisticas y odds.
- Fuente automatizable si se acepta usar API key.

Notas:

- Tiene free tier reportado de 100 requests/dia.
- No debe ser la base inicial de goles porque depende de cuota/API key y la profundidad historica puede variar por plan.

URLs:

- https://www.api-football.com/
- https://www.api-football.com/documentation-v3

## Rankings y Ratings

Fuentes candidatas:

- FIFA rankings.
- World Football Elo.
- Ratings propios derivados de resultados.
- Opta Power Rankings publicos.

Uso:

- Prior de fuerza por equipo.
- Feature de diferencia de fuerza.
- Ajuste para favoritos claros.

### FIFA ranking historico

Uso:

- Senal oficial externa para fuerza relativa de selecciones.
- Feature backtesteable si se usa el ranking vigente antes de cada partido.
- Puede mejorar el Monte Carlo limpio sin depender de Opta.

Estado:

- No ingerido todavia.
- Candidato para una segunda capa de mejora de `bayesian_monte_carlo_scoreline`.

URL:

- https://github.com/Dato-Futbol/fifa-ranking

### World Football Elo historico

Uso:

- Rating externo no Opta para comparar o complementar el Elo interno.
- Fuente candidata si se consigue snapshot historico por fecha.

Estado:

- No ingerido todavia.
- No es necesario para la v0.1 porque el proyecto ya calcula Elo interno walk-forward.

URLs:

- https://www.eloratings.net/
- https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings

### Opta Analyst / Stats Perform

Uso:

- Senal externa de fuerza para 2026.
- Priors de calidad de equipo via Opta Power Ratings/Rankings publicados.
- Probabilidades publicas de avance/campeonato cuando Opta las publica.
- Referencia metodologica: Opta Supercomputer simula el Mundial 25,000 veces.

Estado actual:

- Snapshot local parcial: `data/external/opta/opta_power_ratings_20260607.json`.
- Modelo activo: `opta_power_poisson`.
- Backtest limpio: excluido por defecto porque la fuente publica es de 2026 y no tenemos ratings historicos Opta por fecha.

URLs:

- https://theanalyst.com/articles/world-cup-groups-2026-easiest-hardest-opta-power-rankings
- https://theanalyst.com/articles/world-cup-2026-group-a-predictions-preview
- https://theanalyst.com/articles/who-will-win-2026-fifa-world-cup-predictions-opta-supercomputer
- https://www.statsperform.com/products/opta-data/
- https://www.statsperform.com/products/opta-vision/

## Mercados Externos

### Polymarket

Uso:

- Probabilidades tipo yes/no.
- Mercados de ganador, avanzar de ronda, grupo o partido si estan disponibles.
- Ancla externa para calibracion.

Notas:

- El precio de una accion yes se puede interpretar como probabilidad aproximada.
- La liquidez importa: un mercado iliquido puede ser ruidoso.
- Para quiniela, es mas util si se consiguen mercados de partido o mercados cercanos a 1X2.

URL:

- https://polymarket.com/predictions/world-cup

### Kalshi

Uso:

- Senal externa comparable a Polymarket.
- Posible validacion cruzada de sesgos.

### The Odds API

Uso:

- Odds prepartido y, con planes adecuados, historicos de cuotas.
- Calibracion de probabilidades 1X2, totales y spreads contra mercado.

Notas:

- No reemplaza un dataset historico de goles.
- Los historicos y eventos historicos tienen restricciones de plan/creditos.

URLs:

- https://the-odds-api.com/
- https://the-odds-api.com/liveapi/guides/v4/

## Datos de Rendimiento

### StatsBomb open-data

Uso:

- xG y eventos en competiciones disponibles.
- Entrenar o validar features de tiros.
- Para el Monte Carlo limpio: calibrar goles vs xG en Mundiales 2018/2022, siempre evitando usar xG de un torneo futuro en backtest.

URL:

- https://github.com/statsbomb/open-data
- https://blogarchive.statsbomb.com/news/statsbomb-release-free-fifa-world-cup-data/
- https://blogarchive.statsbomb.com/news/statsbomb-release-free-2022-world-cup-data/

### soccer_xg

Uso:

- Paquete para entrenar modelos xG.
- Puede servir si decidimos crear un modelo propio de xG.

URL:

- https://github.com/ML-KULeuven/soccer_xg

## Politica de Datos

Cada fuente debe registrar:

```text
source_name
source_url
downloaded_at
license
raw_file_path
processed_file_path
notes
```
