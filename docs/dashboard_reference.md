# Dashboard Unificado — Referencia técnica

## Archivos clave

| Archivo | Rol |
|---|---|
| `src/quiniela/ui/dashboard.py` | Generador Python — único archivo a editar para lógica de datos |
| `src/quiniela/ui/dashboard_template.html` | Template HTML/CSS/JS — editar para cambios de interfaz o comportamiento visual |
| `data/ui/prediction_overrides.json` | Fuente de predicciones y picks de quiniela |
| `data/quiniela.db` | Base de datos SQLite con estado del torneo y backtest |
| `docs/index.html` | Salida publica del dashboard; incluye `DATA.friends` si existe el JSON |
| `outputs/dashboard/index.html` | Salida generada (no editar directamente) |

## Cómo regenerar

```bash
# Desde la raiz del proyecto, con el entorno quiniela2026
python scripts/generate_dashboard.py \
  --db data/quiniela.db \
  --predictions data/ui/prediction_overrides.json \
  --output docs/index.html

# Version sin amigos solo si se pide explicitamente
python scripts/generate_dashboard.py \
  --exclude-friends \
  --db data/quiniela.db \
  --predictions data/ui/prediction_overrides.json \
  --output outputs/dashboard/index.html

python scripts/check_public_dashboard.py docs/index.html
```

Activar antes el entorno:

```powershell
conda activate quiniela2026
```

---

## Arquitectura del generador (`dashboard.py`)

```
generate_dashboard()               ← punto de entrada del dashboard
  ├── _load_latest_state()         ← lee v_latest_tournament_state
  ├── _load_matches()              ← lee v_latest_state_matches
  ├── _load_group_tables()         ← lee v_latest_state_group_tables
  ├── _load_prediction_overrides() ← lee prediction_overrides.json
  ├── _load_backtest_data()        ← lee v_latest_backtest_* (None si no hay)
  ├── _build_unified_payload()     ← ensambla el objeto DATA completo
  └── _render_html()               ← lee template + inyecta payload JSON
```

### Constantes (inicio del archivo)

- **`_FAMILY_BY_MODEL_ID`** — mapea `model_id` → `{family, fb}` para badges CSS
- **`_STAGE_TO_PHASE`** — mapea `stage` del DB → `phase` del JS (`"round_of_16"` → `"r16"`)

Para agregar un modelo nuevo, añadir una entrada en `_FAMILY_BY_MODEL_ID`. Las clases CSS disponibles son: `fb-ctrl fb-fgol fb-mba fb-atd fb-emp fb-1x2 fb-neu fb-sim fb-pond`.

---

## Estructura del objeto DATA (payload JS)

```js
const DATA = {
  meta: { generated_at, run_id, phase },
  access: { public_mode, private_sections, private_hash },
  kpis: { total, played, live, scheduled, locked },
  groups: [{ id: "A", teams: [{ t: "México" }] }],
  matches: [{
    id, num, home, away, date, time, venue, city,
    group, phase, status,
    result,       // "2-1" o null si no jugado
    quiniela: { model, score, top, ev, prob },
    frozen,       // true si el pick está bloqueado
    models: [{    // una entrada por modelo
      id, family, fb, top, score, xg, out, conf, ev,
      p1, px, p2, notes
    }]
  }],
  backtest: {
    run_id, years, ref_note,
    metrics: [{ model, label, fb, pts, pts_prob, max, eff, eff_prob,
                exact, exact_prob, winner, draws, is_ref }],
    predictions: [{ match, model, year, phase, result,
                    smp, pmp, spr, ppr, x1, xp, xg }]
  }
}
```

En `docs/index.html`, `friends` puede contener quinielas de amigos para que la version en linea
se vea igual que la local. El enlace/ID de Google Sheets no debe aparecer en el HTML ni en archivos
versionados; se guarda en `configs/friends_sheet.local.json` o variables de entorno.

### Campos de `prediction_overrides.json` → `matches[i]`

| JSON | → DATA |
|---|---|
| `quiniela_pick.score` | `quiniela.score` (pick Max Puntos) |
| `quiniela_pick.top_score` | `quiniela.top` (pick Más Probable) |
| `quiniela_pick.top_score_probability` | `quiniela.prob` ← ojo, NO `probability` |
| `quiniela_pick.expected_points` | `quiniela.ev` |
| `model_predictions[i].expected_goals` | `models[i].xg` — ya viene como string `"1.74-1.13"`, no reformatear |
| `model_predictions[i].p_team_a_win` | `models[i].p1` |
| `model_predictions[i].p_draw` | `models[i].px` |
| `model_predictions[i].p_team_b_win` | `models[i].p2` |

### Campos del DB → `matches[i]`

| DB | → DATA |
|---|---|
| `kickoff_local_iso` | `date = iso[:10]`, `time = iso[11:16]` |
| `stadium_city` | `city` |
| `team_a_name` / `team_b_name` | `home` / `away` |
| `home_score` / `away_score` + status `"finished"/"completed"` | `result` |

---

## Template HTML (`dashboard_template.html`)

Archivo de 1632 líneas. Estructura interna:

| Líneas aprox. | Contenido |
|---|---|
| 1–10 | `<head>`, fuentes Barlow de Google Fonts |
| 11–405 | CSS completo (variables, componentes, dark mode, hojas/graficos) |
| 407–507 | `<body>`: nav, secciones vacías, hover panel, modal, validation hover |
| 509–510 | `<script>` + token `const DATA = __DATA_JSON__;` |
| 511–1630 | JavaScript completo |

### Variables CSS principales

```css
/* Paleta (light) */
--nav: #0d1b2a     /* fondo navbar */
--teal: #087e8b    /* color primario, acentos */
--burg: #641f3e    /* cabeceras de paneles */
--gold: #c9960e    /* highlights, Mi Pick */
--cor: #c94d2f     /* errores, live */
--grn: #1e8050     /* éxito, ranking live */

/* Dark mode: sobreescribe en html[data-theme="dark"] */
```

### Funciones JS clave

| Función | Qué hace |
|---|---|
| `renderHome()` | Construye sección Inicio: countdown, KPIs, ranking, lista partidos |
| `renderQuiniela()` | Construye sección Quiniela: grupos + panel central |
| `renderValidacion()` | Construye sección Validación: rankings backtest + tabla predicciones |
| `renderGuia()` | Construye sección Guía: métricas, reglas, familias |
| `showHover(matchId, el)` | Panel flotante con modelos al hover sobre un partido |
| `openModal(matchId)` | Modal completo con tabla de modelos y campo Mi Pick |
| `setHoverSheet(sheet)` | Cambia el hover entre hoja `stats` y hoja `models` |
| `setModalSheet(sheet)` | Cambia el modal fijo entre hoja `stats` y hoja `models` |
| `buildStatsSheet(m)` | Construye la hoja analítica por partido |
| `buildScatterChart(m, models)` | Scatter xG por modelo con línea diagonal X=Y |
| `buildHeatmapChart(m, models)` | Heatmap de frecuencia de marcadores TOP |
| `buildProbabilityChart(m, models)` | Barras 1X2 con rango min/max y EV promedio |
| `buildCurrentRankData()` | Acumula puntos por modelo de partidos con `result != null` |
| `showCurrentModelHover(modelId, mode, el)` | Panel clickeable de Validación 2026: compara un modelo contra resultados reales del torneo actual |
| `calcMatchPoints(pick, result)` | Regla 5/3/1/0 de la quiniela |
| `filterPreds()` | Filtra tabla de predicciones del backtest con los multi-select |
| `toggleDark()` | Alterna light/dark y persiste en localStorage `q2026_theme` |
| `savePick(matchId, score)` | Guarda Mi Pick en localStorage `q2026_picks` |

### Hover y modal por hojas

El hover flotante y el modal fijo de cada partido tienen dos hojas:

- `stats` es la hoja por defecto. Presenta tres gráficos calculados en el navegador con los datos ya inyectados en `DATA.matches[i].models`.
- `models` conserva la vista operativa anterior: tarjetas en hover y tabla en modal con el pronóstico de cada modelo.

Los botones de flecha usan `setHoverSheet('stats'|'models')` y `setModalSheet('stats'|'models')`. La flecha derecha lleva a `models`; la izquierda vuelve a `stats`.

La hoja `stats` contiene:

- Scatter xG: usa `models[i].xg`, parseado desde strings tipo `"1.74-1.13"`. La diagonal gris indica igualdad de xG; las lineas doradas indican el promedio de xG local y visitante entre modelos. El scatter ignora valores no finitos o mayores a 6.0 como red de seguridad visual.
- Scatter xG tambien puede mostrar una estrella de referencia de amistosos recientes ponderados.
- Heatmap de marcadores: cuenta la frecuencia de `models[i].top` o `models[i].score` en una grilla 0, 1, 2, 3+ para local y visitante. Muestra porcentaje como valor principal y el conteo exacto de modelos como numero pequeno.
- Probabilidad 1X2: usa `p1`, `px`, `p2` para promedio y rango min/max por mercado. Tiene dos carriles: consenso promedio y modelo lider actual. El EV mostrado es el promedio de `models[i].ev` de los modelos cuyo `out` coincide con ese mercado.
- Amistosos recientes: muestra ultimos 5 amistosos por equipo y referencia ponderada de probabilidad de gol.

### Validacion 2026 actual

Los paneles `2026 · Max Pts.` y `2026 · Más Probable` son clickeables por modelo.

Al hacer click, `showCurrentModelHover(modelId, mode, el)` muestra solamente partidos ya jugados del torneo actual:

- si `mode === "score"`, evalua primero `mod.score`
- si `mode === "top"`, evalua primero `mod.top`
- muestra puntos, exactos, aciertos, 1X2, `P(real)`, xG y error xG partido por partido

### Dark mode

El atributo `data-theme="dark"` vive en `<html>`. Se restaura en `DOMContentLoaded` desde localStorage. Para cambiar el valor por defecto, editar tanto el atributo en `<html lang="es" data-theme="dark">` como el fallback en el init:
```js
var savedTheme = localStorage.getItem('q2026_theme') || 'dark';
```

### Regla de puntos (quiniela)

| Puntos | Condición |
|---|---|
| 5 | Marcador exacto |
| 3 | Diferencia de goles igual, o empate con distinto marcador |
| 1 | Ganador correcto (1X2) |
| 0 | Fallo total |

---

## Cómo añadir resultados reales

Cuando se juegue un partido, el generador ya los toma del DB automáticamente:
- `status` pasa a `"finished"` o `"completed"`
- `home_score` / `away_score` se rellenan
- `result` aparece en el partido → el ranking Live se activa solo

No hay que tocar el template ni el JS.

---

## Verificación rápida post-generación

```python
import json
from pathlib import Path

html  = Path("outputs/dashboard/index.html").read_text(encoding="utf-8")
idx   = html.index("const DATA = ") + len("const DATA = ")
data, _ = __import__("json").JSONDecoder().raw_decode(html, idx)

assert len(data["matches"])  == 104
assert len(data["groups"])   >  0
assert len(data["backtest"]["metrics"]) > 0
print("OK", len(html), "chars")
```
