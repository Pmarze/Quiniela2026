# Handoff de Diseno Web - Dashboards Quiniela2026

Este archivo resume todo lo necesario para replicar y mejorar las dos paginas web locales del proyecto:

- Dashboard de seguimiento diario del Mundial 2026.
- Dashboard historico de validacion/backtesting.

La intencion es que un agente de diseno pueda leer solo este documento y entender la logica de consulta, estructura visual, contratos de datos, decisiones de estilo y aprendizajes acumulados.

## Contexto del proyecto

El proyecto estima una quiniela mundialista. La prioridad de puntaje es:

1. Acertar marcador exacto.
2. Acertar empate o diferencia de goles.
3. Acertar ganador del partido.

La regla de puntos actual vive en `configs/scoring.yaml`:

```json
{
  "exact_score": 5,
  "same_margin_or_draw": 3,
  "winner": 1
}
```

Los modelos publican predicciones con un contrato comun. La interfaz no debe depender de detalles internos de cada modelo; solo debe leer la salida publicada:

- `model_id`
- `family`
- `score`: marcador que maximiza puntos esperados de quiniela.
- `top_score`: marcador mas probable de la matriz.
- `expected_goals`
- `outcome`
- `confidence`
- `expected_points`
- `notes`

El modelo por defecto para la quiniela diaria es `weighted_points_ensemble`, pero el dashboard debe mostrar todos los modelos disponibles para permitir comparacion.

## Archivos principales

Dashboard diario:

```text
scripts/generate_dashboard.py
src/quiniela/ui/dashboard.py
outputs/dashboard/index.html
data/ui/prediction_overrides.json
```

Dashboard historico:

```text
scripts/generate_validation_dashboard.py
src/quiniela/backtest/dashboard.py
outputs/validation_dashboard/index.html
```

Backtest y ponderadores:

```text
scripts/run_backtest.py
src/quiniela/backtest/runner.py
src/quiniela/ensemble/weighted.py
configs/backtest.yaml
configs/models.yaml
```

Runtime esperado:

```text
Conda env: quiniela2026
Python: python
Proyecto: D:\Quiniela2026
```

## Esquema de carpetas y almacenamiento

El proyecto esta pensado para separar codigo, configuracion, datos versionados por corrida y salidas visuales. Para una futura unificacion de tableros, esta separacion es importante: la UI deberia consumir artefactos ya preparados, no recalcular modelos ni modificar datos fuente.

Estructura principal:

```text
D:\Quiniela2026
+-- configs\
|   +-- models.yaml
|   +-- backtest.yaml
|   +-- scoring.yaml
|   +-- neural_scoreline.yaml
|   +-- neural_hybrid_v2.yaml
+-- data\
|   +-- quiniela.db
|   +-- predictions\
|   +-- backtests\
|   +-- models\
|   +-- ui\
|   +-- raw\
|   +-- state\
+-- docs\
|   +-- knowledge\
|   +-- ui_dashboard.md
|   +-- backtesting_validation.md
|   +-- ui_design_handoff_claude.md
+-- outputs\
|   +-- dashboard\
|   |   +-- index.html
|   +-- validation_dashboard\
|       +-- index.html
+-- scripts\
+-- src\
|   +-- quiniela\
|       +-- backtest\
|       +-- data\
|       +-- ensemble\
|       +-- features\
|       +-- history\
|       +-- models\
|       +-- scoring\
|       +-- state\
|       +-- storage\
|       +-- training\
|       +-- ui\
+-- notebooks\
```

### Configuracion

`configs/` define comportamiento, modelos activos y reglas de puntaje.

Archivos clave:

- `configs/models.yaml`: lista de modelos, activacion, versiones, parametros y ponderadores.
- `configs/backtest.yaml`: anos incluidos, modelos de referencia, estrategia de corte temporal.
- `configs/scoring.yaml`: regla de puntos 5/3/1.
- `configs/neural_scoreline.yaml`: entrenamiento/tuning del modelo neural base.
- `configs/neural_hybrid_v2.yaml`: entrenamiento/tuning del modelo neural hibrido.

Aunque tienen extension `.yaml`, varios archivos usan sintaxis JSON valida. No asumir YAML libre si se edita con scripts; conservar formato compatible con `json.loads` cuando aplique.

### Base SQLite

La fuente central consultable es:

```text
data/quiniela.db
```

SQLite guarda estado actual, historicos, predicciones, backtests y metricas. La UI actual no consulta SQLite directamente en el navegador; los scripts Python leen SQLite y generan HTML estatico con `const DATA = {...}`.

Tablas/vistas relevantes para el dashboard diario:

```text
tournament_state_runs
state_matches
state_group_tables
state_team_form
v_latest_tournament_state
v_latest_state_matches
v_latest_state_group_tables
v_latest_state_team_form
model_prediction_runs
model_predictions
v_latest_model_predictions
```

Tablas/vistas historicas y de entrenamiento:

```text
history_ingestion_runs
history_source_files
canonical_historical_matches
v_latest_history_run
v_model_training_matches
v_team_rating_inputs
```

Tablas/vistas de backtest:

```text
backtest_runs
backtest_matches
backtest_predictions
backtest_model_metrics
backtest_parameter_trials
v_latest_backtest_run
v_latest_backtest_model_metrics
v_latest_backtest_predictions
```

### Predicciones diarias

Cada corrida de `scripts/run_model.py` crea una carpeta versionada:

```text
data/predictions/{prediction_run_id}/
├── {model_id}.json
└── {model_id}.csv
```

Ademas actualiza el archivo puente para la UI:

```text
data/ui/prediction_overrides.json
```

Ese archivo es el contrato directo del dashboard diario. Contiene `quiniela_pick`, `frozen_pick`, `model_predictions` y notas por partido.

### Modelos entrenados

Los artefactos neuronales y summaries se guardan en:

```text
data/models/neural_scoreline/
├── latest/
│   ├── model.pt
│   ├── metadata.json
│   └── metrics.json
└── training_summary.json

data/models/neural_hybrid_v2/
├── latest/
│   ├── model.pt
│   ├── metadata.json
│   └── metrics.json
└── training_summary.json
```

Los tunings se guardan aparte para no reemplazar automaticamente el modelo final:

```text
data/models/neural_scoreline_tuning/
data/models/neural_hybrid_v2_tuning/
```

Normalmente existe un:

```text
best_config.json
```

que despues se pasa al script de entrenamiento final.

### Backtests

Cada corrida de backtest crea artefactos versionados:

```text
data/backtests/{backtest_run_id}/
├── backtest_results.json
└── backtest_predictions.csv
```

Los resultados tambien se insertan en SQLite para que el dashboard historico lea siempre la ultima corrida desde `v_latest_backtest_*`.

Los tunings de modelos clasicos pueden escribir archivos como:

```text
data/backtests/tuning_{model}_{timestamp}.json
data/backtests/tuning_{model}_{years}_checkpoint.json
```

### Salidas HTML

Las dos paginas actuales son HTML estatico:

```text
outputs/dashboard/index.html
outputs/validation_dashboard/index.html
```

No hay servidor web obligatorio. Se pueden abrir con `start` en Windows o desde el navegador.

La ventaja es que son faciles de compartir y auditar. La desventaja es que si se unifican los tableros, conviene generar un payload comun mas claro en vez de duplicar mucho JS embebido.

### Flujo de datos hacia UI

Flujo diario:

```text
SQLite estado torneo
  + data/ui/prediction_overrides.json
  -> scripts/generate_dashboard.py
  -> outputs/dashboard/index.html
```

Flujo de backtest:

```text
SQLite backtest latest
  + configs/scoring.yaml
  + summaries neuronales opcionales
  -> scripts/generate_validation_dashboard.py
  -> outputs/validation_dashboard/index.html
```

Flujo recomendado si se unifican ambos tableros:

```text
SQLite + JSON de predicciones + configs
  -> capa Python de payloads
  -> un unico HTML/app local
     ├── Vista Torneo 2026
     ├── Vista Backtest
     ├── Vista Modelos
     └── Vista Comparacion/Ensemble
```

## Consideraciones para unificar ambos tableros

Si se decide fusionar `dashboard` y `validation_dashboard`, conviene pensar en una app local unica con secciones o tabs, no mezclar todo en una sola pantalla gigante.

Propuesta de navegacion:

```text
Quiniela2026
├── Torneo 2026
│   ├── Grupos
│   ├── Proximos partidos
│   ├── Bracket
│   └── Hover/modal de modelos
├── Backtest
│   ├── Ranking
│   ├── Partidos evaluados
│   └── Filtros por ano/modelo/fase
├── Modelos
│   ├── Artefactos disponibles
│   ├── Metricas de entrenamiento
│   └── Estado de referencia/no limpio
└── Ensemble
    ├── Pesos vigentes
    ├── Comparacion weighted vs base
    └── Objetivos points/1X2/exact
```

Recomendacion tecnica:

- Mantener dos payloads logicos separados: `tournamentPayload` y `backtestPayload`.
- Compartir componentes visuales: tarjeta de modelo, pill de puntos, tabla scrollable, selector de modelo, badge de referencia.
- Mantener la regla de negocio fuera del frontend: el frontend presenta, filtra y agrega; los modelos y backtests se calculan en Python.
- No eliminar `prediction_overrides.json` sin reemplazarlo por un contrato equivalente.
- Si se introduce servidor local, debe ser opcional; el HTML estatico sigue siendo util como artefacto auditable.

## Flujo diario operativo

Con el entorno `quiniela2026` activado y estando en `D:\Quiniela2026`:

```powershell
python scripts\run_model.py
python scripts\generate_dashboard.py
```

Para actualizar tambien la pagina historica:

```powershell
python scripts\run_backtest.py
python scripts\generate_validation_dashboard.py
```

Para abrir:

```powershell
start outputs\dashboard\index.html
start outputs\validation_dashboard\index.html
```

## Dashboard 1: seguimiento diario del torneo

### Objetivo

La pagina `outputs/dashboard/index.html` es el tablero local para decidir la quiniela diaria. Debe permitir ver:

- Estado del torneo.
- Tablas de grupos.
- Calendario de partidos.
- Fase eliminatoria.
- Resultado registrado si el partido ya ocurrio.
- Pronostico propuesto para la quiniela.
- Pronosticos de todos los modelos por partido.
- Pick congelado cuando un partido ya fue jugado.

### Logica de consulta

El generador abre SQLite en `data/quiniela.db` y consulta:

```sql
SELECT * FROM v_latest_tournament_state;
SELECT * FROM v_latest_state_matches
ORDER BY COALESCE(match_number, CAST(source_match_id AS INTEGER));
SELECT * FROM v_latest_state_group_tables
ORDER BY group_name, rank_sort;
SELECT * FROM v_latest_state_team_form
ORDER BY group_name, team_name;
```

Despues mezcla esos datos con:

```text
data/ui/prediction_overrides.json
```

Ese JSON lo escribe `scripts/run_model.py` y contiene, por partido:

```json
{
  "matches": {
    "1": {
      "quiniela_pick": {
        "model_id": "weighted_points_ensemble",
        "score": "1-0",
        "expected_points": 1.308386,
        "top_score": "1-0",
        "top_score_probability": 0.1186221483
      },
      "frozen_pick": false,
      "model_predictions": [
        {
          "model_id": "elo_poisson",
          "family": "fuerza+goles",
          "score": "1-0",
          "top_score": "1-1",
          "expected_goals": "1.76-1.12",
          "outcome": "1",
          "confidence": 0.5217,
          "expected_points": 1.186661,
          "notes": "incluye ajuste de anfitrion"
        }
      ],
      "notes": "prediction_run_id=pred_..."
    }
  }
}
```

La pagina no hace llamadas de red. Todo queda embebido en el HTML como:

```js
const DATA = {...};
```

### Estructura visual

Se eligio una propuesta hibrida entre poster mundialista y dashboard operativo:

- Header superior con titulo y KPIs.
- Columna izquierda con grupos A-F.
- Centro con panel visual, copa estilizada y fase eliminatoria.
- Columna derecha con grupos G-L.
- Cada grupo es una tarjeta compacta con tabla y partidos.
- Cada partido es un boton con hover y click.
- El hover es el resumen rapido.
- El modal es el detalle amplio.

Layout conceptual:

```text
┌──────────────────────────────────────────────────────────────┐
│ Quiniela Mundial 2026             KPIs                       │
├───────────────┬──────────────────────────────┬───────────────┤
│ Grupos A-F    │ Centro: mapa diario/bracket  │ Grupos G-L    │
│ Tablas        │ Fase eliminatoria            │ Tablas        │
│ Partidos      │                              │ Partidos      │
└───────────────┴──────────────────────────────┴───────────────┘
```

### Interaccion por partido

Hover:

- Aparece una tarjeta flotante junto al cursor.
- Muestra equipos, fecha, grupo/fase y estado.
- Muestra la propuesta de quiniela.
- Muestra una tarjeta compacta por modelo.
- Las tarjetas de modelo van en 3 columnas para evitar que el hover sea demasiado largo.

Click:

- Abre modal.
- Muestra detalle de sede, estado, quiniela, datos temporales y tabla de modelos.
- Debe mantenerse como vista de inspeccion mas completa que el hover.

### Informacion de modelo en hover

Cada modelo debe mostrar principalmente dos marcadores:

```text
Top / Mas probable = top_score
Max pts = score
```

Definiciones:

- `top_score`: marcador individual con mayor probabilidad en la matriz de marcadores.
- `score`: marcador recomendado para la quiniela, porque maximiza puntos esperados segun la regla 5/3/1.

Contexto secundario:

- `xG`: goles esperados.
- `1X2`: ganador/empate/perdedor mas probable y confianza.
- `EV`: puntos esperados de quiniela para el marcador recomendado.
- `notes`: ajustes, pesos o advertencias.

El hover no debe volver a mostrar solo "N modelos". Debe mostrar la salida concreta de cada modelo.

### Pick congelado

Cuando un partido ya ocurrio:

- El resultado registrado se muestra desde `v_latest_state_matches`.
- El pick usado para quiniela debe quedar congelado.
- `frozen_pick=true` evita reemplazar retroactivamente la seleccion.
- La UI debe dejar claro si el pick esta congelado.

Esto es critico para auditoria: la quiniela se llena antes del partido y no se debe mejorar el pronostico despues del resultado.

### Partidos sin equipos asignados

Algunos partidos de eliminatoria aun tienen equipos como `Winner Match 80`. Esos partidos no deben romper la UI:

- Pueden aparecer como "Por definir" o texto equivalente.
- Normalmente no tienen `model_predictions`.
- La interfaz debe mostrar "sin pronosticos todavia" en vez de error.

### Estilo visual vigente

Variables aproximadas del dashboard diario:

```css
--bg: #f6f7f3;
--paper: #fffdf8;
--ink: #1f2933;
--muted: #667085;
--line: #d9ded6;
--burgundy: #641f3e;
--teal: #087e8b;
--gold: #d6a328;
--coral: #d95d39;
--green: #247a57;
```

Rasgos de estilo:

- Fondo claro con grid sutil, no fondo oscuro.
- Paleta mundialista sobria: burgundy, teal, gold, coral, green.
- Cards con radio maximo de 8px.
- Bordes suaves y sombras moderadas.
- Tipografia de sistema tipo Inter.
- Letras sin tracking negativo.
- UI compacta y escaneable, mas herramienta de trabajo que landing page.
- Botones de partido con informacion en una sola fila: fecha, equipo A, marcador/VS, equipo B.

### Decisiones visuales importantes

- No usar una landing hero. La primera pantalla debe ser la herramienta.
- No esconder datos importantes detras de decoracion.
- El panel central puede tener una copa estilizada, pero la informacion operativa manda.
- No usar tarjetas dentro de tarjetas de forma excesiva.
- El hover debe tener ancho grande pero controlado:

```css
width: min(1120px, calc(100vw - 28px));
max-height: min(620px, calc(100vh - 28px));
overflow: auto;
```

- En mobile el hover baja a 2 columnas y luego 1 columna.
- El modal debe permitir scroll interno.
- Las notas de `ensemble_weights` deben soportar salto de linea.

### Aprendizaje tecnico del hover

Hubo un bug: convertir notas con `replace(/\n/g, "<br>")` dentro del JS embebido podia romper el dashboard si los saltos de linea no se escapaban correctamente. La solucion vigente:

```js
function formatNotes(value) {
  return escapeHtml(value);
}
```

Y CSS:

```css
.model-note {
  white-space: pre-wrap;
}

.model-prediction-table td:last-child {
  white-space: pre-wrap;
}
```

Si se mejora el diseno, conservar este enfoque o escapar estrictamente los saltos de linea.

## Dashboard 2: validacion historica / backtesting

### Objetivo

La pagina `outputs/validation_dashboard/index.html` permite evaluar como se comportaron los modelos en Mundiales anteriores. Sirve para comparar:

- Puntos de quiniela obtenidos.
- Maximo posible.
- Eficiencia.
- Exactos.
- Aciertos 1X2.
- Diferencia/empate.
- Brier/logloss.
- Comparacion entre `Max puntos` y `Mas probable`.
- Modelos ponderados vs modelos individuales.
- Modelos neuronales marcados como referencia cuando no son backtest limpio.

### Logica de consulta

El generador abre SQLite y asegura el schema de backtest. Luego consulta:

```sql
SELECT * FROM v_latest_backtest_run;

SELECT *
FROM v_latest_backtest_model_metrics
ORDER BY
  CASE WHEN year = 'all' THEN 0 ELSE 1 END,
  total_quiniela_points DESC,
  exact_hits DESC,
  model_id;

SELECT *
FROM v_latest_backtest_predictions
ORDER BY year, match_number, model_id;
```

Tambien lee:

```text
configs/scoring.yaml
data/models/neural_scoreline/training_summary.json
data/models/neural_scoreline/latest/metrics.json
```

El HTML tambien embebe todo como:

```js
const DATA = {...};
```

No hay backend web ni llamadas asincronas.

### Regla temporal del backtest

El backtest simula que los modelos se ejecutan antes de cada partido. La regla base es:

```text
match_date < fecha_del_partido
```

No se usan partidos del mismo dia porque la fuente historica no siempre tiene kickoff exacto confiable.

Para el Mundial 2026 real, la decision operativa es mas estricta:

```text
La quiniela se llena un dia antes.
Ningun resultado del mismo dia debe entrar como feature.
```

### Estructura visual

La pagina historica es mas de analisis operativo que de poster:

- Header con titulo, run id y KPIs.
- Panel izquierdo: ranking por puntos.
- Panel derecho: tabla de partidos evaluados.
- Panel inferior opcional: metricas del modelo neural entrenado.

Layout conceptual:

```text
┌──────────────────────────────────────────────────────────────┐
│ Validacion de modelos                  KPIs                  │
├──────────────────────────────┬───────────────────────────────┤
│ Ranking por puntos           │ Partidos evaluados             │
│ Filtros + barras             │ Tabla scrollable               │
└──────────────────────────────┴───────────────────────────────┘
│ Modelo neural entrenado (opcional)                           │
└──────────────────────────────────────────────────────────────┘
```

### Filtros

La pagina tiene selects para:

- Ano.
- Modelo.
- Fase.
- Orden.

Ordenes:

- `points`: puntos de quiniela.
- `exact`: exactos.
- `winner`: ganador/empate/perdedor.

La agregacion del ranking se recalcula en el navegador sobre `DATA.predictions`, no desde el servidor.

### Dos perspectivas por modelo

El dashboard historico evalua dos estrategias:

1. `max_points`: usa `selected_score`, el marcador que maximiza puntos esperados de quiniela.
2. `most_probable`: usa `top_score`, el marcador mas probable.

Ambas deben verse en el ranking porque contestan preguntas distintas:

- `max_points` responde: "Que marcador conviene jugar en la quiniela?"
- `most_probable` responde: "Que marcador cree mas probable el modelo?"

La UI usa una etiqueta/pill:

```text
Max puntos
Mas probable
```

### Puntos maximos y eficiencia

La comparacion debe mostrar:

```text
total_quiniela_points / max_possible_points
points_efficiency = total_quiniela_points / max_possible_points
```

Con la regla vigente, cada partido tiene maximo 5 puntos:

```text
max_possible_points = matches_evaluated * 5
```

No basta mostrar puntos absolutos; la eficiencia permite comparar corridas con diferente numero de partidos.

### Tabla de partidos evaluados

Columnas vigentes:

- Partido.
- Modelo.
- Ano.
- Fase.
- Real.
- Max puntos.
- Puntos.
- Mas probable.
- Puntos.
- 1X2 Max.
- 1X2 Prob.
- xG.

La tabla debe ser scrollable y mantener headers sticky.

### Modelos de referencia

Algunos modelos, especialmente neuronales con artefacto final, pueden aparecer como referencia visual y no como backtest limpio.

Motivo:

- Un artefacto final entrenado hoy pudo haber visto informacion posterior a 2018/2022.
- Sirve para inspeccionar comportamiento, pero no debe confundirse con validacion temporal limpia.

La UI marca estos modelos con:

```text
Referencia
```

Y muestra una nota amarilla explicativa.

### Estilo visual vigente

Variables aproximadas:

```css
--bg: #f5f7f6;
--panel: #ffffff;
--ink: #1e2930;
--muted: #65717b;
--line: #d7dedb;
--berry: #7b2546;
--teal: #087e8b;
--gold: #d6a328;
--green: #247a57;
--coral: #d95d39;
--blue: #2f6f9f;
```

Rasgos:

- Fondo claro con grid sutil.
- Paneles blancos con bordes.
- Titulos de panel en berry o teal.
- Barras de ranking con gradiente teal -> gold.
- Badges de referencia amarillos.
- Pildoras de puntos:
  - 0 puntos: coral.
  - 1 punto: teal.
  - 3 puntos: gold.
  - 5 puntos: green.

### Diferencia entre ambas paginas

Dashboard diario:

- Prioriza decision de quiniela antes de partidos.
- Estructura de torneo y grupos.
- Hover por partido.
- Modal de detalle.
- Pick congelado.
- Usa estado actual del Mundial 2026 + predicciones actuales.

Dashboard historico:

- Prioriza comparacion de modelos.
- Ranking y tabla de evaluacion.
- Filtros y orden interactivo.
- Muestra resultados reales historicos.
- Usa backtest y metricas.

Ambas deben compartir lenguaje visual, pero no tienen que tener el mismo layout.

## Ponderadores y como presentarlos

Se agregaron cuatro modelos ponderados:

- `weighted_ensemble`: balanceado.
- `weighted_points_ensemble`: optimiza puntos de quiniela.
- `weighted_1x2_ensemble`: optimiza ganador/empate/perdedor.
- `weighted_exact_ensemble`: optimiza marcador exacto.

El ponderador consume matrices de marcadores normalizadas de modelos base. Publica una prediccion con el mismo contrato que cualquier otro modelo.

En dashboard diario:

- Deben aparecer como modelos adicionales en el hover.
- `weighted_points_ensemble` es la propuesta principal de quiniela.
- Las notas deben mostrar primero `ensemble_objective` y luego, en linea separada, `ensemble_weights`.

Ejemplo:

```text
ensemble_objective=max_points
ensemble_weights={"elo_poisson": 0.15, "neural_scoreline_mlp": 0.17, ...}
```

En dashboard historico:

- Deben aparecer en ranking y tabla igual que los demas.
- Si consumen artefactos neuronales finales, deben tratarse como referencia visual cuando aplique.

## Aprendizajes de diseno y producto

1. La interfaz es una herramienta diaria, no una landing page. Debe priorizar escaneo y decision.
2. El usuario necesita comparar modelos, no solo ver un pronostico final.
3. Para elegir quiniela importan dos marcadores: el mas probable y el de maximo valor esperado.
4. Los pesos del ensemble deben ser auditables en la UI.
5. Los partidos sin equipos definidos no deben romper evaluacion ni visualizacion.
6. Los picks de partidos jugados deben congelarse.
7. Los dashboards deben regenerarse como HTML estatico local.
8. La validacion historica debe advertir cuando un modelo es solo referencia.
9. No conviene ocultar las metricas bajo graficas decorativas; ranking y tabla siguen siendo necesarios.
10. El hover puede ser grande, pero debe ser legible y no tapar todo sin scroll.
11. Los saltos de linea en JSON embebido/JS son delicados; escapar o preservar con CSS.
12. El look debe ser sobrio, claro, con informacion densa pero ordenada.

## Recomendaciones para Claude Design

Mantener:

- HTML estatico autocontenido.
- Datos embebidos en `const DATA`.
- Contrato de modelo actual.
- Distincion `top_score` vs `score`.
- Badges `Referencia`.
- Pick congelado.
- Filtros del dashboard historico.
- Scroll interno en hover/modal/tablas.

Mejorar:

- Jerarquia visual del hover para que `weighted_points_ensemble` destaque sin ocultar modelos base.
- Compactar notas largas de pesos con un disclosure o area plegable.
- Agregar color consistente por familia de modelo.
- Agregar busqueda/filtro de modelo en el hover si crece demasiado.
- Hacer que el dashboard diario tenga una vista alternativa "solo partidos proximos".
- En historico, agregar mini sparklines o small multiples por ano sin reemplazar la tabla.
- Mejorar responsive mobile con tabs: Grupos, Bracket, Proximos, Modelos.

Evitar:

- Reemplazar el dashboard por una pagina hero.
- Usar gradientes/decoracion que compitan con tablas.
- Ocultar `Max puntos` o `Mas probable`.
- Hacer depender la UI de un modelo especifico.
- Mostrar resultados de referencia como si fueran validacion limpia.
- Romper el HTML estatico con dependencias externas innecesarias.

## Validaciones recomendadas despues de cambios

Con entorno `quiniela2026`:

```powershell
python -m compileall src scripts
python scripts\generate_dashboard.py
python scripts\generate_validation_dashboard.py
```

Validacion JS del dashboard diario:

```powershell
node -e "const fs=require('fs'); const html=fs.readFileSync('outputs/dashboard/index.html','utf8'); const m=html.match(/<script>([\s\S]*)<\/script>/); if(!m) throw new Error('script no encontrado'); new Function(m[1]); console.log('JS_OK');"
```

Validacion JS del dashboard historico:

```powershell
node -e "const fs=require('fs'); const html=fs.readFileSync('outputs/validation_dashboard/index.html','utf8'); const m=html.match(/<script>([\s\S]*)<\/script>/); if(!m) throw new Error('script no encontrado'); new Function(m[1]); console.log('JS_OK');"
```

## Resumen ejecutivo

El dashboard diario es una vista tipo poster operativo para tomar decisiones de quiniela antes de cada partido. El dashboard historico es una vista de auditoria y comparacion de modelos. Ambos comparten una estetica clara, compacta y mundialista, con datos densos, cards sobrias y pildoras de estado. La mejora de diseno debe preservar la modularidad: cada modelo es solo una fila/tarjeta que cumple el contrato comun; agregar, quitar o retunear modelos no debe romper ninguna pagina.
