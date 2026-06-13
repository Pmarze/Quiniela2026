# 039 - Handoff dashboard y operacion 2026 live

Fecha de actualizacion: 2026-06-12.

## Resumen

Esta nota deja el contexto operativo para que otro chat, Codex o Claude Code pueda continuar el trabajo sin reconstruir toda la conversacion.

El objetivo sigue siendo maximizar puntos de quiniela, no solamente exactitud 1X2. La prioridad de scoring es:

- marcador exacto
- empate o mismo margen/diferencia
- ganador correcto

## Runtime y outputs

Trabajar siempre desde la raiz del repo con el entorno:

```powershell
conda activate quiniela2026
```

El dashboard principal casi-publicable se genera en:

```text
docs/index.html
```

Tambien mantener sincronizado:

```text
outputs/dashboard/index.html
```

No editar HTML generado a mano. Editar:

- `src/quiniela/ui/dashboard.py`
- `src/quiniela/ui/dashboard_template.html`

Regenerar dashboard publico/local con amigos:

```powershell
python scripts\generate_dashboard.py
python scripts\check_public_dashboard.py docs\index.html
```

Regenerar una version sin amigos solo si se pide:

```powershell
python scripts\generate_dashboard.py --exclude-friends
```

La politica actual de amigos visibles y fuente de Sheets privada esta en `docs/knowledge/040_publicacion_publica_dashboard_privado.md`.

## Estado local del torneo al handoff

Ultimo daily ejecutado:

- state_id: `state_20260613T051235Z_26d17845`
- as_of_utc: `2026-06-13T05:12:35Z`
- partidos completados: 4
- pendientes: 100

Resultados registrados:

1. Mexico 2-0 South Africa
2. South Korea 2-1 Czech Republic
3. Canada 1-1 Bosnia and Herzegovina
4. United States 4-1 Paraguay

Ultimo run de predicciones relevante:

- prediction_run_id: `pred_20260613T051336Z_4c5a6565`
- modelos activos ejecutados: 15
- por modelo: 68 `ok`, 36 `masked`, 0 `failed`
- `baseline_poisson` no esta en `model_predictions`

## Cambios de modelos

### baseline_poisson desactivado

`baseline_poisson` quedo desactivado en `configs/models.yaml` y fue removido de `data/ui/prediction_overrides.json`.

Esto afecta los ponderadores porque el modelo ya no participa como modelo base activo ni aparece en el dashboard operativo.

### similar_match_knn_scoreline agregado

Se agrego el modelo experimental:

```text
similar_match_knn_scoreline
```

Archivo principal:

```text
src/quiniela/models/similar_match_knn_scoreline.py
```

Idea:

- busca partidos historicos analogos por diferencia de rating, entorno de goles, perfiles ataque/defensa, sede y tipo de partido
- construye distribucion empirica de marcadores con vecinos KNN
- mezcla con prior Poisson para evitar sobreajuste

Politica actual:

- activo como modelo independiente
- aparece en dashboard
- excluido de ponderadores
- excluido del pick automatico preferido hasta acumular mas evidencia

## Seleccion del modelo de quiniela

Antes se elegia el modelo operativo por backtest historico. Eso generaba inconsistencia: el dashboard de Validacion 2026 mostraba como lider vivo a `neural_scoreline_mlp`, pero el pick operativo seguia usando `elo_poisson`.

Ahora `scripts/run_model.py` hace:

1. Si no hay partidos reales jugados, usa backtest historico/default.
2. Si ya hay resultados reales, selecciona el modelo por ranking vivo 2026 usando predicciones congeladas antes de cada partido.
3. El desempate sigue puntos, exactos, aciertos y orden estable.

Al handoff, el selector eligio:

```text
neural_scoreline_mlp
```

con:

```text
pts_actuales=7
exactos=1
partidos=4
tier=current_2026
```

Los partidos finalizados conservan el pick congelado original; los pendientes usan el lider vivo actual.

## Cambios principales del dashboard

### Hover de partido

El hover/modal de cada partido incluye:

- scatter de xG por modelo
- estrella de referencia basada en amistosos recientes ponderados
- heatmap de consenso de marcadores
- probabilidad 1X2 con rango de modelos
- amistosos recientes de preparacion por equipo
- referencia de probabilidad de gol de amistosos recientes
- pagina de modelos individuales

### Heatmap de marcadores

El heatmap ahora muestra:

- porcentaje como valor principal
- conteo exacto de modelos como numero pequeno debajo

Ejemplo conceptual:

```text
86.7%
13
```

No debe mostrar el texto `mod.`.

### Probabilidad 1X2

Cada fila 1X2 ahora tiene dos carriles:

- arriba: consenso/promedio de modelos
- abajo: modelo lider actual por performance
- linea dorada: rango minimo-maximo entre modelos

Corregido un bug visual: la tabla 2026 `Mas Probable` usaba `var(--blue)` pero la variable real del tema es `var(--blu)`.

### Validacion 2026 live

En la seccion Validacion hay dos rankings live:

- `2026 · Max Pts.`
- `2026 · Mas Probable`

Al hacer click sobre un modelo se abre el panel `val-hover` con comparacion directa contra resultados reales del torneo actual.

El panel muestra:

- resumen del modelo
- puntos del modo seleccionado
- exactos
- aciertos
- error promedio xG
- tabla partido por partido

Si se abre desde `Max Pts.`, la tabla evalua primero `Max Pts.`.

Si se abre desde `Mas Probable`, la tabla evalua primero `Top`.

La tabla incluye:

- partido
- resultado real
- pick primario y puntos
- pick secundario y puntos
- 1X2 predicho vs 1X2 real
- probabilidad asignada a la salida real (`P(real)`)
- xG del modelo y error xG

### Metodologia

Se recupero una seccion extensa pero compacta de metodologia con explicacion por modelo, incluyendo KNN:

- Elo Poisson
- Elo Dixon-Coles
- Ataque/Defensa Poisson
- Especialista en empates
- Bradley-Terry-Davidson
- Bayesian Monte Carlo
- Opta Power Poisson
- Neural Scoreline MLP
- Neural Hybrid V2
- Partidos Similares KNN
- Weighted ensembles
- Calibrated Scoreline Ensemble

## Scoring profiles

Existe selector de scoring en el dashboard:

- `5-3-1` default
- `3-1-0` alternativa

Los recalculos visuales del dashboard usan `_scoringValues()` y las funciones JS asociadas.

## Archivos tocados mas relevantes

- `configs/models.yaml`
- `scripts/run_model.py`
- `src/quiniela/models/similar_match_knn_scoreline.py`
- `src/quiniela/models/__init__.py`
- `src/quiniela/backtest/runner.py`
- `src/quiniela/ui/dashboard.py`
- `src/quiniela/ui/dashboard_template.html`
- `docs/dashboard_reference.md`
- `docs/index.html`
- `data/ui/prediction_overrides.json`

`data/` y `outputs/` son generados/locales. No asumir que estan versionados.

## Validaciones usadas

Compilacion:

```powershell
python -m py_compile scripts\run_model.py
```

Regeneracion:

```powershell
python scripts\generate_dashboard.py
python scripts\generate_dashboard.py --output outputs\dashboard\index.html
```

Chequeo estructural del HTML:

- parsear `const DATA = ...`
- confirmar `matches=104`
- confirmar `played=3`
- confirmar `baseline_poisson` ausente en `model_predictions`
- confirmar click live `showCurrentModelHover`
- confirmar `heat_mod_text=False`
- confirmar `node_js_ok=1`

## Problema conocido en Codex Desktop Windows

El navegador integrado de Codex fallo repetidamente con:

```text
CreateProcessAsUserW failed: 5
```

Esto parece bloqueo de permisos/sandbox de Windows al lanzar `node_repl`/browser runtime, no problema del repo.

Runners observados:

- `%USERPROFILE%\.codex\.sandbox-bin\codex-command-runner-*.exe`
- `%LOCALAPPDATA%\OpenAI\Codex\runtimes\cua_node\*\bin\node_repl.exe`

Hasta resolverlo con allowlist/permisos, usar validacion estructural con Python/Node. Claude Code puede no fallar porque usa otra capa de ejecucion/browser.

## Continuacion recomendada

1. Correr daily antes de nuevos partidos o cuando haya resultados:

   ```powershell
   python scripts\daily_update.py --skip-git
   ```

2. Si cambia data de amigos:

   ```powershell
   python scripts\build_friends_quinielas.py
   ```

3. Regenerar ambos dashboards.

4. Revisar `Validacion 2026 · Actual`: el lider vivo puede cambiar partido a partido.

5. Si se desea permitir que KNN compita por pick automatico o ponderadores, quitarlo explicitamente de las listas de exclusion y rerun backtests antes de confiar en el cambio.
