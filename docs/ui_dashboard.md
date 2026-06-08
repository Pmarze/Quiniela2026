# Dashboard Local del Torneo

## Objetivo

Crear una interfaz web local para revisar el seguimiento del Mundial, el estado de grupos, partidos pendientes/jugados, pronosticos de modelos y la propuesta de quiniela.

La interfaz debe poder regenerarse diariamente despues de:

```powershell
python scripts\download_data.py
python scripts\build_canonical.py
python scripts\build_state.py
python scripts\generate_dashboard.py
```

Tambien puede usarse el flujo diario unico:

```powershell
python scripts\run_daily.py
```

## Propuestas Visuales

### Propuesta A - Poster de seguimiento

Inspirada en calendarios visuales de torneo:

- Grupos a los lados.
- Centro con copa/bracket.
- Todos los partidos visibles.
- Ideal para tener una vista panoramica.

Ventaja:

- Muy facil de escanear.

Riesgo:

- Puede quedar densa en pantallas pequenas.

### Propuesta B - Dashboard operativo

Vista mas parecida a una herramienta de trabajo:

- KPIs arriba.
- Grupos en cards compactas.
- Fase eliminatoria separada.
- Hover/click para detalle.

Ventaja:

- Mas mantenible y facil de extender con modelos.

Riesgo:

- Menos parecida a poster mundialista clasico.

### Propuesta C - Hibrida

Combina poster y dashboard:

- Grupos A-F a la izquierda.
- Centro con visual de copa y eliminatorias.
- Grupos G-L a la derecha.
- Popups por partido.
- Modal de detalle para quiniela/modelos.

Decision inicial:

```text
Implementar Propuesta C.
```

No contradice decisiones anteriores porque agrega una capa nueva de visualizacion.

## Prototipo

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Quiniela Mundial 2026                         Partidos | Jugados | Pendientes │
│ state_id / as_of_utc                                                        │
├───────────────────────┬───────────────────────────────┬──────────────────────┤
│ Grupo A               │                               │ Grupo G              │
│ Tabla P W D L GF GA   │        MAPA DIARIO            │ Tabla P W D L GF GA  │
│ Mex vs RSA   [hover]  │           COPA                │ Bel vs Egy  [hover]  │
│ Kor vs Cze   [hover]  │                               │ ...                  │
├───────────────────────┤      FASE ELIMINATORIA        ├──────────────────────┤
│ Grupo B               │ R32 | R16 | QF | SF | Final   │ Grupo H              │
│ ...                   │ [partido] [partido] [final]   │ ...                  │
├───────────────────────┤                               ├──────────────────────┤
│ Grupo C-F             │                               │ Grupo I-L            │
└───────────────────────┴───────────────────────────────┴──────────────────────┘

Hover/click partido:

┌──────────────────────────────────────────┐
│ Mexico vs South Africa                   │
│ Fecha, sede, grupo, estado               │
├──────────────────────────────────────────┤
│ Resultado registrado                     │
│ Pronostico quiniela                      │
│ Pick congelado si el partido ya ocurrio  │
│ Pronosticos por modelo                   │
│ - elo_poisson                            │
│ - mercado_calibrado                      │
│ - ml_1x2                                 │
└──────────────────────────────────────────┘
```

## Implementacion Actual

Generador:

```text
scripts/generate_dashboard.py
src/quiniela/ui/dashboard.py
```

Salida:

```text
outputs/dashboard/index.html
```

Datos consumidos:

```text
v_latest_tournament_state
v_latest_state_matches
v_latest_state_group_tables
v_latest_state_team_form
```

## Datos de Pronosticos

La interfaz consume el archivo:

```text
data/ui/prediction_overrides.json
```

Este archivo ya es generado por:

```powershell
python scripts\run_model.py
```

Formato actual:

```json
{
  "matches": {
    "1": {
      "quiniela_pick": {
        "score": "2-0",
        "expected_points": 2.41
      },
      "frozen_pick": false,
      "model_predictions": [
        {
          "model_id": "elo_poisson",
          "score": "1-0",
          "top_score": "2-0",
          "expected_goals": "2.09-0.94",
          "outcome": "1",
          "confidence": 0.6368,
          "expected_points": 1.31531
        }
      ],
      "notes": "Demo"
    }
  }
}
```

El hover de cada partido muestra una tarjeta por modelo con:

```text
modelo
familia del modelo
Mas probable: top_score de la matriz
Max puntos: score recomendado por ese modelo
xG esperado
1X2 y confianza
puntos esperados
```

El modal mantiene una tabla resumida de modelos con:

```text
Modelo
Mas prob.
Max puntos
xG
1X2
EV
```

## Regla de Pick Congelado

Cuando un partido ya ocurrio:

- El resultado registrado se muestra desde `state_matches`.
- El pronostico usado para quiniela debe quedar congelado.
- El modal debe mostrar `frozen_pick=true`.
- El sistema no debe reemplazar retroactivamente el pick usado para ese partido.

## Validaciones

Realizadas:

```text
python -m compileall src scripts
python scripts/generate_dashboard.py
Node vm.Script sobre el script embebido en HTML
Parser HTML para IDs principales
```

El navegador integrado no pudo arrancar en esta sesion por una restriccion del sandbox de Windows. Se valido el HTML generado con chequeos locales de sintaxis JS y presencia de los bloques del hover.
