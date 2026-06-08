# 013 - Hover top score vs max puntos

## Conocimiento

En el hover del dashboard se debe priorizar la comparacion entre dos marcadores por modelo:

```text
Mas probable
Max puntos
```

## Definiciones

`Mas probable` corresponde a:

```text
top_score
```

Es el marcador individual con mayor probabilidad en la matriz de marcadores.

`Max puntos` corresponde a:

```text
score
```

Es el marcador recomendado para quiniela porque maximiza los puntos esperados segun `configs/scoring.yaml`.

## Cambio Implementado

Archivo modificado:

```text
src/quiniela/ui/dashboard.py
```

Salida regenerada:

```text
outputs/dashboard/index.html
```

Cada tarjeta de modelo en hover ahora resalta:

```text
Mas probable: top_score
Max puntos: score
```

Y deja como contexto secundario:

```text
xG
1X2 + confianza
EV
```

## Monte Carlo

Para `baseline_poisson` y `elo_poisson`, Monte Carlo no es necesario para estimar el partido individual porque ambos ya construyen una matriz completa de probabilidades de marcadores.

Monte Carlo seria util despues para:

```text
simular grupos
simular clasificacion
simular bracket
estimar probabilidades de campeon/finalista/avance
resumir muchos escenarios del torneo completo
```

No se necesita para decidir entre `top_score` y `score` en estos dos modelos iniciales.

## Validacion

```text
python -m compileall src scripts
python scripts\generate_dashboard.py
js_syntax: ok
```

## Estado

Activo. Amplia la nota 012 con una decision visual mas enfocada para elegir quiniela.
