# 012 - Dashboard hover con detalle de modelos

## Conocimiento

El hover de cada partido ya no debe mostrar solo el conteo de modelos. Debe mostrar la salida resultante de cada modelo disponible para ese partido.

## Cambio Implementado

Archivo modificado:

```text
src/quiniela/ui/dashboard.py
```

Salida regenerada:

```text
outputs/dashboard/index.html
```

Ahora el hover muestra una tarjeta por modelo con:

```text
model_id
score recomendado
top_score
expected_goals
outcome 1X2
confidence
expected_points
```

Ejemplo esperado:

```text
baseline_poisson | 1-1 | Top 1-1 | xG 1.40-1.40 | 1X2 1 37.4% | EV 0.995
elo_poisson      | 1-0 | Top 2-0 | xG 2.09-0.94 | 1X2 1 63.7% | EV 1.315
```

## Validacion

Se ejecuto:

```powershell
python scripts\generate_dashboard.py
```

Validaciones locales:

```text
python -m compileall src scripts
js_syntax: ok
html_checks: ok
```

El navegador integrado no pudo iniciar por restriccion del sandbox de Windows, igual que en validaciones anteriores.

## Estado

Activo. No contradice conocimientos anteriores; actualiza el comportamiento visual del dashboard descrito en la nota 005.
