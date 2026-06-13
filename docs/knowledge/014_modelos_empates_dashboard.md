# 014 - Modelos de empates y dashboard robustecido

## Conocimiento

Se implementaron cuatro modelos adicionales hasta la linea de empates acordada:

```text
elo_dixon_coles
attack_defense_poisson
draw_specialist
bradley_terry_davidson
```

El modelo 5 de mercado queda pendiente:

```text
market_calibrated_poisson
```

## Decision Operativa

El modelo recomendado por defecto para proponer la quiniela pasa a:

```text
elo_dixon_coles
```

Esto no edita la nota 010, que conserva el estado inicial del proyecto con `elo_poisson`. La lectura vigente es que `elo_poisson` fue el primer modelo serio y `elo_dixon_coles` es ahora el default operativo porque ajusta mejor marcadores bajos, empates y 1-0.

## Dashboard

El dashboard ahora debe presentar cada modelo con:

```text
familia del modelo
Mas probable
Max puntos
xG
1X2 + confianza
EV
```

La familia ayuda a interpretar rapidamente el rol:

```text
control
fuerza+goles
marcadores bajos
ataque/defensa
empates
1X2+empate
```

## Validacion

Corrida realizada con el entorno Conda:

```powershell
python
```

Resultado:

```text
prediction_run_id: pred_20260605T174603Z_81a7f0bd
baseline_poisson: ok=72 masked=32 failed=0
elo_poisson: ok=72 masked=32 failed=0
elo_dixon_coles: ok=72 masked=32 failed=0
attack_defense_poisson: ok=72 masked=32 failed=0
draw_specialist: ok=72 masked=32 failed=0
bradley_terry_davidson: ok=72 masked=32 failed=0
```

Tambien se regenero:

```text
data/ui/prediction_overrides.json
outputs/dashboard/index.html
```

Validaciones:

```text
python -m compileall src scripts
python scripts\run_model.py
python scripts\generate_dashboard.py
js_syntax: ok
```

## Estado

Activo. Amplia las notas 009, 010, 012 y 013 sin modificar notas anteriores.
