# 005 - Dashboard local

## Conocimiento

Se agrego una capa de interfaz web local para visualizar el seguimiento del torneo.

## Comando

Con Anaconda activado y estando en la carpeta del proyecto:

```powershell
python scripts\generate_dashboard.py
```

Flujo diario sugerido:

```powershell
python scripts\download_data.py
python scripts\build_state.py
python scripts\generate_dashboard.py
```

## Salida

```text
D:\Quiniela2026\outputs\dashboard\index.html
```

## Diseno elegido

Se eligio una interfaz hibrida:

- Grupos A-F a la izquierda.
- Centro con panel visual y fase eliminatoria.
- Grupos G-L a la derecha.
- Hover/click por partido.
- Modal con resultado, sede, quiniela y pronosticos por modelo.

## Preparacion para modelos

Aunque los modelos todavia no existen, el dashboard acepta un archivo opcional:

```text
data/ui/prediction_overrides.json
```

Ese archivo podra incluir:

- `quiniela_pick`
- `frozen_pick`
- `model_predictions`
- `notes`

## Pick congelado

Cuando un partido ya ocurrio, el pick usado para la quiniela debe quedar congelado y no reemplazarse retroactivamente.

## Validacion

Se valido con:

```text
python -m compileall src scripts
python scripts\generate_dashboard.py
Node vm.Script para sintaxis JS
Parser HTML para contenedores principales
```

El Browser integrado no pudo iniciar por una restriccion del sandbox de Windows. No es una falla del codigo generado.

## Estado

Activo. No contradice conocimientos anteriores.

