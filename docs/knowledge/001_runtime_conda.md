# 001 - Runtime Conda

## Conocimiento

El proyecto debe usar el entorno Conda `quiniela2026` como runtime principal para validaciones, ejecucion de scripts e instalacion futura de dependencias.

## Detalles

Conda fue encontrado en:

```text
C:\ProgramData\anaconda3\Scripts\conda.exe
```

El entorno fue detectado en:

```text
quiniela2026
```

Python del entorno:

```text
python
```

Version validada:

```text
Python 3.11.15
```

## Comando recomendado

Usar directamente el `python.exe` del entorno para evitar bloqueos temporales de `conda run` en PowerShell:

```powershell
python scripts\db_summary.py --samples
```

## Validaciones realizadas

Se valido:

```text
python -m compileall src scripts
scripts\db_summary.py --samples
```

Ambas validaciones terminaron correctamente.

## Estado

Activo. No contradice conocimientos anteriores.
