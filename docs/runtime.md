# Runtime Python

## Entorno Principal

El entorno Conda principal del proyecto es:

```text
quiniela2026
```

Python detectado:

```text
Python 3.11.15
```

Conda esta disponible en esta maquina en:

```text
C:\ProgramData\anaconda3\Scripts\conda.exe
```

Python del entorno:

```text
C:\Users\pablo\.conda\envs\quiniela2026\python.exe
```

## Comandos Recomendados

Usando Conda:

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n quiniela2026 python scripts\db_summary.py --samples
```

Usando directamente el Python del entorno, recomendado para validaciones repetidas en PowerShell:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\db_summary.py --samples
```

## Validaciones

Compilar codigo:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" -m compileall src scripts
```

Descargar datos:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\download_data.py
```

Revisar resumen de base:

```powershell
& "C:\Users\pablo\.conda\envs\quiniela2026\python.exe" scripts\db_summary.py --samples
```

## Estado Validado

Codex ya valido este entorno para el proyecto:

```text
python -m compileall src scripts: OK
scripts\db_summary.py --samples: OK
```

## Notas

`conda run` puede fallar ocasionalmente en PowerShell por archivos temporales bloqueados. Cuando eso pase, usar directamente `C:\Users\pablo\.conda\envs\quiniela2026\python.exe`.
