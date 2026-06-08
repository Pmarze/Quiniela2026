# 036 - Repositorio privado GitHub

## Contexto

El proyecto se preparara para subir a GitHub como repositorio privado.

Hay artefactos pesados y locales en `data/` y `outputs/`, incluyendo checkpoints de PyTorch de mas de 160 MB cada uno. Esos archivos no deben versionarse en Git normal.

## Decision

Se agregaron:

- `.gitignore`
- `.gitattributes`
- `.env.example`
- `docs/repository_setup.md`
- `.gitkeep` en carpetas generadas clave

El repositorio versiona codigo, configuracion y documentacion. No versiona:

- bases SQLite
- datos descargados
- predicciones generadas
- backtests generados
- dashboards renderizados
- checkpoints/modelos entrenados
- `.env`
- `.claude/settings.local.json`

## Comandos recomendados

```powershell
git add .
git commit -m "Initial private project structure"
gh repo create quiniela2026 --private --source . --remote origin --push
```

Si no se usa GitHub CLI:

```powershell
git remote add origin https://github.com/TU_USUARIO/quiniela2026.git
git branch -M main
git push -u origin main
```

## Estado

Reemplazado parcialmente por la nota 038. Sigue vigente excluir `.env`, configuracion local y artefactos de trabajo local, pero la politica actual agrega `model_registry/` para compartir modelos finales entrenados.
