# CLAUDE.md — Instrucciones para agentes IA

Este archivo es leído automáticamente por Claude Code y cualquier agente IA que trabaje en este repositorio.
**Todas las reglas aquí son obligatorias.**

---

## Rama de trabajo

**REGLA CRÍTICA: Todo trabajo de IA se hace en `development`. NUNCA en `main`.**

```
Rama de desarrollo : development   ← todo push de IA va aquí
Rama de producción : main          ← solo se toca con autorización explícita del usuario
```

### Qué significa esto en la práctica

- Al iniciar una sesión, verificar en qué rama estás: `git branch --show-current`
- Si estás en `main`, cambiar a `development` antes de hacer cualquier cambio: `git checkout development`
- Todos los commits y pushes van a `development`
- **Nunca hacer `git push origin main`** a menos que el usuario lo pida con estas palabras exactas:
  _"haz push a main"_ / _"merge a main"_ / _"promover a producción"_ / _"publicar en main"_

### Cómo promover development → main (solo cuando el usuario lo solicita)

```bash
git checkout main
git merge development
git push origin main
git checkout development   # volver a development inmediatamente
```

### Por qué esta regla existe

`main` es la rama pública/estable. El dashboard de `docs/index.html` en `main` es el que se publica
en GitHub Pages. Los cambios experimentales o en progreso no deben llegar a producción hasta que
el usuario los revise y apruebe explícitamente.

---

## Entorno Python

```
Intérprete : C:\Users\pablo\.conda\envs\quiniela2026\python.exe
Versión    : Python 3.11
```

Siempre usar ese intérprete. No usar `python` o `python3` a secas en PowerShell.

---

## Flujo de actualización diaria

```bash
# 1. Actualizar datos y correr modelos
python scripts/run_daily.py         # o el script correspondiente

# 2. Construir quinielas de amigos (si hay CSVs nuevos)
python scripts/build_friends_quinielas.py

# 3. Regenerar dashboard
python scripts/generate_dashboard.py
# output: docs/index.html (se commitea junto con el resto)

# 4. Commit y push a development
git add ...
git commit -m "..."
git push origin development
```

---

## Estructura del proyecto

| Carpeta / Archivo | Rol |
|---|---|
| `src/quiniela/` | Código fuente principal |
| `src/quiniela/ui/dashboard.py` | Generador Python del dashboard |
| `src/quiniela/ui/dashboard_template.html` | Template HTML/CSS/JS del dashboard |
| `scripts/generate_dashboard.py` | Entry point para generar el dashboard |
| `scripts/build_friends_quinielas.py` | Procesa CSVs de amigos → JSON |
| `curated_inputs/quinielas/` | CSVs de participantes (uno por amigo) |
| `data/ui/prediction_overrides.json` | Predicciones y picks del sistema |
| `data/ui/friends_quinielas.json` | JSON generado de quinielas de amigos |
| `docs/index.html` | Dashboard generado — se publica en GitHub Pages desde `main` |
| `configs/models.yaml` | Configuración de modelos activos |
| `data/quiniela.db` | Base SQLite — local, no se commitea (>100 MB) |

---

## Reglas de código

- No agregar features, refactors ni abstracciones más allá de lo que pide el usuario.
- No agregar comentarios salvo cuando el WHY no sea obvio.
- El objetivo del sistema es maximizar **puntos de quiniela** (exact=5, margin=3, winner=1, miss=0),
  no métricas de ML genéricas.

---

## GitHub Pages

El dashboard se publica automáticamente en:
**`https://pmarze.github.io/Quiniela2026/`**

La GitHub Action en `.github/workflows/deploy-pages.yml` se dispara solo cuando hay push a `main`
que modifique `docs/index.html`. Los pushes a `development` **no** disparan el deploy.
