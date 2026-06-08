# Codex Instructions - Quiniela2026

This repository is a private collaborative project for World Cup 2026 quiniela predictions.

## Read First

When starting a new Codex session, read these files before changing code:

1. `README.md`
2. `docs/collaborator_onboarding.md`
3. `memory/MEMORY.md`
4. `memory/project_status.md`
5. `memory/dashboard_status.md`
6. Relevant files in `docs/knowledge/`

The main objective is to maximize quiniela points, not only 1X2 accuracy.
Current scoring priority:

- exact score
- draw or goal-difference/margin
- match winner

## Runtime

Use the active Conda environment named `quiniela2026`.

Do not hardcode a Python path. Commands should be run from the project root after:

```powershell
conda activate quiniela2026
```

Most scripts add `src/` to `sys.path`, so editable install is not required for normal use.

## Rebuildable Local Data

Do not assume `data/` or `outputs/` are versioned. They are local/generated and should be rebuilt:

```powershell
python scripts\bootstrap_data.py --preset base
python scripts\bootstrap_data.py --preset all
```

Artifact policy is documented in:

- `configs/data_artifacts.json`
- `docs/repository_setup.md`

## Shared Model Policy

Local training outputs stay local in:

- `data/models/`
- `data/models_local/`

Only final published models are shared in:

- `model_registry/`

Use:

```powershell
python scripts\publish_model.py --model-id MODEL_ID --version VERSION --source-dir LOCAL_ARTIFACT_DIR
```

Model weights in `model_registry/` use Git LFS.

## Do Not Commit

Do not commit:

- `data/` generated contents
- `outputs/` generated dashboards
- local checkpoints, folds, tuning runs or logs
- `.env`
- `.claude/settings.local.json`

The `.gitignore` keeps folder skeletons via `.gitkeep`.

## Current Operational Defaults

- Default quiniela model: `weighted_points_ensemble`
- Backtest validation years: 2018 and 2022
- Published neural artifacts are referenced from `model_registry/`
- Opta is a curated external input in `curated_inputs/opta/`

## Dashboard

Generated dashboard files are offline HTML outputs:

- `outputs/dashboard/index.html`
- `outputs/validation_dashboard/index.html`

The unified dashboard implementation lives mainly in:

- `src/quiniela/ui/dashboard.py`
- `src/quiniela/ui/dashboard_template.html`
- `docs/dashboard_reference.md`

Edit source/template files, then regenerate dashboards. Do not edit generated HTML directly.
