# Codex Instructions - Quiniela2026

This repository is prepared for public code/dashboard publication for World Cup 2026 quiniela predictions.
Friend quiniela picks are allowed to be public; local secrets and source links must stay out of the repository.

## Read First

When starting a new Codex session, read these files before changing code:

1. `README.md`
2. `docs/collaborator_onboarding.md`
3. `memory/MEMORY.md`
4. `memory/project_status.md`
5. `memory/dashboard_status.md`
6. `docs/knowledge/000_index.md`
7. Latest operational handoff when present, currently `docs/knowledge/039_handoff_dashboard_y_operacion_2026_live.md`
8. Publication/security policy, currently `docs/knowledge/040_publicacion_publica_dashboard_privado.md`
9. Relevant files in `docs/knowledge/`

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

- `data/` generated contents, except explicit public UI artifacts such as `data/ui/prediction_overrides.json` and `data/ui/friends_quinielas.json`
- `outputs/` generated dashboards
- local checkpoints, folds, tuning runs or logs
- `.env`
- `configs/*.local.json`
- `.claude/settings.local.json`

The `.gitignore` keeps folder skeletons via `.gitkeep`.

## Public Security Policy

- `data/ui/friends_quinielas.json` is allowed to be public.
- The Google Sheets URL/ID for friends is private and must stay in environment variables or `configs/friends_sheet.local.json`.
- Before publishing, run:

```powershell
python scripts\check_public_dashboard.py docs\index.html
python scripts\security_scan_publish.py
```

## Current Operational Defaults

- Default quiniela model: `weighted_points_ensemble`
- Once 2026 real results exist, operational picks use the live 2026 best-performing model from frozen pre-match predictions.
- `baseline_poisson` is currently disabled.
- `similar_match_knn_scoreline` is active as an experimental standalone model, excluded from ensembles and automatic preferred-pick selection.
- Backtest validation years: 2018 and 2022
- Published neural artifacts are referenced from `model_registry/`
- Opta is a curated external input in `curated_inputs/opta/`

## Dashboard

Generated dashboard files are offline HTML outputs:

- `docs/index.html` (public dashboard output, includes `DATA.friends` when available)
- `outputs/dashboard/index.html`
- `outputs/validation_dashboard/index.html`

By default `python scripts\generate_dashboard.py` creates the public dashboard with friends data.
Use `python scripts\generate_dashboard.py --exclude-friends` only when an intentionally friend-free
artifact is needed.

The unified dashboard implementation lives mainly in:

- `src/quiniela/ui/dashboard.py`
- `src/quiniela/ui/dashboard_template.html`
- `docs/dashboard_reference.md`

Edit source/template files, then regenerate dashboards. Do not edit generated HTML directly.
