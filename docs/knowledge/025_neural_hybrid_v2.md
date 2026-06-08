# 025 - Neural Hybrid V2

## Conocimiento

Se implementa `neural_hybrid_v2` como segundo modelo de red neuronal, separado del modelo base `neural_scoreline_mlp`. El modelo base se mantiene como candidato estable en dashboards y resultados; las mejoras arquitectonicas nuevas no deben sobrescribir sus artefactos.

## Regla temporal prioritaria

La quiniela se llena un dia antes. Por lo tanto, `neural_hybrid_v2` usa estrategia `previous_day`: para partidos del dia `D`, ningun resultado del mismo dia puede entrar como feature, aunque haya ocurrido antes. Durante el Mundial 2026, los partidos completados se incorporan al contexto de entrenamiento/prediccion a partir del dia siguiente.

## Implementacion

- `configs/neural_hybrid_v2.yaml`
- `src/quiniela/features/hybrid_features.py`
- `src/quiniela/models/neural_hybrid_v2.py`
- `src/quiniela/training/neural_hybrid_trainer.py`
- `src/quiniela/training/neural_hybrid_tuner.py`
- `scripts/train_neural_hybrid_v2.py`
- `scripts/tune_neural_hybrid_v2.py`
- `docs/neural_hybrid_v2.md`

## Estado

El modelo queda registrado pero inactivo en `configs/models.yaml` hasta que exista un artefacto entrenado en `data/models/neural_hybrid_v2/latest`.

## Comandos con conda activo

```powershell
python scripts\tune_neural_hybrid_v2.py --device cuda --max-trials 6
python scripts\tune_neural_hybrid_v2.py --device cuda --max-trials 48
python scripts\train_neural_hybrid_v2.py --config data\models\neural_hybrid_v2_tuning\best_config.json --output-root data\models\neural_hybrid_v2 --device cuda --fresh
```
