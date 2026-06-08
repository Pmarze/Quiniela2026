# 037 - Compartir datos y modelos entrenados

## Contexto

El usuario quiere compartir el repositorio privado con un amigo para que ambos puedan revisar resultados, crear modelos y empujar cambios periodicamente.

Esto contradice parcialmente la nota 036, donde se habia decidido no subir artefactos generados.

## Decision

Se cambia la politica del repositorio:

- Si se versionan resultados, dashboards, base SQLite, predicciones y modelos entrenados.
- Se usa Git LFS para archivos grandes: `*.pt`, `*.pth`, `*.ckpt`, `*.db`.
- Se mantiene fuera `data/models/neural_hybrid_v2_tuning/`, porque pesa aproximadamente 37 GB.

El set compartido sin ese tuning gigante pesa aproximadamente 1.16 GB.

## Referencia GitHub

Segun GitHub Docs, Git LFS almacena referencias en el repositorio y los archivos reales fuera del Git normal. En GitHub Free/Pro, el limite incluido actual es 10 GiB de storage y 10 GiB de bandwidth mensual para LFS. Si se excede el presupuesto, los pushes o descargas LFS pueden bloquearse o generar cobro segun la configuracion de billing.

## Estado

Reemplazado por la nota 038. La idea de compartir resultados sigue vigente, pero no mediante `data/` completo. La politica actual publica solo modelos finales y metricas asociadas en `model_registry/`; los datos descargables y ejecuciones locales se reconstruyen o permanecen en cada computadora.
