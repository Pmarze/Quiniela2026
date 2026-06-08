# 003 - Politica de conocimientos

## Conocimiento

El proyecto debe conservar una memoria incremental en Markdown para decisiones, lecciones y hechos operativos importantes.

## Regla principal

Cuando se aprenda algo nuevo:

1. Crear una nota nueva con el siguiente numero disponible.
2. Registrar el hecho, la decision y el impacto practico.
3. Enlazar la nota desde `docs/knowledge/000_index.md`.

## Si hay contradiccion

Si una nueva decision contradice una nota anterior:

1. No modificar silenciosamente la nota anterior.
2. Explicar al usuario que nota seria afectada.
3. Pedir aprobacion antes de actualizarla o marcarla como reemplazada.
4. Una vez aprobado, documentar el cambio y dejar claro que conocimiento queda vigente.

## Ejemplo

Si luego se decide cambiar el runtime principal de Conda a Docker, se debe mencionar explicitamente que afectaria:

```text
docs/knowledge/001_runtime_conda.md
docs/runtime.md
docs/data_storage.md
```

Y esperar aprobacion antes de hacer el cambio.

## Estado

Activo. Esta regla tiene prioridad para la gestion futura de conocimiento del proyecto.

