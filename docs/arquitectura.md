# Arquitectura del intérprete

## Estructura

```text
commercial-nlp-model/
├── componente.py
├── interpretacion.py
├── monetario.py
├── revision.py
├── contrato.py
├── custom_ner_model/
├── entrenamiento/
│   ├── __init__.py
│   ├── __main__.py
│   ├── entrenamiento.py
│   └── datos_ner.py
├── tests/
│   ├── evaluacion.py
│   ├── test_componente.py
│   └── test_integracion_local.py
├── herramientas/
│   └── evaluar_producto.py
├── schemas/
│   └── propuesta-v1.schema.json
├── docs/
│   └── arquitectura.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Ejecución

`ComponenteInterprete` carga el modelo activo una vez y valida el tipo y tamaño de la entrada. El análisis de spaCy y las reglas de `interpretacion.py` extraen operaciones y relaciones. `monetario.py` interpreta importes con precisión decimal; `revision.py` reúne los motivos de revisión. Finalmente, `contrato.py` valida la propuesta y sus evidencias.

La inferencia no importa el entrenamiento, las pruebas ni las herramientas de evaluación. El modelo puede seleccionarse con `ruta_modelo` o `INTERPRETE_MODELO`; las rutas relativas se resuelven desde la raíz del proyecto.

## Entrenamiento y evaluación

`python -m entrenamiento` ejecuta el entrenador. `entrenamiento/datos_ner.py` genera ejemplos anotados. El entrenador utiliza las métricas y baterías de `tests/evaluacion.py` para validar las candidatas antes de promoverlas. La evaluación de anotaciones importa las funciones de entrenamiento de manera diferida.

`python -m tests.evaluacion` ejecuta las baterías lingüísticas. `python -m unittest discover -s tests -v` comprueba el componente y su contrato. `python -m herramientas.evaluar_producto` ejecuta la evaluación exploratoria.

## Modelo y archivos generados

`custom_ner_model/` contiene el modelo activo completo de spaCy y se versiona para permitir la ejecución desde un clon del repositorio. Sus archivos internos forman una unidad y no deben reorganizarse manualmente. El entrenamiento puede sustituirlo únicamente después de las verificaciones; cualquier actualización del modelo debe revisarse antes de subirla.

`resultados_ner/` guarda informes y candidatas. `respaldos/` conserva modelos anteriores y archivos históricos locales. Ambas carpetas están excluidas de Git, al igual que entornos virtuales, cachés y configuración personal del editor.
