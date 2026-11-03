# Arquitectura del proyecto

## Estructura mantenida

```text
Algoritmo Python/
├── componente.py
├── contrato.py
├── interpretacion.py
├── monetario.py
├── revision.py
├── entrenamiento.py
├── datos_ner.py
├── pruebas.py
├── requirements.txt
├── README.md
├── .gitignore
├── custom_ner_model/
├── schemas/
│   └── propuesta-v1.schema.json
├── tests/
│   ├── __init__.py
│   ├── test_componente.py
│   └── test_integracion_local.py
├── herramientas/
│   ├── __init__.py
│   └── evaluar_producto.py
├── resultados_ner/
├── docs/
│   ├── arquitectura.md
│   ├── auditoria.md
│   └── limpieza.json
└── respaldos/
    └── modelo_previo/
```

La raíz contiene los módulos del intérprete y sus tareas principales. Se mantuvieron las rutas del modelo y los módulos analíticos para conservar los consumidores existentes. La separación se realiza por responsabilidad y dependencias: ejecutar el componente no carga el entrenamiento, sus ejemplos ni las herramientas de evaluación.

## Archivos de la raíz

| Archivo | Responsabilidad |
|---|---|
| `componente.py` | Punto de entrada recomendado para el futuro backend. Carga una instancia spaCy, limita entradas, serializa llamadas concurrentes y devuelve propuestas tipadas. Permite configurar la ruta del modelo. Expone copias de sus versiones y calcula huellas del modelo y reglas. No depende de archivos de distribución. |
| `contrato.py` | Modelos Pydantic del contrato público: propuesta, operación, importe, evidencia, relación, revisión y versiones. Rechaza campos adicionales, conserva importes como cadenas y verifica las evidencias al construir una propuesta. |
| `interpretacion.py` | Motor lingüístico: carga analítica, acciones, agentes, modalidad, roles comerciales, segmentación, relaciones y evidencia. Conserva la interfaz anterior para análisis y depuración. |
| `monetario.py` | Localiza importes completos y los normaliza con Decimal. Mantiene originales, signos y monedas; se abstiene ante separadores ambiguos o formatos no admitidos. |
| `revision.py` | Reglas de revisión de la propuesta: moneda, cantidad, importe, varios artículos y acciones fuera de alcance. No confirma ni persiste movimientos. |
| `entrenamiento.py` | Corpus original/complementario, validación de anotaciones, entrenamiento NER, comparación de candidatas y promoción condicionada a las baterías. Los modelos reemplazados se guardan en respaldos/. |
| `datos_ner.py` | Generación determinista de familias adicionales de entrenamiento y sus offsets. No importa casos de las pruebas. |
| `pruebas.py` | Datos de evaluación históricos, de desarrollo y finales, y las 262 comprobaciones de NER, semántica, modalidad, relaciones y contrato. También aporta las métricas utilizadas por el entrenador; no se debe importar desde el consumidor de producción. |
| `requirements.txt` | Dependencias directas fijadas: spaCy, Pydantic y el modelo base con hash. El modelo base sirve para entrenar y evaluar; el componente carga el personalizado. |
| `README.md` | Instalación, uso, comandos de verificación y límites principales. |
| `.gitignore` | Excluye entornos, cachés, candidatos generados, el modelo activo y respaldos locales. |

## Carpetas de código y documentación

| Carpeta/archivo | Significado |
|---|---|
| `tests/` | Pruebas automatizadas del componente que no forman parte de la inferencia. |
| `tests/__init__.py` | Identifica tests como módulo Python. |
| `tests/test_componente.py` | Nueve métodos de prueba con subcasos de importes, abstención, signos, producto, contrato, revisión, evidencias y concurrencia. |
| `tests/test_integracion_local.py` | Cuatro métodos de prueba: esquema, modelo ausente, carga desde otro directorio/versiones protegidas y evidencias fuera del texto. |
| `herramientas/` | Utilidades manuales de evaluación, separadas del motor. |
| `herramientas/__init__.py` | Identifica herramientas como módulo Python. |
| `herramientas/evaluar_producto.py` | Contiene los 30 casos exploratorios, 25 con expectativas verificables; mide diferencias, evidencias, repetición y latencia. Guarda resultados en resultados_ner/. |
| `schemas/` | Contratos legibles por otras aplicaciones y tecnologías. |
| `schemas/propuesta-v1.schema.json` | Representación JSON Schema del contrato público v1. Una prueba verifica su correspondencia con los modelos Pydantic. |
| `docs/` | Documentación técnica vigente y trazabilidad de esta limpieza. |
| `docs/arquitectura.md` | Este inventario de arquitectura y flujo de ejecución. |
| `docs/auditoria.md` | Hallazgos de auditoría, correcciones y verificaciones. |
| `docs/limpieza.json` | Relación de las rutas antiguas retiradas y ubicación del respaldo conservado. |

## Datos generados

`resultados_ner/` contiene evidencia, no código ejecutable ni dependencias de la inferencia:

- `auditoria_baterias.log`: salida de las 262 comprobaciones históricas.
- `auditoria_tests.log`: salida de los 13 métodos de prueba actuales.
- `auditoria_producto.log`: salida legible de la exploración de producto.
- `evaluacion_producto_actual.json`: salidas detalladas, diferencias, evidencias y tiempos de la última exploración.
- `validacion_final_2026-11-03.json`: referencia histórica de la evaluación del NER promovido, con sus métricas por etiqueta; no se presenta como una nueva evaluación a ciegas.

Entrenar puede añadir una carpeta `candidata_temporal_actual/` y un informe `comparacion_actual.json` en esa carpeta de resultados. Son artefactos generados que sirven para revisar la candidata; no intervienen en una inferencia normal.

`respaldos/` conserva modelos anteriores para recuperación. `modelo_previo/` es el respaldo existente, trasladado fuera de la raíz. Una futura promoción crea allí carpetas con nombre de modelo y fecha. No se elimina ni se activa un respaldo automáticamente al interpretar.

## Interior de custom_ner_model/

Es el modelo activo de spaCy, guardado con su formato estándar. Sus archivos binarios son necesarios y no deben reorganizarse manualmente.

| Ruta | Contenido |
|---|---|
| `config.cfg` | Configuración del pipeline, arquitectura y componentes. |
| `meta.json` | Metadatos estándar del modelo spaCy. |
| `proyecto_metadata.json` | Procedencia del entrenamiento, corpus, parámetros y validación de la promoción. Es una referencia del modelo, no una nueva validación de todas las reglas actuales. |
| `tokenizer` | Reglas serializadas para dividir el texto en tokens. |
| `attribute_ruler/patterns` | Patrones para ajustar atributos lingüísticos de tokens. |
| `lemmatizer/lookups/lookups.bin` | Tablas de lematización. |
| `morphologizer/cfg` y `model` | Configuración y pesos del análisis morfológico. |
| `ner/cfg`, `model` y `moves` | Configuración, pesos y transiciones del reconocimiento de entidades. |
| `parser/cfg`, `model` y `moves` | Configuración, pesos y transiciones del analizador de dependencias. |
| `senter/cfg` y `model` | Componente de segmentación de oraciones conservado por spaCy; puede existir en disco sin estar habilitado en el pipeline activo. |
| `tok2vec/cfg` y `model` | Representación de tokens utilizada por componentes lingüísticos. |
| `vocab/strings.json` | Inventario de cadenas e identificadores. |
| `vocab/vectors` y `vectors.cfg` | Vectores y su configuración. |
| `vocab/key2row` | Mapeo entre palabras y filas de vectores. |
| `vocab/lookups.bin` | Tablas auxiliares del vocabulario. |

## Carpetas locales auxiliares

`.git/` es el historial del repositorio y no se modificó. `.idea/` contiene la configuración personal del IDE y se conservó. `venv/` contiene el entorno de desarrollo del usuario y sus bibliotecas; no se confundió con el entorno temporal de validación, que se eliminó. `__pycache__/` puede reaparecer al ejecutar Python y se excluye de Git; es una caché regenerable, no código del proyecto.

## Flujo de inferencia para el futuro backend

```text
Código consumidor
  → ComponenteInterprete.interpretar(texto)
  → spaCy + interpretacion.py
  → monetario.py: valores exactos y formatos
  → revision.py: motivos de revisión
  → contrato.py: propuesta validada
  → consumidor recibe datos o JSON
```

El consumidor creará una instancia al iniciar su aplicación y la reutilizará. El entrenamiento y las pruebas son procesos separados. No hay dependencia de ZIP, manifiestos de distribución, servidor HTTP ni base de datos. La confirmación y persistencia serán responsabilidades del backend futuro, sin convertir automáticamente un estado lingüístico VALIDO en un movimiento registrado.
