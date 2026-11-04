# Intérprete de operaciones comerciales en español

Algoritmo de procesamiento de lenguaje natural que transforma descripciones de compras, ventas, gastos, pagos y abonos en propuestas estructuradas. Identifica acciones, participantes, productos, cantidades e importes, y señala información incompleta o ambigua para facilitar su revisión.

## Idea principal

Una descripción comercial contiene información tanto explícita como implícita. En «Compré arroz por 20 dólares», el producto y el importe aparecen en el texto, mientras que el agente se puede inferir de la conjugación del verbo.

El algoritmo combina un modelo lingüístico de **spaCy**, un reconocedor de entidades personalizado (**NER**) y reglas de interpretación comercial. El modelo localiza información relevante; las reglas organizan esos datos, interpretan el contexto y detectan situaciones que requieren revisión.

El resultado es una propuesta de interpretación con evidencias del texto original. No constituye una confirmación de que la operación haya ocurrido ni un registro contable.

## Cómo funciona

1. **Análisis lingüístico.** Divide el texto en tokens y analiza lemas, rasgos morfológicos, dependencias sintácticas y entidades. El NER personalizado reconoce cantidades (`QUANTITY`), productos (`PRODUCT`), precios (`PRICE`) y conceptos (`CONCEPT`).
2. **Detección de acciones y contexto.** Reconoce verbos comerciales, normaliza acciones equivalentes e identifica al agente cuando la información lingüística lo permite. Examina negaciones, intenciones e incertidumbre; la extracción de operaciones conserva las cláusulas afirmadas.
3. **Separación de operaciones.** Cuando hay varias acciones, segmenta el texto y analiza cada operación, conservando las posiciones de sus evidencias respecto al texto original.
4. **Asignación de roles comerciales.** Determina acción, dirección del movimiento (`INGRESO` o `EGRESO`), contraparte, cantidad, producto y concepto. La dirección depende también del contexto: pagar y recibir un pago describen movimientos distintos.
5. **Interpretación monetaria.** Localiza importes, analiza separadores y monedas, y normaliza valores mediante `Decimal`. Distingue el precio unitario del total cuando existe evidencia suficiente y señala formatos ambiguos.
6. **Relaciones entre operaciones.** Busca referencias explícitas que vinculen pagos o abonos con compras anteriores. Si no existe un destino único, conserva la relación como ambigua.
7. **Revisión y validación.** Reúne faltantes, advertencias y motivos de revisión, y construye una salida tipada mediante Pydantic.

```text
Texto en español
    → Análisis lingüístico y entidades
    → Acciones, contexto y segmentación
    → Roles comerciales e importes
    → Relaciones y revisión
    → Propuesta estructurada
```

## Información del resultado

La propuesta conserva el texto original y contiene:

| Elemento | Descripción |
|---|---|
| `estado` | Estado general de la interpretación. |
| `operaciones` | Acciones detectadas con sus participantes, productos, cantidades, importes y faltantes. |
| `relaciones` | Vínculos de tipo `PAGO_DE` o `ABONO_DE`, confirmados o ambiguos. |
| `revision` | Indicadores y motivos para revisar la propuesta. |
| `advertencias` | Observaciones relevantes para interpretar la salida. |
| `versiones` | Identificación del componente, modelo, reglas y esquema utilizados. |

Cada operación incluye evidencias con el fragmento de texto y sus posiciones de inicio y fin. Los importes normalizados se serializan como cadenas para conservar su precisión decimal. Los identificadores como `op_1` son locales a cada propuesta.

### Estados de interpretación

| Estado | Significado |
|---|---|
| `VALIDO` | La lectura lingüística cumple los criterios de interpretación. |
| `PARCIAL` | Se reconoce información comercial, pero faltan datos o hay referencias genéricas. |
| `AMBIGUO` | Existen datos o relaciones que no pueden resolverse de forma única. |
| `NO_RECONOCIDO` | No se obtiene una operación comercial interpretable. |

Todas las propuestas mantienen `requiere_revision=true` y `registrable=false`, incluso cuando su estado es `VALIDO`.

La definición completa de la salida se encuentra en [`contrato.py`](contrato.py) y en [`schemas/propuesta-v1.schema.json`](schemas/propuesta-v1.schema.json).

## Instalación

Entorno de referencia documentado: Python 3.14.3, spaCy 3.8.16, Pydantic 2.13.5 y modelo base `es_core_news_md` 3.8.0. Las dependencias directas están fijadas en `requirements.txt`.

```powershell
py -3.14 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

El modelo personalizado se carga desde `custom_ner_model/`. Esta carpeta está incluida en el repositorio para permitir la ejecución tras instalar las dependencias. Los informes, modelos candidatos y respaldos se generan localmente y quedan excluidos de Git.

## Uso

```python
from componente import ComponenteInterprete

motor = ComponenteInterprete()
propuesta = motor.interpretar("Pagué 1.250,50 dólares al proveedor")

print(propuesta.estado)
for operacion in propuesta.operaciones:
    print(operacion.accion, operacion.importe.valor, operacion.importe.moneda)

# Obtener un diccionario o una representación JSON.
datos = propuesta.model_dump(mode="json")
texto_json = propuesta.model_dump_json()
```

Conviene reutilizar la instancia para evitar cargar el modelo en cada interpretación. Se admiten cadenas de hasta 10000 caracteres.

La ubicación del modelo puede configurarse con `ComponenteInterprete(ruta_modelo=...)` o con la variable de entorno `INTERPRETE_MODELO`. Las rutas relativas se resuelven respecto a la carpeta del proyecto.

## Entrenamiento del modelo

```powershell
.\venv\Scripts\python.exe -m entrenamiento
```

El entrenamiento utiliza el modelo base y los ejemplos anotados para generar una candidata del NER personalizado. Valida las anotaciones, compara resultados y condiciona la promoción de la candidata a las verificaciones establecidas. Una candidata puede ser rechazada; los modelos reemplazados se conservan en `respaldos/`.

El entrenamiento es un proceso separado de la interpretación. Ejecutar el intérprete no entrena ni modifica el modelo.

## Evaluación

```powershell
.\venv\Scripts\python.exe -m tests.evaluacion
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m herramientas.evaluar_producto
```

Las evaluaciones cubren reconocimiento de entidades, interpretación semántica, modalidad, relaciones, importes, evidencias y estructura de la salida. Los informes se conservan en `resultados_ner/`.

## Organización del proyecto

| Archivo | Responsabilidad |
|---|---|
| `componente.py` | Interfaz de uso, carga del modelo y control de entradas. |
| `interpretacion.py` | Análisis lingüístico, roles comerciales, segmentación y relaciones. |
| `monetario.py` | Detección y normalización de importes y monedas. |
| `revision.py` | Criterios y motivos de revisión. |
| `contrato.py` | Modelos de datos y validación de la propuesta. |
| `entrenamiento/entrenamiento.py` | Entrenamiento, evaluación y promoción de candidatas. |
| `entrenamiento/datos_ner.py` | Generación de ejemplos complementarios anotados. |
| `tests/` y `herramientas/` | Baterías automatizadas y utilidades de evaluación. |

## Alcance y limitaciones

- Interpreta descripciones comerciales en español dentro de los patrones contemplados por el modelo y las reglas.
- No identifica artículos de un catálogo ni verifica hechos del mundo real.
- Los productos genéricos, las cantidades vagas y varios artículos sin desglose pueden impedir una interpretación completa.
- Los importes negativos, las monedas indeterminadas y los formatos numéricos ambiguos requieren revisión.
- No debe asumirse que un precio es unitario o total cuando el texto no lo especifica.
- Las devoluciones no están implementadas.

## Documentación complementaria

- [`docs/arquitectura.md`](docs/arquitectura.md): estructura interna y flujo de ejecución.

El modelo incluido deriva de `es_core_news_md` de Explosion. Su licencia y los avisos de sus fuentes se conservan en `custom_ner_model/LICENSE` y `custom_ner_model/LICENSES_SOURCES`. El código y los datos de entrenamiento personalizados están disponibles en `entrenamiento/`.
