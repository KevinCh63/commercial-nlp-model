# Intérprete comercial en español

Componente Python para convertir frases de compras, ventas, gastos y abonos en propuestas estructuradas. Está preparado para importarse desde un futuro backend; no contiene servidor HTTP ni registra movimientos.

## Instalación

Entorno comprobado: Python 3.14.3, spaCy 3.8.16, Pydantic 2.13.5 y modelo base es_core_news_md 3.8.0. Las dependencias directas están fijadas en requirements.txt.

```powershell
py -3.14 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

El modelo personalizado activo está en `custom_ner_model/`, se conserva localmente y no se versiona en Git. Para regenerarlo, ejecute `python entrenamiento.py`. El entrenamiento es una tarea separada y puede rechazar una candidata que no pase las verificaciones.

## Interfaz para el futuro backend

```python
from componente import ComponenteInterprete

# Crear una vez durante el inicio de la aplicación.
motor = ComponenteInterprete()

# Usar esta misma instancia en cada solicitud.
propuesta = motor.interpretar("Pagué 1.250,50 dólares al proveedor")
datos = propuesta.model_dump(mode="json")
texto_json = propuesta.model_dump_json()
```

También se puede pasar `ruta_modelo=` o definir `INTERPRETE_MODELO`. Las rutas relativas se resuelven respecto a la carpeta del proyecto, independientemente del directorio desde el que se ejecute el consumidor. El componente admite cadenas de hasta 10000 caracteres.

La salida pública está definida en `contrato.py` y `schemas/propuesta-v1.schema.json`. Conserva texto original, importes decimales exactos como cadenas, evidencias, relaciones, versiones y motivos de revisión. `VALIDO` describe la lectura lingüística; todas las propuestas mantienen `requiere_revision=true` y `registrable=false`. Los IDs `op_1` son locales a cada respuesta.

No es necesario exportar, comprimir ni construir una distribución para utilizar el componente. La inferencia no importa ni ejecuta el entrenamiento.

## Validación

```powershell
python pruebas.py
python -m unittest discover -s tests -v
python -m herramientas.evaluar_producto
```

Auditoría actual: 262/262 comprobaciones históricas, 13 métodos de prueba del componente y 25/25 casos exploratorios con requisitos verificables. La muestra exploratoria no representa exactitud perfecta para toda entrada comercial. La evaluación de desarrollo del NER conserva dos errores históricos y es independiente de las reglas corregidas.

## Documentación

- `docs/arquitectura.md`: función de cada carpeta y archivo, y flujo de ejecución.
- `docs/auditoria.md`: hallazgos, correcciones, limpieza y límites.
- `docs/limpieza.json`: inventario de rutas obsoletas retiradas.

## Límites

No identifica artículos de un catálogo ni confirma hechos del mundo real. Varios productos sin desglose, importes negativos, monedas indeterminadas y formatos ambiguos requieren revisión. Las devoluciones no están implementadas. Los resultados de evaluaciones se conservan en `resultados_ner/`; el modelo anterior se conserva en `respaldos/`.
