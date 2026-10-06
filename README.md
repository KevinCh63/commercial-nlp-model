# Commercial NLP Model

Modelo NLP basado en spaCy para interpretar operaciones comerciales expresadas en lenguaje natural y convertirlas en información estructurada.

Por ejemplo, ante la frase:

> “vendí 5 camisetas por 30 dólares”

el sistema busca obtener una interpretación parecida a:

```text
acción: vender
tipo: INGRESO
cantidad: 5
producto: camisetas
precio: 30 dólares
```

## Tecnologías

- Python
- spaCy
- `es_core_news_md`
- Named Entity Recognition (NER)
- Análisis morfológico
- Dependencias sintácticas

Las entidades reconocidas actualmente son:

- `QUANTITY`: cantidad asociada a la operación.
- `PRODUCT`: producto mencionado.
- `PRICE`: precio, importe o valor monetario.
- `CONCEPT`: concepto o motivo de un gasto.

Además, el sistema intenta determinar el agente, la acción, el tipo de operación, la cantidad, el producto, el concepto y el precio.

## Instalación

Se requiere Python y spaCy 3.8.16. Instala las dependencias con:

```bash
python -m pip install -r requirements.txt
python -m spacy download es_core_news_md
```

El modelo `es_core_news_md` debe estar instalado antes de ejecutar el entrenamiento.

## Uso

Ejecuta el entrenamiento y la validación con:

```bash
python entrenamiento.py
```

Este comando genera `custom_ner_model/` localmente. El modelo entrenado es un artefacto generado y está excluido del repositorio inicialmente.

## Estructura del proyecto

- `entrenamiento.py`: datos, validación, entrenamiento, guardado, recarga y evaluación del modelo.
- `pruebas.py`: pruebas nuevas del NER y función de evaluación.
- `interpretacion.py`: interpretación semántica, análisis morfológico y dependencias sintácticas.
- `requirements.txt`: dependencia de spaCy fijada para reproducibilidad.

## Estado del proyecto

El proyecto se encuentra en desarrollo.
