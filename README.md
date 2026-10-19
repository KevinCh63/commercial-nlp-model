# Intérprete NLP comercial

Intérprete en español basado en spaCy que convierte frases informales de un
pequeño centro comercial en resultados estructurados y serializables a JSON.

## Instalación

```bash
python -m pip install -r requirements.txt
python -m spacy download es_core_news_md
```

Se recomienda Python 3.14.3 y spaCy 3.8.16. El modelo personalizado se genera
localmente en `custom_ner_model/` y está excluido por `.gitignore`.

## Uso

Para entrenar, guardar, recargar y evaluar el modelo:

```bash
python entrenamiento.py
```

Para ejecutar la batería de extracción NER, semántica y los casos críticos:

```bash
python pruebas.py
```

La inferencia no reentrena el modelo. Desde Python:

```python
from interpretacion import interpretar_texto, serializar_resultado

resultado = interpretar_texto("Pagué 10 dólares al proveedor")
print(serializar_resultado(resultado))
```

## Contrato de salida

Todos los resultados incluyen los campos principales:

```json
{
  "texto": "Pagué 10 dólares al proveedor",
  "agente": "YO",
  "contraparte": "proveedor",
  "accion": "pagar",
  "tipo": "EGRESO",
  "cantidad": null,
  "producto": null,
  "concepto": null,
  "precio": "10 dólares",
  "moneda": "USD",
  "estado": "VALIDO",
  "faltantes": [],
  "advertencias": []
}
```

`estado` puede ser `VALIDO`, `PARCIAL`, `AMBIGUO` o `NO_RECONOCIDO`.
Una deuda, por ejemplo, se representa como `accion: "deber"`, `tipo: null` y
una advertencia: deber dinero no demuestra que se haya realizado un pago.
Los valores desconocidos son `null`; el intérprete no inventa agentes,
monedas, productos ni importes.

La salida conserva además las entidades NER y dependencias para depuración.
`serializar_resultado` produce JSON UTF-8 y constituye la interfaz Python que
puede consumir posteriormente un backend Spring Boot sin una integración
ficticia ni un servidor HTTP adicional.

## Arquitectura

- `interpretacion.py`: carga del modelo, análisis lingüístico, normalización de acciones, extracción de roles, validación y serialización.
- `entrenamiento.py`: datos anotados, validación de offsets, entrenamiento, guardado, recarga y metadata reproducible.
- `pruebas.py`: evaluación exacta de NER, evaluación semántica y regresiones críticas.
- `custom_ner_model/proyecto_metadata.json`: fecha, parámetros, etiquetas y huella SHA-256 de los datos usados para entrenar.

El NER propone entidades; las reglas semánticas posteriores corrigen estructuras
conocidas como `7 botellas de agua`, normalizan `adquirió` a `comprar` y
distinguen una contraparte (`al proveedor`) de un concepto de gasto.

## Resultados actuales

Con el modelo reentrenado en los datos actuales:

- NER de entrenamiento: `45/45`.
- NER de generalización: `27/27`.
- Pruebas semánticas: `18/18`.

Estas métricas proceden de los comandos anteriores y deben volver a medirse
después de modificar los datos o la configuración de entrenamiento. El
conjunto sigue siendo pequeño; conviene ampliarlo con ejemplos reales antes de
considerar las métricas concluyentes.

## Limitaciones y evolución

Todavía no se extraen todas las posibles contrapartes ni se separan varias
operaciones en una sola frase. Los errores ortográficos y la confianza no se
resuelven con una puntuación arbitraria. La siguiente evolución recomendada es
ampliar un conjunto de evaluación independiente y añadir pruebas de frases
ambiguas y de múltiples operaciones antes de exponer la interfaz a Spring Boot.
