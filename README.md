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

Para recibir una frase compuesta con operaciones y relaciones explícitas:

```python
from interpretacion import interpretar_operaciones_con_relaciones

resultado = interpretar_operaciones_con_relaciones(
    "Compré mercadería por 100 dólares y aboné 30 dólares de esa compra"
)
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

`interpretar_operaciones_con_relaciones` devuelve `texto`, `estado`,
`operaciones`, `relaciones`, `advertencias` y `ambiguedades`. Cada operación
conserva los campos actuales y añade un `id` local a la respuesta, además de
`campos`, que anota para cada dato su valor, evidencia, origen y estado:
`IDENTIFICADO`, `INFERIDO`, `AUSENTE`, `NO_INTERPRETADO` o `AMBIGUO`. Las
relaciones conservan índices compatibles y agregan referencias por ID. El
metadato del precio incluye `ambito`: `UNITARIO`, `TOTAL`, `DESCONOCIDO` o
`NO_ESPECIFICADO`; si la frase contiene cantidad e importe sin especificar el
alcance, el contrato conserva el importe y marca la ambigüedad. Una
referencia ambigua se representa con destino `null`; no se interpreta como una
relación confirmada. Los importes originales nunca se redistribuyen.

## Arquitectura

- `interpretacion.py`: carga del modelo, análisis lingüístico, normalización de acciones, extracción de roles, contrato estructurado, relaciones, validación y serialización.
- `entrenamiento.py`: datos anotados, validación de offsets, entrenamiento, guardado, recarga y metadata reproducible.
- `pruebas.py`: evaluación de NER, semántica, operaciones múltiples, relaciones, contrato y conjunto adversarial independiente.
- `custom_ner_model/proyecto_metadata.json`: fecha, parámetros, etiquetas y huella SHA-256 de los datos usados para entrenar.

El NER propone entidades; las reglas semánticas posteriores corrigen estructuras
conocidas como `7 botellas de agua`, normalizan `adquirió` a `comprar` y
distinguen una contraparte (`al proveedor`) de un concepto de gasto.

## Resultados de la última validación

- NER de generalización: `27/27`.
- Entidades adicionales: `5/5`.
- Semántica: `29/29`.
- Operaciones múltiples: `12/12`.
- Relaciones: `7/7`.
- Contrato: `8/8`.
- Evaluación adversarial independiente: `22/22`, con resultados por categoría en `pruebas.py`.

La evaluación independiente es pequeña y usa un conjunto de reglas lingüísticas
acotado; sus porcentajes no deben interpretarse como una estimación de
rendimiento en producción.

## Limitaciones y evolución

La resolución de referencias es deliberadamente acotada: reconoce algunas
referencias textuales a compras y abstiene el enlace si hay más de un destino
posible. No resuelve deudas externas, no calcula saldos ni interpreta errores
ortográficos mediante una puntuación arbitraria. Las posiciones de evidencia
son relativas al segmento de la operación, no a la frase completa.
