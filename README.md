# Intérprete NLP comercial

Intérprete en español basado en spaCy que convierte frases informales de un
pequeño centro comercial en resultados estructurados y serializables a JSON.

## Instalación

```bash
python -m pip install -r requirements.txt
python -m spacy validate
```

Entorno comprobado en esta revisión: Python 3.14.3, spaCy 3.8.16 y
`es_core_news_md` 3.8.0. Las dos dependencias directas están fijadas en
`requirements.txt`, incluido el hash SHA-256 del modelo base. El modelo
personalizado se genera localmente en `custom_ner_model/` y se excluye de Git;
se regenera desde los datos fuente y no se versiona porque es un artefacto
derivado de tamaño considerable.

Preparación en Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m spacy validate
```

En macOS o Linux, use `.venv/bin/python`. El modelo base se instala desde el
wheel fijado por `requirements.txt`; no hace falta descargar una versión
flotante aparte.

## Uso

Para validar los datos, entrenar una candidata y evaluarla contra el modelo
vigente:

```bash
python entrenamiento.py
```

El entrenador compara desarrollo NER y cada grupo de la matriz histórica
completa antes de promover. Rechaza una candidata si alguna batería retrocede;
la carga desde un artefacto temporal mediante el cargador real del intérprete y
conserva el modelo anterior como respaldo. El conjunto final reservado no se
usa para seleccionar la candidata.

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

El cargador resuelve `custom_ner_model/` relativo al archivo Python y falla con
`RuntimeError` si no existe o no puede abrirse; no sustituye silenciosamente el
modelo personalizado por el base. `cargar_modelo("es_core_news_md")` permite
solicitar explícitamente el modelo base. Las funciones analíticas
`interpretar_texto` e `interpretar_operaciones` lanzan `ValueError` para texto
vacío/no textual y para entradas mayores que `nlp.max_length`. La función
estructurada devuelve `NO_RECONOCIDO` para entrada vacía o de tipo incorrecto.

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
`serializar_resultado` produce JSON UTF-8 para consumidores Python.

`interpretar_operaciones_con_relaciones` devuelve `texto`, `estado`,
`operaciones`, `relaciones`, `diagnosticos_ner`, `advertencias` y `ambiguedades`. Cada operación
conserva los campos actuales y añade un `id` local a la respuesta, además de
`campos`, que anota para cada dato su valor, evidencia, origen y estado:
`IDENTIFICADO`, `INFERIDO`, `AUSENTE`, `NO_INTERPRETADO` o `AMBIGUO`. Las
relaciones conservan índices compatibles y agregan referencias por ID. El
metadato del precio incluye `ambito`: `UNITARIO`, `TOTAL`, `DESCONOCIDO` o
`NO_ESPECIFICADO`; si la frase contiene cantidad e importe sin especificar el
alcance, el contrato conserva el importe y marca la ambigüedad. Una
referencia ambigua se representa con destino `null`; no se interpreta como una
relación confirmada. Los importes originales nunca se redistribuyen.

La firma pública es `interpretar_operaciones_con_relaciones(texto, nlp=None)`;
acepta una cadena Unicode y devuelve un `dict` independiente de la consola con
estas claves superiores:

- `texto`: entrada original, o `null` cuando el tipo de entrada no era texto.
- `estado`: `VALIDO`, `PARCIAL`, `AMBIGUO` o `NO_RECONOCIDO`.
- `operaciones`: lista ordenada que puede estar vacía.
- `relaciones`: lista de vínculos con origen/destino por índice y por ID.
- `diagnosticos_ner`: spans completos proyectados de forma segura y conflictos
  observados al segmentar; conserva offsets del texto original y especifica
  cuándo una proyección se omitió.
- `advertencias` y `ambiguedades`: listas de diagnósticos.

Cada operación conserva `id` (único en la respuesta), `texto`, `agente`,
`contraparte`, `modalidad`, `accion`, `tipo`, `cantidad`, `producto`,
`concepto`, `precio`, `moneda`, `estado`, `faltantes`, `advertencias`,
`entidades`, `dependencias` y `campos`. El valor de un campo ausente es `null`;
el metadato distingue `IDENTIFICADO`, `INFERIDO`, `AUSENTE`,
`NO_INTERPRETADO` y `AMBIGUO`. Las modalidades son `AFIRMADA`, `NEGADA`,
`INTENCION` e `INCIERTA`; las operaciones admitidas son las cláusulas que el
analizador reconoce como afirmadas y realizadas. Esto interpreta el texto, no
certifica que la transacción sucediera en el mundo real.

Una relación contiene `operacion_origen`, `operacion_destino`, sus IDs, `tipo`,
`evidencia` y `estado` (`CONFIRMADA` o `AMBIGUA`). Una referencia ambigua tiene
destino `null`. Los offsets de evidencia son índices de caracteres del texto
original: inicio inclusivo, fin exclusivo. Errores de carga del modelo se
propagan como `RuntimeError`; texto vacío o de tipo incorrecto en la API
estructurada devuelve `NO_RECONOCIDO`; las funciones analíticas de una frase
lanzan `ValueError` para esos casos. El límite de longitud es `nlp.max_length`.
Los índices de operaciones en relaciones comienzan en cero; los IDs comienzan
en `op_1` para cada respuesta.

Al dividir una oración con varias acciones, el intérprete conserva las
entidades locales del segmento y solo agrega una entidad de la oración completa
si el span está íntegramente dentro del segmento, se alinea a tokens, no se
solapa con una predicción local y el segmento ya contiene otra entidad local
como ancla. Los spans que cruzan límites, no se alinean o no tienen ancla no se
transfieren; se registran en `diagnosticos_ner`. La proyección no cambia la
admisión por modalidad ni decide relaciones entre operaciones.

La resolución actual se limita a referencias explícitas a compras,
adquisiciones y algunas deudas. Referencias como `del primer pago` no están
resueltas y no deben inferirse.

Ejemplo de contrato ejecutable:

```python
salida = interpretar_operaciones_con_relaciones(
    "Compré papel y aboné 2 dólares de esa compra"
)
assert [op["id"] for op in salida["operaciones"]] == ["op_1", "op_2"]
assert salida["relaciones"][0]["estado"] == "CONFIRMADA"
assert salida["relaciones"][0]["operacion_destino_id"] == "op_1"
```

El campo `producto` conserva el texto reconocido sin consultar un catálogo.
`campos.producto.interpretacion` distingue `GENERICA` (por ejemplo,
`producto`, `artículos` o `mercadería`) de `TEXTO_ESPECIFICO`; este último
describe el texto, no certifica que exista un artículo de inventario con ese
nombre. Una unidad como `cajas` no se convierte por sí sola en producto.

En las relaciones, una referencia explícita que no identifica un único destino
se devuelve con `estado: "AMBIGUA"` y destino `null`. Si no hay referencia
textual suficiente, `relaciones` no incluye una relación ficticia. La expresión
`la compra anterior` identifica la compra precedente inmediata entre las
compras anteriores de esa frase; la mera cercanía entre una compra y un abono
no crea un vínculo. Esta regla no resuelve deudas externas.

`VALIDO` significa que el intérprete pudo construir una lectura estructural sin
faltantes ni ambigüedades que haya detectado. No significa que los datos estén
completos para el negocio ni que puedan registrarse sin validación de la
aplicación. `PARCIAL`, `AMBIGUO` y `NO_RECONOCIDO` expresan respectivamente
campos requeridos ausentes, incertidumbre detectada y falta de una operación
reconocible.

Los offsets de `campos.*.evidencia` son posiciones de caracteres relativas al
texto original completo, con inicio inclusivo y fin exclusivo. El fragmento
señalado reproduce la evidencia superficial; puede diferir del valor
normalizado, por ejemplo, la forma verbal `adquirí` respalda la acción
normalizada `comprar`. Las entidades NER y dependencias que se conservan para
depuración siguen siendo relativas a su segmento.

Las funciones `interpretar_texto` e `interpretar_operaciones` requieren una
cadena no vacía y lanzan `ValueError` para `""` o texto compuesto solo por
espacios. La envoltura `interpretar_operaciones_con_relaciones` representa esos
casos mediante `estado: "NO_RECONOCIDO"`, operaciones y relaciones vacías.
Texto sin verbo comercial reconocido también produce `NO_RECONOCIDO`; un verbo
reconocido con datos faltantes se representa como operación `PARCIAL`.

`interpretar_texto` es una lectura lingüística de una frase y conserva el campo
`modalidad` (`AFIRMADA`, `NEGADA`, `INTENCION` o `INCIERTA`) para análisis. La
función de admisión estructurada `interpretar_operaciones_con_relaciones`
devuelve como operaciones únicamente cláusulas reconocidas como afirmadas y
ejecutadas. Omite negaciones, deseos, posibilidades, hipótesis y formas futuras;
si una frase contiene una cláusula incierta y otra afirmada, puede conservar
solo la afirmada. Esto no verifica que la transacción haya ocurrido en el mundo
real: esa validación pertenece a la aplicación u otros mecanismos.

Las referencias genéricas a productos se conservan como texto y se anotan como
`GENERICA`; no se validan contra catálogo. Una cantidad vaga como `varios` se
conserva literalmente y su metadato queda `AMBIGUO`, porque no determina una
cantidad numérica. Monedas reconocidas explícitamente se normalizan a `USD`,
`EUR` o `PESO`; no se realizan conversiones. Una moneda no reconocida queda
como `null`.

Un importe en `por caja` o `por unidad` se marca como `UNITARIO`; `en total`
se marca como `TOTAL`. El intérprete conserva el importe expresado y no
calcula un total multiplicando cantidad por precio unitario.

## Arquitectura

- `interpretacion.py`: carga del modelo, análisis lingüístico, normalización de acciones, extracción de roles, contrato estructurado, relaciones, validación y serialización.
- `entrenamiento.py`: datos anotados, validación de offsets, entrenamiento, guardado, recarga y metadata reproducible.
- `pruebas.py`: evaluación de NER, semántica, operaciones múltiples, relaciones, contrato y conjunto adversarial independiente.
- `custom_ner_model/proyecto_metadata.json`: fecha, parámetros, etiquetas y huella SHA-256 de los datos usados para entrenar.

El NER propone entidades; las reglas semánticas posteriores corrigen estructuras
conocidas como `7 botellas de agua`, normalizan `adquirió` a `comprar` y
distinguen una contraparte (`al proveedor`) de un concepto de gasto.

## Resultados de pruebas y límites de evaluación

- Batería NER histórica: `27/27`.
- Entidades adicionales: `5/5`.
- Semántica: `29/29`.
- Operaciones múltiples: `12/12`.
- Relaciones: `7/7`.
- Contrato: `8/8`.
- Evaluación adversarial independiente: `22/22`, con resultados por categoría en `pruebas.py`.

La validación histórica del 2026-10-28 obtuvo: NER `27/27`,
entidades adicionales `5/5`, semántica `29/29`, operaciones múltiples `12/12`,
relaciones `7/7`, contrato `8/8`, adversarial `22/22`, regresiones de fase 2
`11/11`, regresiones de fase 4 `17/17`, endurecimiento lingüístico `37/37`,
admisión `26/26`, matriz de modalidad `10/10` y relaciones/evidencia `8/8`.
El ejecutor incluye todas las baterías en el total y devuelve código distinto
de cero si alguna falla o lanza una excepción.

En la ejecución final de esta auditoría se obtuvieron `238/250`: 12 fallos,
0 grupos no ejecutados y 0 errores de ejecución. Los 12 fallos son los ejemplos
de NER independiente que no coincidieron exactamente con sus spans anotados;
las otras baterías pasaron. Por ello `python pruebas.py` devuelve `1` hasta
resolver la generalización NER.

La batería histórica no mide generalización limpia: 2 frases son idénticas a
entrenamiento y 6 son cercanas por similitud de caracteres (umbral `0.82`). Se
añadió un conjunto independiente de 15 casos sin coincidencias exactas ni
cercanas con entrenamiento. El modelo actual obtuvo `3/15` por coincidencia
exacta de spans en esta ejecución; esto indica que su generalización NER es
insuficiente para declararlo listo para integración. La nueva evaluación forma
parte de `python pruebas.py`, por lo que sus fallos producen un código distinto
de cero.

La evaluación independiente es pequeña y usa un conjunto de reglas lingüísticas
acotado; sus porcentajes no deben interpretarse como una estimación de
rendimiento en producción.

La prueba adversarial histórica `Vendi cinco cajas por dies dólares` antes
esperaba cero operaciones. El contrato ahora conserva la venta y la cantidad,
deja el precio ausente y marca la operación `PARCIAL`: acción y cantidad tienen
evidencia independiente, pero la forma sin tilde no sustenta con seguridad el
agente. `dies` no se autocorrige a `diez`; la expectativa actual conserva esta
operación parcial y se valida dentro de las métricas adversariales.

La auditoría de aceptación añadió dos regresiones: no extraer `compré` como
cantidad en `Tal vez compré mercadería por 10 dólares`, y no confirmar una
relación de abono con una compra negada. Ambas correcciones se verificaron en la
batería ampliada. La evaluación es una batería acotada de ejemplos y no
demuestra estabilidad general para el español informal ni sustituye validación
con datos representativos del negocio.

La aceptación semántica agrega pruebas para excluir operaciones no afirmadas,
conservar cláusulas afirmadas independientes, respetar campos ausentes y vagos,
y atribuir agentes según la evidencia gramatical; los casos nuevos obtuvieron
`26/26`. En `Compré cinco cajas por dies dólares`, el contrato conserva acción
y cantidad y deja el precio ausente; no corrige `dies` automáticamente.

En la fase de generalización del 2026-10-28, el artefacto vigente (entrenado con
45 ejemplos) obtuvo 3/15 coincidencias exactas en la evaluación independiente
y 238/250 en las baterías. El archivo de entrenamiento ya contiene 74 ejemplos
(45 originales y 29 complementarios); el modelo local no había incorporado
esa ampliación. Una candidata temporal entrenada desde el modelo base con esos
74 ejemplos logró 11/12 en desarrollo, 5/15 en la evaluación independiente y
9/15 en el conjunto final reservado, frente a 1/12, 3/15 y 0/15 del vigente.
Sin embargo, cayó a 224/250 en las baterías, con retrocesos en operaciones
múltiples, relaciones, modalidad y otras regresiones. Se rechazó y el modelo
vigente quedó intacto. El informe con las 15 comparaciones y métricas completas
está en `resultados_ner/informe_fase_2026-10-28.md`.

## Limitaciones y evolución

La resolución de referencias es deliberadamente acotada: reconoce algunas
referencias textuales a compras y abstiene el enlace si hay más de un destino
posible. No resuelve deudas externas, no calcula saldos ni interpreta errores
ortográficos mediante una puntuación arbitraria. La detección de entidades y su
interpretación como roles se mantienen separadas: una entidad NER no se
convierte automáticamente en producto específico, y las reglas de extracción
no admiten preposiciones como nombres de producto.

El modelo personalizado está ignorado por Git y es un artefacto local. El
artefacto vigente refleja los 45 ejemplos originales; `entrenamiento.py`
contiene ahora además 29 datos complementarios que todavía no se promovieron.
El modelo base evaluado fue `es_core_news_md` 3.8.0. La batería independiente
actual expone errores en límites multi-token, spans monetarios con símbolos,
conceptos de varias palabras y entidades compuestas. Las reglas semánticas
pueden recuperar algunos valores, pero eso no equivale a corregir ni validar el
modelo NER.
