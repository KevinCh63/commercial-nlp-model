# Informe: fase de mejora de generalización NER

Fecha: 2026-10-28

## Alcance y estado local

Se trabajó en el módulo NLP local, sin integración con backend ni cambios de aplicación principal. Al iniciar, `git status` mostró cambios locales previos en `.gitignore`, `README.md`, `entrenamiento.py`, `interpretacion.py`, `pruebas.py`, `requirements.txt` y el directorio no versionado `resultados_ner/`; todos se conservaron. El artefacto vigente reporta 45 datos de entrenamiento, mientras el `train_data` actual contiene 74 (45 originales y 29 complementarios). No se cambiaron expectativas de pruebas ni se usaron frases independientes literalmente en entrenamiento.

La configuración de la candidata fue la que ya está en `entrenamiento.py`: modelo base `es_core_news_md 3.8.0`, spaCy 3.8.16, 100 épocas, dropout 0.3, semilla 42. Se validaron offsets, etiquetas, límites de tokens y solapamientos. El umbral de similitud textual fue 0.82; no hubo coincidencias exactas ni cercanas entre entrenamiento y desarrollo, evaluación independiente o conjunto final reservado.

## Línea base reproducida

- Evaluación NER independiente: **3/15 frases** con todas las entidades exactas. La misma ejecución reprodujo las 12 fallas de la batería integral: **238/250**, sin excepciones ni baterías no ejecutadas.
- Micro por entidad exacta (span + etiqueta): P=0.471, R=0.615, F1=0.533. No es métrica por token.
- Macro-F1 por las cuatro etiquetas: 0.454.

### Métricas independientes por etiqueta (entidad exacta)

| Etiqueta | TP | FP | FN | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| QUANTITY | 3 | 9 | 5 | 0.250 | 0.375 | 0.300 |
| PRODUCT | 7 | 10 | 5 | 0.412 | 0.583 | 0.483 |
| PRICE | 13 | 4 | 2 | 0.765 | 0.867 | 0.812 |
| CONCEPT | 1 | 4 | 3 | 0.200 | 0.250 | 0.222 |

### Diagnóstico de los 15 ejemplos independientes

Los errores se emparejan por solapamiento. `∅` indica entidad omitida o falsa entidad. Los spans y etiquetas esperadas/predichas aparecen literalmente; los errores múltiples de una frase se separan con `<br>`.

| # | Frase | Fragmento esperado [etiqueta] → predicho [etiqueta] (tipo) |
|---:|---|---|
| 1 | Por 37 dólares recibieron ocho sacos de quinua | `ocho sacos` [QUANTITY] → `sacos` [PRODUCT] (confusión de etiqueta y límites)<br>`∅` [∅] → `ocho` [QUANTITY] (falsa entidad) |
| 2 | El municipio desembolsó 642 dólares para el mantenimiento vial | `mantenimiento vial` [CONCEPT] → `mantenimiento` [CONCEPT] (límites incorrectos o entidad parcial)<br>`∅` [∅] → `vial` [CONCEPT] (falsa entidad) |
| 3 | Ayer vendí, sin intermediarios, once bidones de aceite vegetal por 48 euros | `once bidones` [QUANTITY] → `bidones` [PRODUCT] (confusión de etiqueta y límites)<br>`aceite vegetal` [PRODUCT] → `vegetal` [PRODUCT] (límites incorrectos o entidad parcial)<br>`∅` [∅] → `once` [QUANTITY] (falsa entidad)<br>`∅` [∅] → `aceite` [QUANTITY] (falsa entidad) |
| 4 | ¿Cuánto pagó Rosa por tres metros de cable? 26 dólares | `tres metros` [QUANTITY] → `metros` [PRODUCT] (confusión de etiqueta y límites)<br>`∅` [∅] → `¿` [CONCEPT] (falsa entidad)<br>`∅` [∅] → `Cuánto pagó` [PRICE] (falsa entidad)<br>`∅` [∅] → `tres` [QUANTITY] (falsa entidad) |
| 5 | Recibimos $58 por las canastas artesanales | `$58` [PRICE] → `58 por` [PRICE] (límites incorrectos o entidad parcial)<br>`canastas artesanales` [PRODUCT] → `artesanales` [PRODUCT] (límites incorrectos o entidad parcial) |
| 6 | Con 72 euros compraron las láminas galvanizadas | `láminas galvanizadas` [PRODUCT] → `galvanizadas` [PRODUCT] (límites incorrectos o entidad parcial)<br>`∅` [∅] → `las` [QUANTITY] (falsa entidad)<br>`∅` [∅] → `láminas` [QUANTITY] (falsa entidad) |
| 7 | Nueve kilos de cacao se colocaron a 33 dólares | Sin error; coincidencia exacta. |
| 8 | Para reparar el techo, gastamos 215 dólares | `reparar el techo` [CONCEPT] → `techo` [PRODUCT] (confusión de etiqueta y límites) |
| 9 | El proveedor facturó 19,75 dólares por dos frascos de esencia | `dos frascos` [QUANTITY] → `frascos` [PRODUCT] (confusión de etiqueta y límites)<br>`esencia` [PRODUCT] → `∅` [∅] (entidad omitida)<br>`∅` [∅] → `dos` [QUANTITY] (falsa entidad) |
| 10 | Se pagaron 84 pesos por el transporte escolar | `transporte escolar` [CONCEPT] → `transporte` [CONCEPT] (límites incorrectos o entidad parcial) |
| 11 | La cooperativa vendió mallas y redes por 1.250 dólares | Sin error; coincidencia exacta. |
| 12 | Entregué tres docenas de empanadas al puesto por 60 dólares | `tres docenas` [QUANTITY] → `docenas` [PRODUCT] (confusión de etiqueta y límites)<br>`∅` [∅] → `tres` [QUANTITY] (falsa entidad) |
| 13 | Con €22 compró 1.5 kg de nuez pecana | `€22` [PRICE] → `22` [PRICE] (límites incorrectos o entidad parcial)<br>`nuez pecana` [PRODUCT] → `pecana` [PRODUCT] (límites incorrectos o entidad parcial)<br>`∅` [∅] → `nuez` [QUANTITY] (falsa entidad) |
| 14 | Factura de repuestos: 39 USD, cancelados hoy | Sin error; coincidencia exacta. |
| 15 | Vendí tres cajas de azulejos por un total de 125 dólares | `∅` [∅] → `un total` [PRICE] (falsa entidad) |

Patrones agregados observados: errores de límites/parcial (8), falsas entidades (13), confusión de etiqueta y límites (6), una entidad omitida. No hubo casos emparejados con límites idénticos y etiqueta incorrecta. Hay confusión QUANTITY/PRODUCT en unidades como `sacos`, `bidones`, `metros` y `docenas`; límites imprecisos de PRICE con `$` y símbolos; PRODUCT/CONCEPT compuesto; y spans espurios alrededor de puntuación o expresiones como `un total`. No se observó un patrón dominante de errores de valor numérico decimal; sí se deben cubrir símbolos, orden y unidades. Las clases de error pueden solaparse con el emparejamiento automático de entidades.

## Revisión de los datos

La validación actual aceptó las 74 anotaciones sin offsets inválidos, etiquetas desconocidas, desalineación de tokens, solapamientos ni textos duplicados. Conteo de entidades anotadas: QUANTITY 44, PRODUCT 49, PRICE 71 y CONCEPT 20; hay más ejemplos de importes que de conceptos, por lo que el balance no es uniforme. Los datos cubren números escritos y dígitos, unidades comerciales, productos simples/compuestos, nombres de moneda y símbolos, conceptos de gastos/ingresos, puntuación y entidades cercanas. No se identificó una cobertura deliberada suficiente de frases con varias operaciones comerciales ni una partición agrupada por familias lingüísticas.

El chequeo de similitud normaliza a minúsculas y palabras Unicode y usa `SequenceMatcher.ratio()` con umbral 0.82. Resultado de todos los pares: entrenamiento–desarrollo 0, entrenamiento–independiente 0, entrenamiento–final 0, desarrollo–independiente 0, desarrollo–final 0 e independiente–final 0 exactos/cercanos.

## Candidata y comparación

- Datos utilizados: 74 ejemplos; SHA-256 `50f19322fc8d2da5a8a57070162ee57329b2fabb14c904bce8f1e9f3bdbfce22`. No se utilizó la evaluación final para entrenar ni seleccionar ejemplos.
- El artefacto de candidata quedó guardado como `resultados_ner/candidata_temporal_2026-10-28/` y se cargó con `interpretacion.cargar_modelo` antes de medir. No se promovió.

| Conjunto | Modelo vigente: frases exactas | Candidata: frases exactas | F1 micro vigente → candidata |
|---|---:|---:|---:|
| Desarrollo (12) | 1/12 | 11/12 | 0.441 → 0.967 |
| Independiente (15) | 3/15 | 5/15 | 0.533 → 0.759 |
| Final reservado (15) | 0/15 | 9/15 | 0.295 → 0.795 |

Las 15 frases independientes de compatibilidad reprodujeron 5/15 al recargar la candidata por el cargador real. Después de esa carga se repitieron las 16 baterías con la misma matriz compartida y dieron 224/250, sin errores de ejecución. La comparación por grupo bloquea la promoción por retrocesos en operaciones múltiples, relaciones, adversarial, fase 2, fase 4 y modalidad. La evaluación final reservada subió de 0/15 a 9/15, pero es pequeña y no compensa una regresión en comportamiento semántico.

### Baterías históricas

| Batería | Vigente | Candidata |
|---|---:|---:|
| NER | 27/27 | 27/27 |
| NER independiente | 3/15 | 5/15 |
| validación de datos | 8/8 | 8/8 |
| interfaz independiente | 8/8 | 8/8 |
| entidades | 5/5 | 5/5 |
| semántica | 29/29 | 29/29 |
| operaciones múltiples | 12/12 | 11/12 |
| relaciones | 7/7 | 1/7 |
| contrato | 8/8 | 8/8 |
| adversarial | 22/22 | 20/22 |
| fase 2 | 11/11 | 9/11 |
| fase 4 | 17/17 | 15/17 |
| endurecimiento | 37/37 | 37/37 |
| admisión | 26/26 | 26/26 |
| matriz modalidad | 10/10 | 7/10 |
| relaciones/evidencia | 8/8 | 8/8 |

**Total:** modelo vigente 238/250; candidata 224/250. La candidata retrocedió 14 aprobaciones agregadas, en relaciones (1/7), modalidad (7/10), operaciones múltiples (11/12), fase 2 (9/11), fase 4 (15/17) y adversarial (20/22). Semántica base permaneció 29/29; esto no borra las regresiones en otras baterías.

## Decisión y cambios

Se **conserva el modelo vigente** y no se promueve la candidata. El resultado independiente sí mejoró 3/15→5/15 y la final 0/15→9/15, pero el retroceso de 238/250→224/250, junto con regresiones claras en relaciones y modalidad, invalida la promoción para el módulo actual. La política de admisión pasó 26/26 y el contrato 8/8; no obstante, operaciones múltiples y varias baterías relacionales/modalidad fallaron.

Se modificó el mecanismo de entrenamiento para compartir una única definición de la matriz histórica entre `pruebas.py` y `entrenamiento.py`, comparar cada batería antes de promocionar y cargar la candidata temporal mediante `interpretacion.cargar_modelo`. Se rechazará si retrocede en desarrollo o en cualquier grupo histórico. El artefacto vigente no cambió. Se ejecutó `pruebas.py` hasta completar las 16 baterías (238/250; 12 fallos de coincidencia NER independiente, sin excepciones ni grupos omitidos), además de `py_compile` y `git diff --check`; el candidato no se promovió.

## Limitaciones y estado

El conjunto independiente tiene 15 frases y el final reservado otras 15; sus tasas son inestables y no representan por sí solas tráfico comercial. La calidad sigue débil en precisión de QUANTITY y CONCEPT y en spans compuestos. Falta ampliar evaluación externa representativa y resolver regresiones semánticas antes de promover. Estado: **preparado para revisión técnica**, no listo para integración.

Archivos de soporte: `linea_base_reproducida_2026-10-28.json` (predicciones baseline), `candidata_actual_2026-10-28.json` (métricas y grupos de batería), `reproduccion_linea_base_2026-10-28.log`, `candidata_actual_2026-10-28.log`, `verificacion_baterias_recarga_2026-10-28.log` y `comparacion_baterias_matriz_compartida_2026-10-28.json`.
