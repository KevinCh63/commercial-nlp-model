# Fase siguiente: diagnóstico y estrategia de generalización

Fecha: 2026-10-28

## 1. Estado inicial y archivos

Antes de crear este informe se revisaron `git status`, el diff vigente y el directorio de resultados. Ya había cambios locales no comprometidos en `.gitignore`, `README.md`, `entrenamiento.py`, `interpretacion.py`, `pruebas.py` y `requirements.txt`, además de informes y la candidata temporal no versionados bajo `resultados_ner/`. Se conservaron todos. En esta fase no se editó código, datos, expectativas de pruebas ni artefactos spaCy; no hubo entrenamiento, promoción ni borrado.

Se inspeccionaron `entrenamiento.py`, `interpretacion.py`, `pruebas.py`, las 74 anotaciones actuales, `resultados_ner/informe_fase_2026-10-28.md`, las métricas JSON y el artefacto candidato que dejó la fase anterior. El artefacto vigente declara 45 ejemplos en su metadata; `entrenamiento.py` ya contiene 74 (45 originales + 29 complementarios).

## 2. Recorrido real del procesamiento

| Etapa | Código y datos | Qué hace y qué supone | Riesgos / pruebas |
|---|---|---|---|
| Carga | `interpretacion.cargar_modelo` | Resuelve por ruta y cachea una sola instancia; carga el modelo pedido o lanza `RuntimeError`. | Una ruta cacheada conserva la instancia mientras viva el proceso. Carga e interfaz se cubren en `evaluar_interfaz_independiente`. |
| NER de oración | `interpretacion.extraer_entidades`; `interpretar_texto` | Ejecuta `nlp(texto)`, recoge spans y etiquetas, encuentra un verbo principal, acción y modalidad; luego convierte entidades en roles. | NER puede omitir, extender o cambiar etiquetas. `evaluar_ner` y `medir_ner` verifican spans exactos; `PRUEBAS_NER_INDEPENDIENTE` y `PRUEBAS_NER_FINAL` son sus conjuntos. |
| Normalización de roles | `_roles_desde_entidades`, `_entidades_comerciales`, `_moneda` | Usa la primera entidad por rol y después aplica patrones para cantidad+unidad, importes con moneda textual y algunos conceptos/contrapartes. Algunas reglas reemplazan el rol NER; otras conservan spans NER originales en `entidades`. | Las reglas de cantidad/precio solo cubren patrones enumerados. Un regex amplio puede confundir producto, concepto, unidad, contraparte o importe. Semántica, adversariales, endurecimiento y contrato cubren casos concretos. |
| Validación semántica | `validar_resultado`; `interpretar_texto` | Añade faltantes/advertencias/estado y considera modalidad. Marca varias acciones en una interpretación como advertencia porque esa API representa una frase. | La interpretación de una sola frase puede elegir el primer verbo principal; `evaluar_semantica`, modalidad y contrato verifican campos y estados. |
| Segmentación | `interpretar_operaciones` | Ejecuta primero spaCy en la oración completa para encontrar acciones; si hay varias, corta el texto desde cada verbo, recorta separadores y vuelve a llamar `interpretar_texto` para cada segmento. Filtra segmentos no afirmados y recupera modalidad contextual desde el prefijo. | La segunda ejecución de NER ocurre en texto más corto y con distinto contexto. No hay entrenamiento con múltiples acciones: las 74 frases anotadas contienen cero ejemplos con dos o más verbos comerciales (conteo por spaCy). `evaluar_operaciones_multiples` y fase 4 prueban el resultado. |
| Relaciones y evidencia | `interpretar_operaciones_con_relaciones` | Asigna IDs y evidencias; busca referencia textual de compra/adquisición/factura/deuda en cláusulas pagar/abonar y solo enlaza compras anteriores según producto, ordinal, “compra anterior” o un único destino. | La heurística requiere cláusulas/roles previos bien interpretados; deja ambiguas referencias no únicas. `evaluar_relaciones`, fase 4, contrato y `evaluar_relaciones_evidencia` cubren vínculos. |
| Admisión | `interpretar_operaciones` + `modalidad_contextual` y `detectar_modalidad` | Conserva operaciones cuya modalidad final sea `AFIRMADA`; excluye negadas, inciertas e intenciones. | Las pruebas actuales cubren negación, deseo, futuro y cláusulas contrastivas. La regresión “modalidad 7/10” de la candidata es mayormente fallo de producto, no de la etiqueta de modalidad. |

### Detalle de segmentación verificado

Cuando hay más de una acción, la implementación detecta acciones sobre el documento completo, pero vuelve a ejecutar `nlp(segmento)` desde `interpretar_texto(segmento)`. Por tanto el sistema produce dos predicciones NER diferentes: oración completa y cada cláusula recortada. La candidata reconoció `fundas` como PRODUCT en la oración completa de `Vendí tomate por 6 dólares y vendí fundas por 4 dólares`, pero no en el segmento `vendí fundas por 4 dólares`. Esto confirma un cambio de distribución de contexto entre entrenamiento (frases completas de una operación) e inferencia segmentada.

## 3. Diagnóstico NER independiente

La línea base sigue siendo 3/15 frases exactas. Las métricas son por entidad: una entidad cuenta como TP solo si coinciden límites y etiqueta; no son métricas por token. La exactitud de frase requiere que coincida el conjunto entero de spans de esa frase.

| Etiqueta | Vigente P / R / F1 | Candidata P / R / F1 |
|---|---:|---:|
| QUANTITY | 0.250 / 0.375 / 0.300 | 1.000 / 1.000 / 1.000 |
| PRODUCT | 0.412 / 0.583 / 0.483 | 0.667 / 0.500 / 0.571 |
| PRICE | 0.765 / 0.867 / 0.812 | 0.875 / 0.933 / 0.903 |
| CONCEPT | 0.200 / 0.250 / 0.222 | 0.286 / 0.500 / 0.364 |

Los diagnósticos de la tabla de 15 casos de la [fase anterior](informe_fase_2026-10-28.md) producen 8 spans parciales/de límites, 13 falsas entidades, 6 confusiones simultáneas de etiqueta y límites y 1 entidad omitida; una frase puede producir varios diagnósticos. No hubo casos emparejados con límites idénticos y solo etiqueta equivocada. Los spans anotados pasan `doc.char_span`: no hay evidencia de error de alineación del conjunto oro/tokenizador.

| Familia observada | Evidencia reproducida | Primera causa / límite seguro |
|---|---|---|
| Cantidad + unidad | `ocho sacos`, `once bidones`, `tres metros`, `tres docenas` se separan: número como QUANTITY y unidad como PRODUCT u omitida. | NER etiqueta/segmenta mal; postprocesar solo patrones cerrados número + unidad conocida puede ser seguro, con lista de unidades y pruebas de colisión. `vasos` ya se trata como excepción producto/unidad. |
| PRODUCT compuesto | `aceite vegetal`, `canastas artesanales`, `nuez pecana` quedan parciales o con un token como QUANTITY. | Principalmente spans y etiquetas NER; una regla que una adjetivos/nombres vecinos sin evidencia puede absorber verbos, preposiciones o un concepto. |
| CONCEPT compuesto | `mantenimiento vial`, `transporte escolar` parciales; `reparar el techo` aparece como `techo` PRODUCT. | El límite depende del contexto sintáctico. No es seguro ampliar todo sustantivo/adjetivo cercano como CONCEPT. |
| PRICE | `$58` salió como `58 por`; `€22` perdió el símbolo; decimales y monedas en otros ejemplos sí se reconocen en algunos casos. | El regex de recuperación reconoce principalmente número seguido de moneda textual; no cubre de forma equivalente símbolos prefijo `$`/`€`, y NER puede extender spans a palabras funcionales. Un parser anclado a moneda + número exactos podría arreglar solo ese formato. |
| Distractores/puntuación | `¿` recibió CONCEPT, `Cuánto pagó` PRICE y `un total` PRICE. | Falsas entidades NER. No conviene añadir reglas positivas por mera proximidad a un verbo comercial; añadir negativos anotados y reglas de supresión acotadas. |
| Variantes y contexto | Las formas con puntuación, cláusulas y negación producen spans muy distintos; el corte de cláusulas cambia el contexto del NER. | Falta contexto segmentado en entrenamiento; confirmado en la comparación de `fundas`. |

## 4. Comparación de regresiones de la candidata

Se cargaron el vigente y la candidata temporal guardada; se ejecutaron exactamente los evaluadores existentes para ambos. En los textos comparados, tokenización, POS, etiqueta morfológica y dependencias coinciden; el primer cambio observado está en NER de oración completa o en la segunda ejecución NER por segmento. No observé evidencia de que el segmentador o el resolutor de relaciones cambien por sí mismos.

### Relaciones: 7/7 → 1/7

El evaluador compara conjuntamente las operaciones y las relaciones. La candidata produjo las siete relaciones esperadas exactamente (incluidos enlaces confirmados, ambiguos y ausencia de enlace); **no** se reproduce una regresión del algoritmo de resolución de relaciones en estos siete fixtures. Los seis fallos de la batería son producto omitido por NER. Tabla de los seis casos que hacen fallar la prueba combinada:

| Entrada | Esperado | Vigente | Candidata | Primera divergencia / riesgo |
|---|---|---|---|---|
| Compré mercadería por 100 dólares y aboné 30 dólares de esa compra | producto(s): mercadería, None; relación(es): CONFIRMADA→0 | producto(s): mercadería, None; relación(es): CONFIRMADA→0 | producto(s): None, None; relación(es): CONFIRMADA→0 | NER sobre frase completa; al faltar producto también puede impedir resolver referencias futuras por nombre de producto. |
| Compré mercadería por 100 dólares y gasté 5 dólares en transporte | producto(s): mercadería, None; relación(es): — | producto(s): mercadería, None; relación(es): — | producto(s): None, None; relación(es): — | NER sobre frase completa; al faltar producto también puede impedir resolver referencias futuras por nombre de producto. |
| Compré mercadería por 100 dólares y pagué 30 dólares | producto(s): mercadería, None; relación(es): — | producto(s): mercadería, None; relación(es): — | producto(s): None, None; relación(es): — | NER sobre frase completa; al faltar producto también puede impedir resolver referencias futuras por nombre de producto. |
| Compré papel por 100 dólares y compré fundas por 50 dólares y aboné 10 dólares de esa segunda compra | producto(s): papel, fundas, None; relación(es): CONFIRMADA→1 | producto(s): papel, fundas, None; relación(es): CONFIRMADA→1 | producto(s): papel, None, None; relación(es): CONFIRMADA→1 | NER sobre frase completa; al faltar producto también puede impedir resolver referencias futuras por nombre de producto. |
| Compré papel por 100 dólares y compré fundas por 50 dólares y aboné 10 dólares de esa compra | producto(s): papel, fundas, None; relación(es): AMBIGUA→None | producto(s): papel, fundas, None; relación(es): AMBIGUA→None | producto(s): papel, None, None; relación(es): AMBIGUA→None | NER sobre frase completa; al faltar producto también puede impedir resolver referencias futuras por nombre de producto. |
| Compré papel por 100 dólares y vendí fundas por 50 dólares | producto(s): papel, fundas; relación(es): — | producto(s): papel, fundas; relación(es): — | producto(s): papel, None; relación(es): — | NER sobre frase completa; al faltar producto también puede impedir resolver referencias futuras por nombre de producto. |

La relación realmente emitida coincide con la esperada en los 7/7 casos; el 1/7 resume la igualdad compuesta de operación + vínculo. La causa probable del retroceso agregado es un recall bajo de productos genéricos o cortos (`mercadería`, `fundas`), no la regex relacional. Riesgo: si las relaciones futuras se resuelven por producto, un producto ausente sí causaría un error funcional aunque el fixture actual use un ordinal o una sola compra.

### Operaciones múltiples: 12/12 → 11/12

| Entrada | Esperado | Vigente | Candidata | Etapa / causa |
|---|---|---|---|---|
| Vendí tomate por 6 dólares y vendí fundas por 4 dólares | segundo producto `fundas` | tomate, fundas | tomate, None | NER después de segmentar: el texto completo sí contiene `fundas` PRODUCT en ambos modelos; la candidata pierde ese span al volver a etiquetar el segmento aislado. |
| Vendí tomate por 6 dólares y compré | segundo producto `fundas` | tomate, None | tomate, None | sin primera divergencia determinada: el texto completo sí contiene `fundas` PRODUCT en ambos modelos; la candidata pierde ese span al volver a etiquetar el segmento aislado. |
| Compré y pagué | segundo producto `fundas` | None, None | None, None | sin primera divergencia determinada: el texto completo sí contiene `fundas` PRODUCT en ambos modelos; la candidata pierde ese span al volver a etiquetar el segmento aislado. |

El número y los límites de las dos operaciones se conservan. No hay evidencia de que la segmentación por acciones sea la causa de esta diferencia concreta; la segunda inferencia NER con contexto reducido sí es causa observada.

### Matriz de modalidad: 10/10 → 7/10

| Entrada | Expectativa funcional | Vigente | Candidata | Primera divergencia |
|---|---|---|---|---|
| No compré tomate, pero vendí cebollas | acciones/productos: [['vender', 'cebollas']] | vender (producto=cebollas, cantidad=None, concepto=None, precio=None) | vender (producto=None, cantidad=None, concepto=None, precio=None) | NER sobre frase completa; la modalidad/admisión de la cláusula esperada se mantiene, pero `producto` falta o se contamina. |
| Tal vez compré tomate, pero ayer vendí cebollas | acciones/productos: [['vender', 'cebollas']] | vender (producto=cebollas, cantidad=None, concepto=None, precio=None) | vender (producto=None, cantidad=None, concepto=None, precio=None) | NER sobre frase completa; la modalidad/admisión de la cláusula esperada se mantiene, pero `producto` falta o se contamina. |
| Compré tomate y no vendí cebollas | acciones/productos: [['comprar', 'tomate']] | comprar (producto=tomate, cantidad=None, concepto=None, precio=None) | comprar (producto=tomate y, cantidad=None, concepto=None, precio=None) | NER sobre frase completa; la modalidad/admisión de la cláusula esperada se mantiene, pero `producto` falta o se contamina. |

Por eso 7/10 no prueba una regresión de detección de negación/incertidumbre. En las tres filas la acción afirmada correcta se conserva; lo que hace fallar la aserción conjunta son spans de producto incorrectos. Riesgo: campos faltantes reducen utilidad aunque la política de admisión siga estable.

### Fase 2 y adversariales: 11/11 → 9/11 y 22/22 → 20/22

Los dos fallos son los mismos en ambas baterías; no son cuatro errores distintos.

| Entrada | Esperado | Vigente | Candidata | Etapa / causa probable |
|---|---|---|---|---|
| Adquirí jabón por 12 dólares y luego aboné 4 dólares de aquella compra | primera operación `comprar`, producto `jabón`, importe `12 dólares`; abono enlazado a compra | producto `jabón` presente; vínculo confirmado | producto ausente; vínculo confirmado | NER de frase completa omite PRODUCT `jabón`; la relación aún se resuelve por referencia/compra única. Riesgo funcional si el destino se resuelve por artículo. |
| Vendí cuatro cuadernos, pero todavía no fijé el importe | producto `cuadernos`, cantidad `cuatro`, sin concepto ni importe | concepto ausente | concepto falso `fijé el importe`; cantidad y producto conservados | NER de frase completa marca como CONCEPT una cláusula verbal negativa/distractora. El regex no elimina esa entidad; suprimir verbos indiscriminadamente también borraría conceptos infinitivos válidos. |

### Fase 4: 17/17 → 15/17

| Entrada | Esperado | Vigente | Candidata | Etapa / riesgo |
|---|---|---|---|---|
| `Vendí mercadería por 20 dólares` | producto `mercadería`, `interpretacion=GENERICA` | presente | campo ausente | NER de frase completa omite el PRODUCT genérico; perderlo degrada el contrato aunque el patrón sea genérico. |
| `Compré artículos` | producto `artículos`, `interpretacion=GENERICA` | presente | campo ausente | NER de frase completa omite PRODUCT genérico; una regla fija de catálogo podría recuperar esta palabra pero no generaliza a vocabulario nuevo. |

La batería de admisión se mantuvo en 26/26 y contrato en 8/8; no se observó una operación ficticia nueva en esos grupos. Sí hay campos de producto que faltan, y una falsa entidad CONCEPT. La suite modalidad/relaciones muestra que esto sigue siendo una regresión de contenido NER, aunque la admisión de cláusulas sea correcta.

### Causas confirmadas y abiertas

**Confirmadas por salida comparada y código:**

- No cambian tokenizer ni análisis morfosintáctico en los casos diagnósticos; las diferencias se originan primero en spans/etiquetas NER. El entrenamiento llama `entrenar_ner` con NER activo y desactiva los demás pipes; el resto del intérprete aplica el mismo código a ambas salidas.
- Hay pérdida de entidad dependiente de contexto: `fundas` está en el NER de oración completa y desaparece en la NER del segmento aislado.
- `evaluar_relaciones` es compuesto; su 1/7 no equivale a 6 vínculos incorrectos. Las siete relaciones generadas son correctas según las expectativas de esos fixtures.
- La modalidad para los tres casos de 7/10 conserva las acciones esperadas; el producto es la causa observada de esas aserciones fallidas.
- En `fijé el importe`, el falso positivo CONCEPT llega a `producto/concepto` semántico porque `_roles_desde_entidades` toma etiquetas NER y no existe una exclusión específica de esa frase.

**Hipótesis que requieren evidencia adicional:**

- La desproporción de etiquetas contribuye a menor generalización CONCEPT/PRODUCT. Es plausible (20 CONCEPT vs 71 PRICE) pero no prueba causalidad por sí sola.
- El entrenamiento desde el modelo base y `nlp.initialize()` puede hacer olvidar vocabulario/etiquetas genéricas respecto del vigente; se requiere ablation controlada de inicialización/retención de pesos para confirmarlo.
- La forma de segmentar por tokens de acción puede cortar complementos o producir inferencias con poca evidencia. Se observó cambio de NER en un caso, pero no se demuestra que cambiar el segmentador sea globalmente mejor.
- El emparejamiento de errores por máximo solapamiento produce categorías diagnósticas útiles pero no una matriz de confusión pura por entidad; algunos falsos positivos y spans parciales del mismo caso se contabilizan por separado.

## 5. Revisión de las 74 anotaciones

| Indicador | Observación |
|---|---|
| Tamaño / etiquetas | 74 frases; QUANTITY 44, PRODUCT 49, PRICE 71, CONCEPT 20 entidades anotadas. |
| Validación | Las validaciones locales aceptan offsets, alineación, etiquetas, no solapamiento y texto único. |
| Negativos | 0 ejemplos tienen `entities=[]`; el validador actual sí permite listas vacías de entidades. |
| Operaciones múltiples | 0 frases contienen dos o más verbos comerciales en el conteo por spaCy. |
| Entidades repetidas | 0 frases repiten una etiqueta personalizada dos veces o más dentro de la frase. |
| Moneda símbolo | 2 ejemplos anotados usan `$`; no hay `$`/`€` como familias equilibradas; `€` prefijo no está representado. |
| Cantidades / unidades | Hay formas escritas y dígitos y 18 spans coinciden con unidades comerciales de la lista examinada; falta variación de unidades/segmentos y del contexto multiacción. |
| Productos/conceptos | Hay productos compuestos, pero no hay balance de conceptos; 20 spans CONCEPT frente a 71 PRICE. No se cuantifican sistemáticamente pre/postposición o categorías por tipo sintáctico. |

El principal hueco confirmado para la regresión semántica no es solo el total por etiqueta: la inferencia ejecuta NER en cláusulas recortadas, mientras el conjunto de entrenamiento no tiene ejemplos con varias acciones ni pares frase completa/segmento. Tampoco hay negativos para `¿Cuánto pagó…?`, `un total` o “todavía no fijé el importe”.

### Propuesta antes de editar datos

Anotar **unas 36 frases distintas**, revisadas por una persona y anotadas por offsets, sin copiar literalmente ningún fixture independiente/final. Las cantidades son focos que se solapan; no se suman como grupos disjuntos:

| Foco dentro de las ~36 frases | Aproximación | Qué variación cubrir |
|---|---:|---|
| CONCEPT | 12 | gastos e ingresos; sintagmas nominales y verbos en infinitivo; una a varias palabras; concepto al inicio/final; nombres que pueden confundirse con producto. |
| QUANTITY | 10 | mitad dígitos/mitad palabras; unidades familiares y nuevas; cantidad sola, cantidad+unidad y cantidad+producto; posición antes/después del importe. |
| PRODUCT | 8 | una palabra y compuestos de 2–3 palabras; adjetivo/material; producto con y sin unidad; producto en cada cláusula y repetido. |
| PRICE / formato | 6 | `$` y `€` prefijos, USD/EUR/PESO, decimales con coma/punto, miles con separadores, moneda textual y monto escrito con palabras. |
| Estructura transversal | 8 de las anteriores | 2–3 operaciones por frase, compra/venta/gasto/abono, productos o precios repetidos, y órdenes sintácticos contrastivos. Debe haber anotación para la cláusula completa y spans exactos de entidades. |
| Negativos/distractores | 6 de las anteriores o frases breves propias | signo de pregunta, “total”, verbo no financiero de fijación/planificación, negación/intención, contexto sin ninguna entidad comercial. Usar `entities=[]` cuando no haya span verdadero; el esquema y validador lo admiten. |

El foco adicional de 8 multiacción responde a una ausencia cuantificada (0 actual); los 12 conceptos aumentan con prioridad una clase con solo 20 spans; los casos de precio son pocos pero corrigen formatos concretos sin inflar artificialmente PRICE. No hace falta copiar vocabulario reservado; se deben crear nombres y marcos lingüísticos distintos.

## 6. Partición y métricas

Las auditorías actuales detectaron cero similitudes >=0.82 en todos los cruces de train/dev/independiente/final, pero el umbral char-level no detecta necesariamente marcos semánticos cercanos. Para la siguiente partición:

1. Etiquetar cada ejemplo con familia de plantilla: acción y orden, tipo de cantidad/unidad, formato monetario, construcción de producto/concepto, polaridad/modalidad, número de operaciones y patrón de puntuación.
2. Asignar familias completas a train, dev o evaluación; no repartir sustituciones léxicas de una misma plantilla entre particiones. Revisar duplicados exactos y similitud cercana con el normalizador existente y una revisión humana de pares por patrón.
3. Usar desarrollo para decidir cambios. El histórico independiente de 15 sigue siendo comparador, pero no mide una muestra grande. El conjunto final anterior fue consultado en fases previas y en esta candidata, así que **ya no es sellado para próximas iteraciones**. Crear un conjunto final nuevo, representativo y con acceso restringido antes de optimizar más.
4. Registrar por candidato semilla/configuración/hash, datos exactos y artifacts. Evaluar una sola selección en el nuevo final cuando desarrollo y baterías pasen; no iterar con ese resultado.

Para cada etiqueta y partición, reportar TP/FP/FN, precisión, recall y F1 por entidad con span+etiqueta exactos; macro-F1, micro-F1 y número de frases con todos los spans exactos. Separadamente dar conteo de frase exacta, falsos spans, omisiones, límites incompletos/excesivos y confusiones por tipo. No llamarlo token accuracy.

La promoción debe requerir mejora verificable en desarrollo/generalización y ausencia de regresión en cada batería crítica —relaciones, modalidad, operaciones múltiples, admisión y contrato— revisadas individualmente. Incluir comparativa NER por etiqueta y evals históricas semánticas/funcionales completas. No aceptar una mejora en un total agregado para compensar un grupo crítico que cae; no se propone un umbral numérico nuevo sin datos de uso y tamaño de muestra suficiente.

## 7. Plan priorizado y siguiente experimento

1. **Menor experimento recomendado:** comparar sin modificar el flujo, en las 7 frases fallidas de relaciones y las cláusulas múltiples de las pruebas existentes, los spans NER de oración completa frente a los spans de cada segmento, proyectados sobre offsets originales. Registrar si usar solo los spans completos preserva producto sin contaminar otra cláusula. Hipótesis concreta: el re-NER de segmento causa pérdida de entidades que sí se reconocieron en contexto completo. El caso `fundas` ya confirma un ejemplo; falta medir todos los casos y colisiones entre spans/segmentos.
2. Aislar falsas entidades de la candidata para puntuación y distractores con negativos anotados, manteniendo la política actual de admisión y la resolución explícita de relaciones.
3. Preparar el lote equilibrado pequeño propuesto y particiones agrupadas por familia; no entrenar hasta cerrar anotaciones y revisar fuga.
4. Probar alternativas de inicialización/retención de pesos en una sola candidata por iteración y comparar los mismos datasets/matriz completa.
5. Después de cerrar desarrollo, crear/sellar evaluación final nueva y realizar una única evaluación final; no usar la anterior como criterio repetido.

Riesgo del experimento 1: spans de una oración que cruzan límite de cláusula deben recortarse o asignarse de forma explícita, no copiarse en ambas operaciones. El test debe verificar offsets/evidencia y que no aparezcan productos cruzados.

## 8. Pruebas ejecutadas y límites

| Batería / evaluación | Vigente | Candidata |
|---|---:|---:|
| Desarrollo NER, frase exacta | 1/12 | 11/12 |
| NER independiente, frase exacta | 3/15 | 5/15 |
| Final anterior, frase exacta (ya expuesto) | 0/15 | 9/15 |
| NER histórico | 27/27 | 27/27 |
| NER independiente en matriz integral | 3/15 | 5/15 |
| Validación de datos | 8/8 | 8/8 |
| Interfaz y carga | 8/8 | 8/8 |
| Entidades adicionales | 5/5 | 5/5 |
| Semántica | 29/29 | 29/29 |
| Operaciones múltiples | 12/12 | 11/12 |
| Relaciones | 7/7 | 1/7 (vínculos solos 7/7 exactos; seis fallos son campos de operación) |
| Contrato | 8/8 | 8/8 |
| Adversarial | 22/22 | 20/22 |
| Fase 2 | 11/11 | 9/11 (mismos dos fallos adversariales) |
| Fase 4 | 17/17 | 15/17 (dos productos genéricos omitidos) |
| Endurecimiento lingüístico | 37/37 | 37/37 |
| Admisión | 26/26 | 26/26 |
| Modalidad | 10/10 | 7/10 (fallos de producto/span; las acciones esperadas se conservan) |
| Relaciones/evidencia | 8/8 | 8/8 |
| Total integral | 238/250 | 224/250 |

Las ejecuciones detalladas se guardaron en `diagnostico_regresiones_detallado_2026-10-28.json` y `regression_evaluator_logs_2026-10-28.json`. No se ejecutó un nuevo entrenamiento ni se modificaron archivos de código, datos o modelo durante esta fase. La candidata y los informes de la fase previa se preservan. Las baterías son ejemplos acotados, no una estimación de producción; el NER independiente y final tienen solo 15 frases cada uno.

## Limitaciones y estado

La causa primaria de regresiones de esta candidata está respaldada por divergencia NER; no existe evidencia de que el código de relaciones/admisión cambiara su comportamiento en esos ejemplos. El tamaño muestral no permite estimar rendimiento productivo ni probar que el desequilibrio por etiquetas sea causal. El nuevo conjunto final sellado sigue faltando.

**Estado:** preparado para revisión técnica; no listo para integración. Próxima acción recomendada: medir la discrepancia NER frase completa vs segmentos con proyección de spans, antes de cambiar datos o reentrenar.

Archivos fuente relacionados: `entrenamiento.py`, `interpretacion.py`, `pruebas.py`; [informe fase anterior](informe_fase_2026-10-28.md); [predicciones/regresiones detalladas](diagnostico_regresiones_detallado_2026-10-28.json).
