# Fase autónoma de diagnóstico, corrección y validación NLP

**Fecha:** 2026-10-28
**Alcance:** intérprete NLP local; sin backend ni integración con la aplicación principal.

## Resumen ejecutivo

Corregí la pérdida comprobada de entidades entre NER de oración completa y NER de segmentos mediante una proyección conservadora: solo considera spans completos, alineables, contenidos en el segmento y sin solape, y exige que el segmento tenga otra entidad local como ancla. Los conflictos y las proyecciones omitidas ahora quedan en `diagnosticos_ner`, con offsets del texto original.

El modelo vigente conserva todas las métricas anteriores: **238/250**, más **4/4** verificaciones nuevas de segmentación (**242/254** en la matriz ampliada). La candidata mejora tras esta corrección de **224/250** a **230/254**, pero sigue por debajo del vigente y falla pruebas críticas, por lo que no se promovió. El vigente conserva 12 fallos exactos de spans NER independientes; esta fase no los disfraza con cambios de expectativas ni reglas léxicas especiales.

## Estado inicial y línea base

Antes de editar, `git status --short` mostraba cambios locales preexistentes en `.gitignore`, `README.md`, `entrenamiento.py`, `interpretacion.py`, `pruebas.py`, `requirements.txt` y archivos no versionados en `resultados_ner/`. Se preservaron. No se restauró ni eliminó ningún archivo. No hubo conflicto que impidiera trabajar.

La línea base se ejecutó con el comando real `python pruebas.py`:

- Modelo vigente: **238/250**, 12 fallos, 0 baterías omitidas, 0 excepciones; código de salida 1.
- Los 12 fallos eran coincidencias exactas de entidades en NER independiente (**3/15**); las otras 15 baterías pasaban.
- Candidata temporal previamente medida: **224/250**, NER independiente **5/15**, relaciones **1/7**, operaciones múltiples **11/12**, modalidad **7/10**. La candidata seguía rechazada por regresiones en varios grupos.
- El experimento de segmentos anterior documentó 8 spans completos contenidos que la inferencia local del vigente perdía y 6 en la candidata, además de entidades completas que cruzan límites (1 vigente, 9 candidata). No todos estos spans son semánticamente seguros de reasignar.

Registro de línea base: `resultados_ner/baseline_fase_autonoma_2026-10-28.log`.

## Causas raíz y prioridad

| Prioridad | Causa | Evidencia | Decisión |
|---|---|---|---|
| Alta | Cambio de contexto NER al segmentar | La candidata reconoce `fundas` en la oración completa, pero no en el segmento; el experimento encuentra pérdidas en ambos modelos. | Corregir con proyección acotada y ancla local; validar offsets, ambigüedad y modalidad. |
| Alta | Generalización NER exacta débil | El vigente falla 12/15 frases independientes por span/etiqueta; aparecen límites parciales, etiquetas confundidas y falsos spans. | No añadir regex para frases puntuales ni reentrenar sin nuevos datos etiquetados y un final realmente reservado. Documentar el siguiente paso de datos. |
| Media | Campos incompletos tras relaciones | El 1/7 de relaciones de la candidata es una medida compuesta; parte de los vínculos están bien y falla el campo de producto. La proyección recupera dos casos; no hay evidencia suficiente para cambiar el resolvedor. | Corregir en la capa de entidades cuando la recuperación sea segura; mantener el resolvedor y las referencias ambiguas. |
| Media | Cobertura desequilibrada del entrenamiento | La revisión existente contabiliza 74 ejemplos, con 20 entidades CONCEPT frente a 71 PRICE y cero frases con múltiples acciones comerciales. | No cambiar datos ni entrenar: faltan anotaciones independientes revisadas y un nuevo conjunto final sellado. |
| Baja para el vigente; alta para promoción de la candidata | Modalidad y admisión | El vigente conserva 10/10 modalidad y 26/26 admisión; la candidata tiene 7/10 modalidad y otros retrocesos. | No cambiar la lógica de modalidad ni promover la candidata. |

## Plan ejecutado

1. Reproducir la línea base antes de editar y revisar código, fixtures, resultados, modelo vigente y candidata.
2. Reutilizar el diagnóstico previo de oración completa frente a segmentos, sin adoptar B ni C tal cual: ambos tenían regresiones.
3. Probar una variante híbrida más estricta en el evaluador aislado. Solo permite recuperar spans completos contenidos, alineados, sin solape y con otra entidad local como ancla.
4. Incorporar la variante al intérprete y publicar diagnósticos de proyección/colisión, preservando offsets del texto original.
5. Agregar una batería permanente de cuatro verificaciones de segmentación a `construir_baterias()`, matriz que comparte el evaluador de pruebas y el control de promoción.
6. Ejecutar las pruebas específicas, `py_compile`, la batería integral y la matriz final para ambos modelos.

No se modificaron datos de entrenamiento ni expectativas existentes. No se entrenó, no se promovió modelo, no se tocó el backend y no se conectó la aplicación.

## Correcciones implementadas

### Proyección acotada en segmentos

En `interpretacion.py`, la inferencia de oración completa continúa identificando las acciones y entidades. Al interpretar cada segmento, el flujo compara sus entidades locales con spans de oración completa y agrega únicamente proyecciones que:

- quedan enteramente dentro del segmento y se alinean estrictamente a los tokens;
- no solapan ninguna predicción local incompatible;
- tienen al menos otra entidad local en el segmento como ancla.

Se conserva la predicción local si hay solape; no se recortan spans. La falta de ancla, spans no alineables y spans que cruzan límites se informan en `diagnosticos_ner`, con índices inclusivos/exclusivos relativos al texto original y una resolución explícita. La modalidad y la admisión siguen aplicándose por cláusula después de interpretar el segmento. La inferencia de cada segmento se reutiliza para no ejecutar spaCy dos veces sobre el mismo texto.

### Batería específica

En `pruebas.py` añadí `evaluar_segmentacion_ner` con cuatro casos: ancla ausente, una cláusula negada, una acción futura junto a una afirmada y evidencia de producto con offset original. La batería comparte la matriz real usada por `python pruebas.py` y por `entrenamiento.medir_baterias`; no altera expectativas anteriores.

### Documentación

`README.md` documenta la nueva clave superior `diagnosticos_ner`, el criterio de proyección y los conflictos que se abstienen de proyectar.

### Evaluador experimental

Amplié el evaluador existente de `resultados_ner/` con la estrategia refinada “híbrida con ancla” y guardé resultados en un JSON nuevo. El JSON anterior y los informes previos permanecen intactos.

## Resultados antes y después

| Modelo / matriz | Antes | Después | Cambio relevante |
|---|---:|---:|---|
| Vigente, matriz original | 238/250 | 238/250 | Sin cambios en las 16 baterías previas; quedan los mismos 12 fallos exactos de NER independiente. |
| Vigente, batería segmentación segura | — | 4/4 | Sin asignar `cajas` a una operación sin ancla; negación y futuro no generan operaciones; offsets/evidencia válidos. |
| Vigente, matriz ampliada | — | 242/254 | Las 12 fallas originales de NER independiente siguen siendo las únicas fallas. |
| Candidata, matriz original | 224/250 | 227/250 equivalente | Operaciones múltiples suben 11/12→12/12; relaciones 1/7→3/7; los otros retrocesos persisten. |
| Candidata, batería segmentación segura | — | 3/4 | Mantiene un error de entidad local (`tomate y`) en la cláusula de compra negada; las otras tres verificaciones pasan. |
| Candidata, matriz ampliada | — | 230/254 | Sigue 12 aprobaciones por debajo del vigente en la matriz ampliada. |

Matriz final por baterías:

| Batería | Vigente | Candidata |
|---|---:|---:|
| NER histórica | 27/27 | 27/27 |
| NER independiente | 3/15 | 5/15 |
| Validación de datos | 8/8 | 8/8 |
| Interfaz independiente | 8/8 | 8/8 |
| Entidades | 5/5 | 5/5 |
| Semántica | 29/29 | 29/29 |
| Operaciones múltiples | 12/12 | 12/12 |
| Relaciones | 7/7 | 3/7 |
| Contrato | 8/8 | 8/8 |
| Adversarial | 22/22 | 20/22 |
| Fase 2 | 11/11 | 9/11 |
| Fase 4 | 17/17 | 15/17 |
| Endurecimiento | 37/37 | 37/37 |
| Admisión | 26/26 | 26/26 |
| Matriz de modalidad | 10/10 | 7/10 |
| Relaciones/evidencia | 8/8 | 8/8 |
| Segmentación NER segura | 4/4 | 3/4 |
| **Total ampliado** | **242/254** | **230/254** |

## Casos y seguridad semántica

- **Recuperación confirmada:** en `Vendí tomate por 6 dólares y vendí fundas por 4 dólares`, la candidata vuelve a interpretar `fundas`; el campo de evidencia apunta exactamente a `fundas` en el texto original.
- **Relaciones:** en dos fixtures de abono, la recuperación del producto `fundas` corrige el campo que impedía validar la operación relacionada. La lógica de destinos no se cambió; la referencia a la segunda compra sigue apuntando al índice esperado. Las referencias no únicas continúan ambiguas.
- **No transferir sin ancla:** `Compré cajas, pagué el transporte y vendí productos` mantiene el producto `productos` en la tercera operación y no adjudica `cajas` a la primera. El span omitido se registra con su offset.
- **Negación/futuro:** el vigente sigue admitiendo solo `comprar tomate` en `Compré tomate y no vendí cebollas`, y solo `vender arroz` en `Mañana compraré tomate y hoy vendí arroz`. Las baterías de modalidad y admisión conservan 10/10 y 26/26.
- **Candidata:** su salida NER local puede incluir `tomate y` para la primera operación en la frase negada. La proyección no crea ese span, pero tampoco puede corregir una predicción local errónea sin una política adicional; es un motivo para no promoverla.

## Pruebas ejecutadas

1. Diagnóstico experimental aislado, 65 textos, vigente y candidata, con estrategias A/B/C y variante con ancla; resultados en `experimento_segmentos_ner_ancla_2026-10-28.json`.
2. Pruebas focalizadas sobre ambos modelos para recuperación de `fundas`, offsets y destino relacional, más casos de no-ancla y modalidad; todas las aserciones focalizadas ejecutadas pasaron.
3. `python -m py_compile interpretacion.py pruebas.py resultados_ner/experimentar_segmentos_ner.py`: correcto.
4. `python pruebas.py`: matriz ampliada **242/254**, código 1 por los mismos 12 fallos de NER independiente; 0 baterías omitidas y 0 excepciones. La nueva batería pasó 4/4.
5. `entrenamiento.medir_baterias` contra modelo vigente y candidata, sin entrenar: **242/254** y **230/254**, respectivamente; resultados guardados en `comparativa_final_fase_autonoma_2026-10-28.json`.
6. `git diff --check`: sin errores de whitespace. Git avisó que los archivos previos tienen LF y podrían normalizarse a CRLF; no se hizo conversión intencional.

Registro integral: `validacion_integral_post_correccion_2026-10-28.log`.

## Archivos de esta fase

- `interpretacion.py`: proyección conservadora, reuso del Doc local y diagnósticos NER.
- `pruebas.py`: cuatro checks permanentes en la matriz integral.
- `README.md`: documentación del campo y política de proyección.
- `resultados_ner/experimentar_segmentos_ner.py`: comparación experimental ampliada.
- `resultados_ner/experimento_segmentos_ner_ancla_2026-10-28.json`: salidas por caso/estrategia.
- `resultados_ner/baseline_fase_autonoma_2026-10-28.log`: línea base previa.
- `resultados_ner/validacion_integral_post_correccion_2026-10-28.log`: batería final.
- `resultados_ner/comparativa_final_fase_autonoma_2026-10-28.json`: comparación de las 17 baterías.

Los otros archivos previamente presentes bajo `resultados_ner/` y los cambios locales iniciales se conservaron.

## Pendientes y siguiente paso recomendado

El módulo no está listo para integración ni para promover la candidata. La corrección de segmentación es conservadora y validada en las fixtures existentes, pero no resuelve falsos positivos/negativos del NER completo. El siguiente paso con mejor retorno es preparar un conjunto nuevo y revisado de ejemplos de desarrollo y evaluación final, centrado en cantidades con unidad, conceptos compuestos, spans monetarios y varias operaciones; agrupar la partición por familias lingüísticas y mantener la evaluación final sellada. Después se puede entrenar una candidata nueva en una ruta nueva y compararla con esta matriz, sin tocar `custom_ner_model` ni la candidata existente.
