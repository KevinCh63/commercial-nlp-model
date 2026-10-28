# Experimento: NER de oración completa frente a segmentos

**Fecha:** 2026-10-28
**Alcance:** diagnóstico aislado; sin entrenamiento, promoción de modelos ni cambios al flujo de producción.

## Pregunta

¿Cambian las entidades NER cuando una oración se procesa completa y cuando el intérprete vuelve a procesar cada segmento de operación? Se compararon el modelo vigente y la candidata guardada en `resultados_ner/candidata_temporal_2026-10-28` con tres estrategias:

- **A — comportamiento actual:** cada segmento usa las entidades de su propia inferencia local.
- **B — proyección de oración:** se usan en el segmento las entidades de la oración completa que estén totalmente contenidas en él. Las entidades que cruzan el borde no se recortan ni se asignan.
- **C — híbrida:** conserva las predicciones locales y agrega una proyección contenida solo cuando no solapa con una predicción local. Ante solapes incompatibles conserva la predicción local y registra el conflicto. Tampoco recorta spans que cruzan segmentos.

El adaptador de evaluación está en `resultados_ner/experimentar_segmentos_ner.py`. Intercepta llamadas spaCy únicamente durante esta evaluación y registra offsets originales, entidades completas/locales/proyectadas/finales, conflictos, campos semánticos, modalidad y relaciones. El contexto se reinicia para cada texto independiente.

## Datos y reproducibilidad

Se reunieron **65 textos diagnósticos únicos** de las baterías existentes de entidades, operaciones múltiples, relaciones, adversariales, fase 2, fase 4, modalidad, admisión y contrato. Se añadieron cuatro casos de seguridad para bordes y solapes; son diagnósticos, no ejemplos de entrenamiento ni nuevas expectativas de producción. El JSON conserva resultados por caso, modelo y estrategia.

## Discrepancia cruda entre inferencias

Las cifras siguientes comparan NER de frase completa con inferencia local antes de aplicar B o C; por eso no dependen de la estrategia y se reportan una sola vez por modelo.

| Medida (65 textos) | Vigente | Candidata temporal |
|---|---:|---:|
| Entidades de frase completa contenidas en un segmento | 183 | 141 |
| Ocurrencias de entidades en segmentos locales | 180 | 143 |
| Coincidencia exacta de span y etiqueta | 175 | 135 |
| Span completo contenido, ausente en la inferencia local | 8 | 6 |
| Tuplas exactas solo locales | 5 | 8 |
| Entidades completas que cruzan bordes de segmento | 1 | 9 |
| Fallos de alineación a tokens al proyectar | 0 | 0 |

La candidata presenta más casos de entidades completas que cruzan los límites. Estas entidades no son aptas para copiarse a un segmento sin una regla semántica adicional. Las cifras son ocurrencias agregadas sobre textos deduplicados, no conteos de tipos únicos ni métricas de precisión global.

## Efecto funcional por batería

Cada celda es aciertos/total del evaluador de esa batería. Las expectativas son las ya guardadas en las fixtures del proyecto; los casos de seguridad adicionales no se suman como gold.

| Modelo | Estrategia | NER | Operaciones múltiples | Relaciones | Adversariales | Fase 2 | Fase 4 | Modalidad | Admisión | Contrato | Evidencia de relaciones |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vigente | A | 27/27 | 12/12 | 7/7 | 22/22 | 11/11 | 17/17 | 10/10 | 26/26 | 8/8 | 8/8 |
| Vigente | B | 27/27 | 11/12 | 7/7 | 22/22 | 11/11 | 16/17 | 10/10 | 26/26 | 8/8 | 8/8 |
| Vigente | C | 27/27 | 11/12 | 7/7 | 22/22 | 11/11 | 17/17 | 10/10 | 26/26 | 8/8 | 8/8 |
| Candidata | A | 27/27 | 11/12 | 1/7 | 20/22 | 9/11 | 15/17 | 7/10 | 26/26 | 8/8 | 8/8 |
| Candidata | B | 27/27 | 11/12 | 3/7 | 20/22 | 9/11 | 14/17 | 6/10 | 26/26 | 8/8 | 8/8 |
| Candidata | C | 27/27 | 12/12 | 3/7 | 20/22 | 9/11 | 15/17 | 7/10 | 26/26 | 8/8 | 8/8 |

**Lectura:** B eleva las relaciones de la candidata de 1/7 a 3/7 y recupera la detección de `fundas` en dos frases con abonos. No resuelve los otros cuatro casos de relaciones: en ellos el problema no se limita a que una entidad de frase completa se pierda al segmentar. C conserva esas dos mejoras, devuelve a 12/12 las operaciones múltiples de la candidata y evita las regresiones concretas de modalidad vistas con B. En el modelo vigente, ambas proyecciones tienen regresiones, por lo que no son un cambio seguro general.

## Casos que explican los cambios

- **Recuperación por proyección:** en `Vendí tomate por 6 dólares y vendí fundas por 4 dólares`, el modelo candidato reconoce `fundas` en la oración completa, pero lo pierde al inferir el segundo segmento. B y C lo recuperan. Lo mismo sucede con `fundas` en dos frases de abono; permite pasar esos checks de relación de falso a verdadero.
- **Proyección errónea por alcance:** en `Compré cajas, pagué el transporte y vendí productos`, B deja sin producto la tercera operación, mientras A y C conservan `productos`. C registra la predicción local incompatible y la retiene.
- **Regresión de modalidad bajo B:** en `Mañana compraré tomate y hoy vendí arroz`, B pierde `arroz`, mientras A y C lo conservan. La modalidad no cambia en C.
- **Bordes y ambigüedad:** en pruebas exploratorias, spans que abarcan conectores, negación o más de un segmento se registran como cruces/conflictos y no se recortan. En C se registraron **3 conflictos** del modelo vigente y **10** de la candidata sobre los casos y ejecuciones examinados. El aumento apunta a mayor ambigüedad de spans candidata; no significa que C haya resuelto esos casos.
- **Errores que la proyección no arregla:** la candidata sigue por debajo del vigente en relaciones, adversariales, fase 2, fase 4 y modalidad. Si la inferencia completa tampoco predice la entidad correcta, proyectarla no puede recuperarla.

## Límites e interpretación

La comparación evalúa entidades y efectos semánticos en fixtures ya existentes; no mide generalización en un corpus nuevo. Las cifras por batería no deben interpretarse como evaluación independiente, porque varias baterías se solapan con los 65 textos diagnósticos y entre sí. La proyección tampoco prueba que el límite de segmento corresponda al alcance semántico correcto: una entidad contenida geométricamente puede seguir asociándose a la operación incorrecta, como evidencia la regresión de `productos`.

No se modificaron entrenamiento, interpretación de producción, fixtures ni expectativas de gold. No se entrenó ni promovió ningún modelo. Se compilaron y ejecutaron el evaluador experimental y las baterías que este invoca; no se lanzó entrenamiento ni una batería completa ajena a este experimento.

## Archivos generados

- `resultados_ner/experimentar_segmentos_ner.py`: adaptador/evaluador aislado.
- `resultados_ner/experimento_segmentos_ner_2026-10-28.json`: resultados por caso y estrategia.
- `resultados_ner/ejecucion_experimento_segmentos_ner.log`: salida de la ejecución.

## Conclusión

La evidencia confirma una distribución distinta entre NER de frase completa y NER local, con discrepancias de ambos sentidos y spans que atraviesan límites. B mejora casos puntuales de la candidata pero añade regresiones; C mitiga los ejemplos observados mediante retención local y registro de conflictos, aunque no es una recomendación de promoción automática. El paso siguiente es revisar los casos marcados como conflicto y las entidades candidatas ausentes incluso en la oración completa antes de considerar cualquier cambio al flujo de producción.
