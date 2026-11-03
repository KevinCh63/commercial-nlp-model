# Auditoría y limpieza

## Hallazgos corregidos

1. Se había interpretado la preparación para integración como distribución ZIP. Se retiraron el constructor de distribuciones, sus pruebas, ZIP, manifiestos y comprobaciones de plataforma/dependencias de ese mecanismo. La interfaz actual se importa directamente con ComponenteInterprete().
2. Había candidatas temporales, informes duplicados y experimentos antiguos sin uso en ejecución ni en el entrenamiento actual. Se eliminaron tras revisar sus referencias. Se conserva la evaluación vigente y una referencia de métricas del modelo; las rutas retiradas se enumeran en limpieza.json.
3. La ruta relativa del modelo dependía del directorio de ejecución. Ahora se resuelve respecto al módulo del componente, incluso al configurarla mediante INTERPRETE_MODELO. Una prueba ejecuta la carga desde un directorio temporal diferente.
4. La propiedad de versiones podía modificarse desde un consumidor. Ahora se devuelve una copia y se conserva el estado interno. La prueba modifica la copia y comprueba que la versión interna no cambie.
5. La validación de evidencia no rechazaba siempre un índice final que excediera la longitud del texto si el recorte de Python producía el mismo fragmento. Se exige 0 <= inicio < fin <= longitud y coincidencia exacta. Se agregó una prueba específica.
6. Se ordenaron pruebas, herramientas y documentación en carpetas separadas; se actualizaron imports y comandos. Los datos generados ya no contienen scripts ejecutables.
7. El modelo anterior estaba oculto en la raíz. Se trasladó a respaldos/modelo_previo; las futuras promociones también guardan sus respaldos en esa carpeta.

## Verificación

- 262/262 comprobaciones históricas aprobadas.
- 13 métodos de prueba del componente e integración local aprobados.
- 25/25 casos exploratorios con requisitos verificables aprobados, sin excepciones ni offsets inválidos.
- Compilación de los módulos modificados y git diff --check sin errores.
- Los pesos del modelo activo no se reentrenaron ni se sustituyeron.

No se deben sumar estas cifras como si fueran observaciones independientes: algunas pruebas reutilizan los casos exploratorios. Tampoco prueban exactitud perfecta en lenguaje comercial nuevo.

## Decisiones de conservación

Se conservó el formato interno completo del modelo, incluido senter, porque eliminar componentes serializados manualmente puede romper su carga. Se conservaron el modelo anterior, el entorno venv, los datos de entrenamiento y los conjuntos de evaluación de pruebas.py. El historial Git y la configuración del IDE no se limpiaron como si fueran residuos del intérprete.

## Límites pendientes

El intérprete produce propuestas y exige revisión; no confirma ni registra movimientos. Varios productos sin desglose y formatos ambiguos se señalan para revisión. Las devoluciones, el catálogo y los saldos externos no están implementados. La revisión realizada cubre código, dependencias locales, rutas, contrato y regresiones; no es una auditoría de seguridad de un backend, porque todavía no existe.
