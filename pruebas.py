"""Batería de pruebas NER centrada en generalización."""


CATEGORIAS_POR_TEXTO = {}


def _crear_prueba(texto, categoria, entidades):
    """Construye una prueba y calcula offsets a partir de sus fragmentos."""
    anotaciones = []
    for fragmento, etiqueta in entidades:
        inicio = texto.index(fragmento)
        anotaciones.append((inicio, inicio + len(fragmento), etiqueta))
    CATEGORIAS_POR_TEXTO[texto] = categoria
    return texto, {"entities": anotaciones}


# Todas estas frases son distintas de las utilizadas en train_data.
# El formato de cada elemento se conserva para que entrenamiento.py pueda
# validar y evaluar la batería sin cambios.
PRUEBAS_NUEVAS = [
    _crear_prueba("Vendí 7 botellas de agua por 12 dólares", "VENTAS", [("7 botellas", "QUANTITY"), ("agua", "PRODUCT"), ("12 dólares", "PRICE")]),
    _crear_prueba("Se vendieron 4 cajas de galletas por 18 dólares", "VENTAS", [("4 cajas", "QUANTITY"), ("galletas", "PRODUCT"), ("18 dólares", "PRICE")]),
    _crear_prueba("Por 27 USD, vendimos 9 cuadernos", "VENTAS", [("27 USD", "PRICE"), ("9", "QUANTITY"), ("cuadernos", "PRODUCT")]),
    _crear_prueba("Ellos vendieron 6 lámparas a 45 dólares", "VENTAS", [("6", "QUANTITY"), ("lámparas", "PRODUCT"), ("45 dólares", "PRICE")]),
    _crear_prueba("Ana vendió 12 vasos por 9 euros", "VENTAS", [("12", "QUANTITY"), ("vasos", "PRODUCT"), ("9 euros", "PRICE")]),
    _crear_prueba("María adquirió 3 cajas de papel por 20 dólares", "COMPRAS", [("3 cajas", "QUANTITY"), ("papel", "PRODUCT"), ("20 dólares", "PRICE")]),
    _crear_prueba("Compramos varios artículos por 35 dólares", "COMPRAS", [("varios", "QUANTITY"), ("artículos", "PRODUCT"), ("35 dólares", "PRICE")]),
    _crear_prueba("Jorge compró 2 mochilas por 70 dólares", "COMPRAS", [("2", "QUANTITY"), ("mochilas", "PRODUCT"), ("70 dólares", "PRICE")]),
    _crear_prueba("Adquirí 4 cables por 10 USD", "COMPRAS", [("4", "QUANTITY"), ("cables", "PRODUCT"), ("10 USD", "PRICE")]),
    _crear_prueba("Pagamos 25 dólares por transporte", "EGRESOS / GASTOS", [("25 dólares", "PRICE"), ("transporte", "CONCEPT")]),
    _crear_prueba("Se gastaron 40 dólares en suministros", "EGRESOS / GASTOS", [("40 dólares", "PRICE"), ("suministros", "CONCEPT")]),
    _crear_prueba("Lucía desembolsó 18 dólares para papelería", "EGRESOS / GASTOS", [("18 dólares", "PRICE"), ("papelería", "CONCEPT")]),
    _crear_prueba("Nos gastamos 9 euros en detergente", "EGRESOS / GASTOS", [("9 euros", "PRICE"), ("detergente", "CONCEPT")]),
    _crear_prueba("Pedro me entregó 15 dólares", "INGRESOS / COBROS", [("15 dólares", "PRICE")]),
    _crear_prueba("El cliente abonó 8 dólares", "INGRESOS / COBROS", [("8 dólares", "PRICE")]),
    _crear_prueba("Nos pagaron 32 USD", "INGRESOS / COBROS", [("32 USD", "PRICE")]),
    _crear_prueba("Laura recibió 22 dólares", "INGRESOS / COBROS", [("22 dólares", "PRICE")]),
    _crear_prueba("Yo vendí 3 mochilas por 24 dólares", "PERSONA: PRIMERA SINGULAR", [("3", "QUANTITY"), ("mochilas", "PRODUCT"), ("24 dólares", "PRICE")]),
    _crear_prueba("Nosotros compramos 5 vasos por 11 dólares", "PERSONA: PRIMERA PLURAL", [("5", "QUANTITY"), ("vasos", "PRODUCT"), ("11 dólares", "PRICE")]),
    _crear_prueba("Ella vendió 8 lámparas por 90 dólares", "PERSONA: TERCERA SINGULAR", [("8", "QUANTITY"), ("lámparas", "PRODUCT"), ("90 dólares", "PRICE")]),
    _crear_prueba("Ellos compraron 6 cuadernos por 42 dólares", "PERSONA: TERCERA PLURAL", [("6", "QUANTITY"), ("cuadernos", "PRODUCT"), ("42 dólares", "PRICE")]),
    _crear_prueba("20 dólares por 5 vasos cobré", "ORDEN DE PALABRAS", [("20 dólares", "PRICE"), ("5", "QUANTITY"), ("vasos", "PRODUCT")]),
    _crear_prueba("Por 30 dólares compró Elena 2 lámparas", "ORDEN DE PALABRAS", [("30 dólares", "PRICE"), ("2", "QUANTITY"), ("lámparas", "PRODUCT")]),
    _crear_prueba("De 12 cuadernos, vendimos 24 dólares", "ORDEN DE PALABRAS", [("12", "QUANTITY"), ("cuadernos", "PRODUCT"), ("24 dólares", "PRICE")]),
    _crear_prueba("Vendí cinco cuadernos por treinta dólares", "CASO DIFÍCIL", [("cinco", "QUANTITY"), ("cuadernos", "PRODUCT"), ("treinta dólares", "PRICE")]),
    _crear_prueba("Compré cables por 12", "CASO DIFÍCIL", [("cables", "PRODUCT"), ("12", "PRICE")]),
    _crear_prueba("Se vendió una mochila", "CASO DIFÍCIL", [("una", "QUANTITY"), ("mochila", "PRODUCT")]),
]


# Estas pruebas no comparan offsets NER: comprueban la interpretación final.
# Solo se exigen campos que el texto permite justificar.
PRUEBAS_SEMANTICAS = [
    # Casos críticos de regresión solicitados para la primera versión.
    ("Vendí 7 botellas de agua por 12 dólares", {
        "agente": "YO", "accion": "vender", "tipo": "INGRESO",
        "cantidad": "7 botellas", "producto": "agua", "precio": "12 dólares",
    }),
    ("Se vendieron 4 cajas de galletas por 18 dólares", {
        "agente": None, "accion": "vender", "tipo": "INGRESO",
        "cantidad": "4 cajas", "producto": "galletas", "precio": "18 dólares",
    }),
    ("María adquirió 3 cajas de papel por 20 dólares", {
        "agente": "María", "accion": "comprar", "tipo": "EGRESO",
        "cantidad": "3 cajas", "producto": "papel", "precio": "20 dólares",
    }),
    ("Compré cables por 12", {
        "agente": "YO", "accion": "comprar", "tipo": "EGRESO",
        "producto": "cables", "precio": "12",
    }),
    ("Se vendió una mochila", {
        "agente": None, "accion": "vender", "tipo": "INGRESO",
        "cantidad": "una", "producto": "mochila",
    }),
    ("Gasté 5 dólares en transporte", {
        "agente": "YO", "accion": "gastar", "tipo": "EGRESO",
        "concepto": "transporte", "precio": "5 dólares",
    }),
    ("Pagué 10 dólares al proveedor", {
        "agente": "YO", "accion": "pagar", "tipo": "EGRESO",
        "contraparte": "proveedor", "precio": "10 dólares",
    }),
    ("Carlos quedó debiendo 10 dólares", {
        "agente": "Carlos", "accion": "deber", "tipo": None,
        "precio": "10 dólares", "estado": "AMBIGUO",
    }),
    ("Vendí 3 libras de tomate a 2 dólares", {
        "agente": "YO", "accion": "vender", "tipo": "INGRESO",
        "cantidad": "3 libras", "producto": "tomate", "precio": "2 dólares",
    }),
    ("María compró un producto", {
        "agente": "María", "accion": "comprar", "tipo": "EGRESO",
        "cantidad": "un", "producto": "producto",
    }),
    ("Carlos compró 5 camisetas por 30 dólares", {
        "agente": "Carlos", "accion": "comprar", "tipo": "EGRESO",
        "cantidad": "5", "producto": "camisetas", "precio": "30 dólares",
    }),
    ("Gasté 5 dólares en transporte", {
        "agente": "YO", "accion": "gastar", "tipo": "EGRESO",
        "concepto": "transporte", "precio": "5 dólares",
    }),
    ("Carlos me abonó 5 dólares", {
        "agente": "Carlos", "accion": "abonar", "tipo": "INGRESO",
        "precio": "5 dólares",
    }),
    ("Pagué 10 dólares al proveedor", {
        "agente": "YO", "accion": "pagar", "tipo": "EGRESO",
        "precio": "10 dólares",
    }),
    ("María vendió 4 productos", {
        "agente": "María", "accion": "vender", "tipo": "INGRESO",
        "cantidad": "4", "producto": "productos",
    }),
    ("Compré un producto", {
        "agente": "YO", "accion": "comprar", "tipo": "EGRESO",
        "cantidad": "un", "producto": "producto",
    }),
    ("Carlos quedó debiendo 10 dólares", {
        "agente": "Carlos", "accion": "deber", "tipo": None,
        "precio": "10 dólares",
    }),
    ("Se vendieron 4 cajas de galletas por 18 dólares", {
        "agente": None, "accion": "vender", "tipo": "INGRESO",
        "cantidad": "4 cajas", "producto": "galletas", "precio": "18 dólares",
    }),
    # Variantes para verificar voz, persona, campos desconocidos y dirección del dinero.
    ("Vendimos 4 cajas de galletas por 18 dólares", {
        "agente": "NOSOTROS", "accion": "vender", "tipo": "INGRESO",
        "cantidad": "4 cajas", "producto": "galletas", "precio": "18 dólares",
    }),
    ("Quedé debiendo 20 dólares", {
        "agente": "YO", "accion": "deber", "tipo": None, "precio": "20 dólares",
    }),
    ("Gasté 10 dólares", {
        "agente": "YO", "accion": "gastar", "tipo": "EGRESO",
        "cantidad": None, "producto": None, "concepto": None, "precio": "10 dólares",
    }),
    ("Pagué 5 dólares", {
        "agente": "YO", "accion": "pagar", "tipo": "EGRESO",
        "cantidad": None, "producto": None, "concepto": None, "precio": "5 dólares",
    }),
    ("Juan me pagó 10 dólares", {
        "agente": "Juan", "accion": "pagar", "tipo": "INGRESO", "precio": "10 dólares",
    }),
    ("María compró 3 cajas de papel", {
        "agente": "María", "accion": "comprar", "tipo": "EGRESO",
        "cantidad": "3 cajas", "producto": "papel", "precio": None,
    }),
    ("Vendí 4 botellas a 2 dólares cada una", {
        "agente": "YO", "accion": "vender", "tipo": "INGRESO",
        "cantidad": "4 botellas", "producto": None, "concepto": None, "precio": "2 dólares",
    }),
    ("Pagué 20 dólares de una deuda de 50", {
        "agente": "YO", "accion": "pagar", "tipo": "EGRESO",
        "cantidad": None, "producto": None, "concepto": None, "precio": "20 dólares",
    }),
    ("María compró 3 cajas y pagó 15 dólares", {
        "agente": "María", "accion": "comprar", "tipo": "EGRESO",
        "cantidad": "3 cajas", "producto": None, "concepto": None, "precio": "15 dólares",
    }),
    ("Gasté 10 dólares en transporte", {
        "agente": "YO", "accion": "gastar", "tipo": "EGRESO",
        "cantidad": None, "producto": None, "concepto": "transporte", "precio": "10 dólares",
    }),
    ("Se vendieron 2 cajas", {
        "agente": None, "accion": "vender", "tipo": "INGRESO",
        "cantidad": "2 cajas", "producto": None, "concepto": None, "precio": None,
    }),
]

CAMPOS_PRINCIPALES = ("agente", "accion", "tipo", "cantidad", "producto", "concepto", "precio")

# Casos de reconocimiento aparte de la interpretación: aquí se compara el
# texto y la etiqueta NER, mientras que las reglas pueden corregir roles después.
PRUEBAS_ENTIDADES = [
    ("Vendí 3 libras de tomate a 2 dólares", [("3 libras", "QUANTITY"), ("tomate", "PRODUCT"), ("2 dólares", "PRICE")]),
    ("Se vendieron 4 cajas de galletas por 18 dólares", [("4 cajas", "QUANTITY"), ("galletas", "PRODUCT"), ("18 dólares", "PRICE")]),
    ("María compró 3 cajas de papel", [("3 cajas", "QUANTITY")]),
    ("Gasté 10 dólares", [("10 dólares", "PRICE")]),
    ("Quedé debiendo 20 dólares", [("20 dólares", "PRICE")]),
]

PRUEBAS_OPERACIONES_MULTIPLES = [
    ("Gasté 10 dólares en transporte", [
        {"agente": "YO", "accion": "gastar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": "transporte", "precio": "10 dólares"},
    ]),
    ("Vendí tomate por 6 dólares y compré fundas por 4", [
        {"agente": "YO", "accion": "vender", "tipo": "INGRESO", "cantidad": None, "producto": "tomate", "concepto": None, "precio": "6 dólares"},
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": "fundas", "concepto": None, "precio": "4"},
    ]),
    ("Vendí tomate por 6 dólares y vendí fundas por 4 dólares", [
        {"agente": "YO", "accion": "vender", "tipo": "INGRESO", "cantidad": None, "producto": "tomate", "concepto": None, "precio": "6 dólares"},
        {"agente": "YO", "accion": "vender", "tipo": "INGRESO", "cantidad": None, "producto": "fundas", "concepto": None, "precio": "4 dólares"},
    ]),
    ("Vendí tomate por 6 dólares y gasté 2 dólares en transporte", [
        {"agente": "YO", "accion": "vender", "tipo": "INGRESO", "cantidad": None, "producto": "tomate", "concepto": None, "precio": "6 dólares"},
        {"agente": "YO", "accion": "gastar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": "transporte", "precio": "2 dólares"},
    ]),
    ("Compré cajas, pagué el transporte y vendí productos", [
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": None},
        {"agente": "YO", "accion": "pagar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": "transporte", "precio": None},
        {"agente": "YO", "accion": "vender", "tipo": "INGRESO", "cantidad": None, "producto": "productos", "concepto": None, "precio": None},
    ]),
    ("Vendí 3 libras de tomate por 6 dólares, compré 2 cajas por 4 dólares y pagué 1 dólar de transporte", [
        {"agente": "YO", "accion": "vender", "tipo": "INGRESO", "cantidad": "3 libras", "producto": "tomate", "concepto": None, "precio": "6 dólares"},
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": "2 cajas", "producto": None, "concepto": None, "precio": "4 dólares"},
        {"agente": "YO", "accion": "pagar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": "transporte", "precio": "1 dólar"},
    ]),
    ("Compré 3 cajas y pagué 15 dólares", [
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": "3 cajas", "producto": None, "concepto": None, "precio": None},
        {"agente": "YO", "accion": "pagar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": "15 dólares"},
    ]),
    ("Pagué 10 dólares de una deuda de 30", [
        {"agente": "YO", "accion": "pagar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": "10 dólares"},
    ]),
    ("Se vendieron 2 cajas", [
        {"agente": None, "accion": "vender", "tipo": "INGRESO", "cantidad": "2 cajas", "producto": None, "concepto": None, "precio": None},
    ]),
    ("Vendí tomate por 6 dólares y compré", [
        {"agente": "YO", "accion": "vender", "tipo": "INGRESO", "cantidad": None, "producto": "tomate", "concepto": None, "precio": "6 dólares"},
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": None, "estado": "PARCIAL"},
    ]),
    ("Tal vez hubo una venta", []),
    ("Compré y pagué", [
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": None, "estado": "PARCIAL"},
        {"agente": "YO", "accion": "pagar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": None, "estado": "PARCIAL"},
    ]),
]

PRUEBAS_RELACIONES = [
    ("Compré mercadería por 100 dólares y aboné 30 dólares de esa compra", [
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": "mercadería", "concepto": None, "precio": "100 dólares"},
        {"agente": "YO", "accion": "abonar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": "30 dólares"},
    ], [{"operacion_origen": 1, "operacion_destino": 0, "tipo": "ABONO_DE", "evidencia": "de esa compra", "estado": "CONFIRMADA"}]),
    ("Compré mercadería por 100 dólares y gasté 5 dólares en transporte", [
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": "mercadería", "concepto": None, "precio": "100 dólares"},
        {"agente": "YO", "accion": "gastar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": "transporte", "precio": "5 dólares"},
    ], []),
    ("Compré mercadería por 100 dólares y pagué 30 dólares", [
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": "mercadería", "concepto": None, "precio": "100 dólares"},
        {"agente": "YO", "accion": "pagar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": "30 dólares"},
    ], []),
    ("Compré papel por 100 dólares y compré fundas por 50 dólares y aboné 10 dólares de esa segunda compra", [
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": "papel", "concepto": None, "precio": "100 dólares"},
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": "fundas", "concepto": None, "precio": "50 dólares"},
        {"agente": "YO", "accion": "abonar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": "10 dólares"},
    ], [{"operacion_origen": 2, "operacion_destino": 1, "tipo": "ABONO_DE", "evidencia": "de esa segunda compra", "estado": "CONFIRMADA"}]),
    ("Compré papel por 100 dólares y compré fundas por 50 dólares y aboné 10 dólares de esa compra", [
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": "papel", "concepto": None, "precio": "100 dólares"},
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": "fundas", "concepto": None, "precio": "50 dólares"},
        {"agente": "YO", "accion": "abonar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": "10 dólares"},
    ], [{"operacion_origen": 2, "operacion_destino": None, "tipo": "ABONO_DE", "evidencia": "de esa compra", "estado": "AMBIGUA"}]),
    ("Compré papel por 100 dólares y aboné 30 dólares de esa compra", [
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": "papel", "concepto": None, "precio": "100 dólares"},
        {"agente": "YO", "accion": "abonar", "tipo": "EGRESO", "cantidad": None, "producto": None, "concepto": None, "precio": "30 dólares"},
    ], [{"operacion_origen": 1, "operacion_destino": 0, "tipo": "ABONO_DE", "evidencia": "de esa compra", "estado": "CONFIRMADA"}]),
    ("Compré papel por 100 dólares y vendí fundas por 50 dólares", [
        {"agente": "YO", "accion": "comprar", "tipo": "EGRESO", "cantidad": None, "producto": "papel", "concepto": None, "precio": "100 dólares"},
        {"agente": "YO", "accion": "vender", "tipo": "INGRESO", "cantidad": None, "producto": "fundas", "concepto": None, "precio": "50 dólares"},
    ], []),
]


def evaluar_ner(nlp, pruebas, titulo, detallado=False):
    """Evalúa entidades con comparación exacta y reporta fallos detallados."""
    print(f"\n{titulo}")
    print("-" * 60)

    aciertos = 0
    resultados_por_categoria = {}

    for texto, anotaciones in pruebas:
        doc = nlp(texto)
        esperado = sorted(anotaciones["entities"])
        obtenido = sorted((e.start_char, e.end_char, e.label_) for e in doc.ents)
        correcto = esperado == obtenido
        categoria = CATEGORIAS_POR_TEXTO.get(texto, "SIN CATEGORÍA")

        resultados_por_categoria.setdefault(categoria, [0, 0])
        resultados_por_categoria[categoria][1] += 1
        if correcto:
            aciertos += 1
            resultados_por_categoria[categoria][0] += 1
            if detallado:
                print(f"[OK] {texto!r}")
        else:
            print(f"[FALLO] {texto!r} | categoría: {categoria}")
            print(f"        esperadas: {esperado}")
            print(f"        encontradas: {obtenido}")

    print("\nResultados por categoría:")
    for categoria, (categoria_aciertos, categoria_total) in resultados_por_categoria.items():
        print(f"- {categoria}: {categoria_aciertos}/{categoria_total}")
    print(f"\nResultado total: {aciertos}/{len(pruebas)} frases correctas")
    return aciertos


def evaluar_semantica(nlp, pruebas=PRUEBAS_SEMANTICAS):
    """Comprueba campos semánticos explícitos y muestra cada caso."""
    print("\nPRUEBAS SEMÁNTICAS")
    print("-" * 60)
    aciertos = 0
    for texto, esperado in pruebas:
        from interpretacion import resultado_semantico, interpretar_texto

        obtenido = resultado_semantico(interpretar_texto(texto, nlp=nlp))
        # Todos los casos verifican el contrato uniforme de siete campos.
        esperado_completo = {campo: None for campo in CAMPOS_PRINCIPALES}
        esperado_completo.update(esperado)
        correcto = all(obtenido.get(campo) == valor for campo, valor in esperado_completo.items())
        if correcto:
            aciertos += 1
            print(f"[OK] {texto!r}")
        else:
            print(f"[FALLO] {texto!r}")
            print(f"        esperados: {esperado}")
            print(f"        obtenido:  {obtenido}")
    print(f"\nResultado semántico: {aciertos}/{len(pruebas)} frases correctas")
    return aciertos


def evaluar_entidades(nlp, pruebas=PRUEBAS_ENTIDADES):
    print("\nPRUEBAS DE ENTIDADES")
    print("-" * 60)
    aciertos = 0
    for texto, esperado in pruebas:
        obtenido = [(ent.text, ent.label_) for ent in nlp(texto).ents]
        correcto = all(par in obtenido for par in esperado)
        if correcto:
            aciertos += 1
            print(f"[OK] {texto!r}")
        else:
            print(f"[FALLO] {texto!r} | esperadas: {esperado} | obtenidas: {obtenido}")
    print(f"\nResultado de entidades: {aciertos}/{len(pruebas)} casos correctos")
    return aciertos


def evaluar_operaciones_multiples(nlp, pruebas=PRUEBAS_OPERACIONES_MULTIPLES):
    from interpretacion import interpretar_operaciones

    print("\nPRUEBAS DE OPERACIONES MÚLTIPLES")
    print("-" * 60)
    aciertos = 0
    for texto, esperadas in pruebas:
        items_obtenidos = interpretar_operaciones(texto, nlp=nlp)
        obtenidas = [
            {campo: item.get(campo) for campo in (*CAMPOS_PRINCIPALES, "estado") if campo in esperada}
            for item, esperada in zip(items_obtenidos, esperadas)
        ]
        correcto = len(items_obtenidos) == len(esperadas) and obtenidas == esperadas
        if correcto:
            aciertos += 1
            print(f"[OK] {texto!r}")
        else:
            print(f"[FALLO] {texto!r}")
            print(f"        esperadas: {esperadas}")
            print(f"        obtenidas: {obtenidas}")
    print(f"\nResultado múltiples: {aciertos}/{len(pruebas)} casos correctos")
    return aciertos


def evaluar_relaciones(nlp, pruebas=PRUEBAS_RELACIONES):
    from interpretacion import interpretar_operaciones_con_relaciones

    print("\nPRUEBAS DE RELACIONES ENTRE OPERACIONES")
    print("-" * 60)
    aciertos = 0
    for texto, operaciones_esperadas, relaciones_esperadas in pruebas:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        operaciones = [
            {campo: operacion.get(campo) for campo in CAMPOS_PRINCIPALES}
            for operacion in resultado["operaciones"]
        ]
        relaciones_esperadas_con_id = []
        for relacion in relaciones_esperadas:
            relacion_con_id = dict(relacion)
            origen = relacion_con_id["operacion_origen"]
            destino = relacion_con_id["operacion_destino"]
            relacion_con_id["operacion_origen_id"] = f"op_{origen + 1}"
            relacion_con_id["operacion_destino_id"] = f"op_{destino + 1}" if destino is not None else None
            relaciones_esperadas_con_id.append(relacion_con_id)
        correcto = (
            operaciones == operaciones_esperadas
            and resultado["relaciones"] == relaciones_esperadas_con_id
        )
        if correcto:
            aciertos += 1
            print(f"[OK] {texto!r}")
        else:
            print(f"[FALLO] {texto!r}")
            print(f"        operaciones esperadas: {operaciones_esperadas}")
            print(f"        operaciones obtenidas: {operaciones}")
            print(f"        relaciones esperadas: {relaciones_esperadas_con_id}")
            print(f"        relaciones obtenidas: {resultado['relaciones']}")
    print(f"\nResultado relaciones: {aciertos}/{len(pruebas)} casos correctos")
    return aciertos


PRUEBAS_CONTRATO = [
    ("Vendí tomates", "PARCIAL", 1, "precio", "AUSENTE"),
    ("Compré 3 cajas por 15 dólares", "AMBIGUO", 1, "precio", "AMBIGUO"),
    ("Vendí 4 botellas a 2 dólares cada una", "AMBIGUO", 1, "precio", "IDENTIFICADO"),
    ("Vendí 3 cajas por 12 dólares y compré 2 paquetes por 7 dólares", "AMBIGUO", 2, "precio", "AMBIGUO"),
    ("Vendí 4 productos por 20 dólares y después aboné 8 dólares a una deuda anterior", "AMBIGUO", 2, "relacion", "AMBIGUA"),
    ("Compré mercadería y luego pagué una deuda", "AMBIGUO", 2, "relacion", "AMBIGUA"),
    ("Tal vez hubo una venta", "NO_RECONOCIDO", 0, None, None),
    ("", "NO_RECONOCIDO", 0, None, None),
]


def evaluar_contrato(nlp, pruebas=PRUEBAS_CONTRATO):
    from interpretacion import interpretar_operaciones_con_relaciones

    print("\nPRUEBAS DEL CONTRATO ESTRUCTURADO")
    print("-" * 60)
    aciertos = 0
    for texto, estado_esperado, cuenta_esperada, campo, estado_campo in pruebas:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        correcto = (
            resultado.get("texto") == texto
            and resultado["estado"] == estado_esperado
            and len(resultado["operaciones"]) == cuenta_esperada
        )
        if campo == "precio" and resultado["operaciones"]:
            correcto = correcto and resultado["operaciones"][0]["campos"][campo]["estado"] == estado_campo
        if texto == "Compré 3 cajas por 15 dólares" and resultado["operaciones"]:
            correcto = correcto and resultado["operaciones"][0]["campos"]["precio"]["ambito"] == "DESCONOCIDO"
        if texto == "Vendí 4 botellas a 2 dólares cada una" and resultado["operaciones"]:
            correcto = correcto and resultado["operaciones"][0]["campos"]["precio"]["ambito"] == "UNITARIO"
        if campo == "relacion":
            correcto = correcto and any(rel["estado"] == estado_campo for rel in resultado["relaciones"])
        ids = {operacion["id"] for operacion in resultado["operaciones"]}
        correcto = correcto and len(ids) == len(resultado["operaciones"])
        correcto = correcto and all(
            all(campo in operacion for campo in CAMPOS_PRINCIPALES)
            and set(operacion["campos"]) == set(CAMPOS_PRINCIPALES)
            and all(operacion["campos"][campo]["valor"] == operacion[campo] for campo in CAMPOS_PRINCIPALES)
            for operacion in resultado["operaciones"]
        )
        correcto = correcto and all(
            rel["operacion_origen_id"] in ids
            and (rel["operacion_destino_id"] is None or rel["operacion_destino_id"] in ids)
            for rel in resultado["relaciones"]
        )
        if correcto:
            aciertos += 1
            print(f"[OK] {texto!r}")
        else:
            print(f"[FALLO] {texto!r} | estado={resultado['estado']}, operaciones={len(resultado['operaciones'])}, relaciones={resultado['relaciones']}")
    print(f"\nResultado contrato: {aciertos}/{len(pruebas)} casos correctos")
    return aciertos


def _operacion_esperada(agente=None, accion=None, tipo=None, cantidad=None, producto=None, concepto=None, precio=None, ambito=None):
    return {
        "agente": agente, "accion": accion, "tipo": tipo, "cantidad": cantidad,
        "producto": producto, "concepto": concepto, "precio": precio, "ambito": ambito,
    }


# Conjunto independiente de evaluación adversarial. Las frases y expectativas
# se definieron para esta fase; no se reutilizan los ejemplos de entrenamiento
# ni los fixtures de las suites de regresión anteriores.
EVALUACION_ADVERSARIAL = [
    # A. Variaciones lingüísticas
    {"categoria": "variaciones", "texto": "Hoy salieron 18 dólares en transporte", "operaciones": [_operacion_esperada(None, "gastar", "EGRESO", None, None, "transporte", "18 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    {"categoria": "variaciones", "texto": "Me entraron 45 dólares por la venta de frutas", "operaciones": [_operacion_esperada(None, "recibir", "INGRESO", None, "frutas", None, "45 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    {"categoria": "variaciones", "texto": "Le entregué 20 dólares al proveedor", "operaciones": [_operacion_esperada("YO", "entregar", "EGRESO", None, None, None, "20 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    {"categoria": "variaciones", "texto": "Por las verduras recibí 32 dólares", "operaciones": [_operacion_esperada("YO", "recibir", "INGRESO", None, "verduras", None, "32 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    {"categoria": "variaciones", "texto": "Gasté 9 dólares en fundas", "operaciones": [_operacion_esperada("YO", "gastar", "EGRESO", None, None, "fundas", "9 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    # B. Operaciones separadas por conectores variados
    {"categoria": "separacion", "texto": "Vendí limones por 14 dólares; después compré envases por 6 dólares", "operaciones": [_operacion_esperada("YO", "vender", "INGRESO", None, "limones", None, "14 dólares", "NO_ESPECIFICADO"), _operacion_esperada("YO", "comprar", "EGRESO", None, "envases", None, "6 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    {"categoria": "separacion", "texto": "Vendí clavos por 9 dólares; además gasté 2 dólares en embalaje", "operaciones": [_operacion_esperada("YO", "vender", "INGRESO", None, "clavos", None, "9 dólares", "NO_ESPECIFICADO"), _operacion_esperada("YO", "gastar", "EGRESO", None, None, "embalaje", "2 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    {"categoria": "separacion", "texto": "Compré cables por 8 dólares, pero vendí alambre por 13 dólares", "operaciones": [_operacion_esperada("YO", "comprar", "EGRESO", None, "cables", None, "8 dólares", "NO_ESPECIFICADO"), _operacion_esperada("YO", "vender", "INGRESO", None, "alambre", None, "13 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    # C/D. Relaciones explícitas e identificación ambigua del destino
    {"categoria": "relaciones_inequivocas", "texto": "Adquirí jabón por 12 dólares y luego aboné 4 dólares de aquella compra", "operaciones": [_operacion_esperada("YO", "comprar", "EGRESO", None, "jabón", None, "12 dólares", "NO_ESPECIFICADO"), _operacion_esperada("YO", "abonar", "EGRESO", None, None, None, "4 dólares", "NO_ESPECIFICADO")], "relaciones": [(1, 0, "ABONO_DE", "CONFIRMADA")]},
    {"categoria": "relaciones_inequivocas", "texto": "Compré tela por 20 dólares y pagué 7 dólares de la compra de tela", "operaciones": [_operacion_esperada("YO", "comprar", "EGRESO", None, "tela", None, "20 dólares", "NO_ESPECIFICADO"), _operacion_esperada("YO", "pagar", "EGRESO", None, None, None, "7 dólares", "NO_ESPECIFICADO")], "relaciones": [(1, 0, "PAGO_DE", "CONFIRMADA")]},
    {"categoria": "relaciones_ambiguas", "texto": "Adquirí café por 10 dólares y compré frascos por 6 dólares; aboné 2 dólares de esa adquisición", "operaciones": [_operacion_esperada("YO", "comprar", "EGRESO", None, "café", None, "10 dólares", "NO_ESPECIFICADO"), _operacion_esperada("YO", "comprar", "EGRESO", None, "frascos", None, "6 dólares", "NO_ESPECIFICADO"), _operacion_esperada("YO", "abonar", "EGRESO", None, None, None, "2 dólares", "NO_ESPECIFICADO")], "relaciones": [(2, None, "ABONO_DE", "AMBIGUA")]},
    {"categoria": "relaciones_ambiguas", "texto": "Compré tinte por 11 dólares y aboné 3 dólares de esa compra", "operaciones": [_operacion_esperada("YO", "comprar", "EGRESO", None, "tinte", None, "11 dólares", "NO_ESPECIFICADO"), _operacion_esperada("YO", "abonar", "EGRESO", None, None, None, "3 dólares", "NO_ESPECIFICADO")], "relaciones": [(1, 0, "ABONO_DE", "CONFIRMADA")]},
    # E. Cercanía sin relación explícita y relaciones inexistentes
    {"categoria": "sin_relacion", "texto": "Compré vasijas por 20 dólares y gasté 8 dólares en renta", "operaciones": [_operacion_esperada("YO", "comprar", "EGRESO", None, "vasijas", None, "20 dólares", "NO_ESPECIFICADO"), _operacion_esperada("YO", "gastar", "EGRESO", None, None, "renta", "8 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    {"categoria": "sin_relacion", "texto": "Vendí lana por 16 dólares y pagué 5 dólares", "operaciones": [_operacion_esperada("YO", "vender", "INGRESO", None, "lana", None, "16 dólares", "NO_ESPECIFICADO"), _operacion_esperada("YO", "pagar", "EGRESO", None, None, None, "5 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    # F. Cantidades, alcances e importes independientes
    {"categoria": "importes", "texto": "Vendí 5 botellas de aceite a 2 dólares cada una", "operaciones": [_operacion_esperada("YO", "vender", "INGRESO", "5 botellas", "aceite", None, "2 dólares", "UNITARIO")], "relaciones": []},
    {"categoria": "importes", "texto": "Vendí 5 cajas de té por un total de 40 dólares", "operaciones": [_operacion_esperada("YO", "vender", "INGRESO", "5 cajas", "té", None, "40 dólares", "TOTAL")], "relaciones": []},
    {"categoria": "importes", "texto": "Vendí 2 lámparas por 20 dólares y compré 3 cables por 20 dólares", "operaciones": [_operacion_esperada("YO", "vender", "INGRESO", "2", "lámparas", None, "20 dólares", "DESCONOCIDO"), _operacion_esperada("YO", "comprar", "EGRESO", "3", "cables", None, "20 dólares", "DESCONOCIDO")], "relaciones": []},
    {"categoria": "importes", "texto": "Vendí cuatro cuadernos, pero todavía no fijé el importe", "operaciones": [_operacion_esperada("YO", "vender", "INGRESO", "cuatro", "cuadernos", None, None, "NO_ESPECIFICADO")], "relaciones": []},
    # G. Entradas difíciles o insuficientes
    {"categoria": "entradas_dificiles", "texto": "compre 2 vasos por 5 dolares", "operaciones": [_operacion_esperada(None, "comprar", "EGRESO", "2", "vasos", None, "5 dolares", "DESCONOCIDO")], "relaciones": []},
    {"categoria": "entradas_dificiles", "texto": "Aboné 5 dólares de esa", "operaciones": [_operacion_esperada("YO", "abonar", "EGRESO", None, None, None, "5 dólares", "NO_ESPECIFICADO")], "relaciones": []},
    {"categoria": "entradas_dificiles", "texto": "Venta de artículos, aún sin registrar", "operaciones": [], "relaciones": []},
    {"categoria": "entradas_dificiles", "texto": "Vendi cinco cajas por dies dólares", "operaciones": [], "relaciones": []},
]


def evaluar_conjunto_independiente(nlp, pruebas=EVALUACION_ADVERSARIAL):
    from interpretacion import interpretar_operaciones_con_relaciones

    print("\nEVALUACIÓN ADVERSARIAL INDEPENDIENTE")
    print("-" * 60)
    conteos = {}
    resultados_relacion = {"tp": 0, "fp": 0, "fn": 0, "ambiguas_erroneas": 0}
    errores_importe = 0
    errores = []
    for caso in pruebas:
        salida = interpretar_operaciones_con_relaciones(caso["texto"], nlp=nlp)
        ops_obtenidas = []
        for operacion in salida["operaciones"]:
            precio_meta = operacion.get("campos", {}).get("precio", {})
            ops_obtenidas.append({
                campo: operacion.get(campo)
                for campo in CAMPOS_PRINCIPALES
            } | {"ambito": precio_meta.get("ambito")})
        relaciones_obtenidas = [
            (r["operacion_origen"], r["operacion_destino"], r["tipo"], r["estado"])
            for r in salida["relaciones"]
        ]
        correcto = ops_obtenidas == caso["operaciones"] and relaciones_obtenidas == caso["relaciones"]
        categoria = caso["categoria"]
        conteos.setdefault(categoria, [0, 0])
        conteos[categoria][1] += 1
        conteos[categoria][0] += int(correcto)
        if not correcto:
            errores.append((caso, ops_obtenidas, relaciones_obtenidas, salida))
            if len(ops_obtenidas) != len(caso["operaciones"]):
                errores_importe += 1
            else:
                for actual, esperado in zip(ops_obtenidas, caso["operaciones"]):
                    if any(actual.get(campo) != esperado.get(campo) for campo in ("cantidad", "producto", "precio", "ambito")):
                        errores_importe += 1
                        break
        expected_confirmed = [r for r in caso["relaciones"] if r[3] == "CONFIRMADA"]
        predicted_confirmed = [r for r in relaciones_obtenidas if r[3] == "CONFIRMADA"]
        expected_ambiguous = [r for r in caso["relaciones"] if r[3] == "AMBIGUA"]
        resultados_relacion["tp"] += sum(r in expected_confirmed for r in predicted_confirmed)
        resultados_relacion["fp"] += sum(r not in expected_confirmed for r in predicted_confirmed)
        resultados_relacion["fn"] += sum(r not in predicted_confirmed for r in expected_confirmed)
        if expected_ambiguous and any(r[0] == expected_ambiguous[0][0] and r[3] == "CONFIRMADA" for r in relaciones_obtenidas):
            resultados_relacion["ambiguas_erroneas"] += 1
    print("Resultados por categoría:")
    for categoria, (aciertos, total) in conteos.items():
        print(f"- {categoria}: {aciertos}/{total} ({100 * aciertos / total:.1f}%) | fallos: {total - aciertos}")
    tp, fp, fn = (resultados_relacion[k] for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    exhaustividad = tp / (tp + fn) if tp + fn else 0.0
    print(f"Relaciones: TP={tp}, FP={fp}, FN={fn}, precisión={precision:.3f}, exhaustividad={exhaustividad:.3f}, ambigüedades resueltas mal={resultados_relacion['ambiguas_erroneas']}")
    print(f"Errores de asociación de importe/cantidad: {errores_importe}")
    for caso, operaciones, relaciones, salida in errores:
        print(f"[FALLO {caso['categoria']}] {caso['texto']!r}")
        print(f"        operaciones esperadas: {caso['operaciones']}")
        print(f"        operaciones obtenidas: {operaciones}")
        print(f"        relaciones esperadas: {caso['relaciones']} | obtenidas: {relaciones}")
    return conteos, resultados_relacion, errores


# Casos fallidos en la primera medición; permanecen aislados como regresiones.
PRUEBAS_REGRESION_FASE_2 = [
    caso for caso in EVALUACION_ADVERSARIAL
    if caso["texto"] in {
        "Hoy salieron 18 dólares en transporte",
        "Me entraron 45 dólares por la venta de frutas",
        "Le entregué 20 dólares al proveedor",
        "Por las verduras recibí 32 dólares",
        "Vendí clavos por 9 dólares; además gasté 2 dólares en embalaje",
        "Adquirí jabón por 12 dólares y luego aboné 4 dólares de aquella compra",
        "Compré tela por 20 dólares y pagué 7 dólares de la compra de tela",
        "Vendí 5 cajas de té por un total de 40 dólares",
        "Vendí cuatro cuadernos, pero todavía no fijé el importe",
        "compre 2 vasos por 5 dolares",
        "Aboné 5 dólares de esa",
    }
]


def evaluar_regresiones_fase_2(nlp):
    print("\nREGRESIONES DE LOS FALLOS ADVERSARIALES INICIALES")
    return evaluar_conjunto_independiente(nlp, PRUEBAS_REGRESION_FASE_2)


def main():
    import spacy

    nlp = spacy.load("custom_ner_model")
    evaluar_ner(nlp, PRUEBAS_NUEVAS, "PRUEBAS NER", detallado=True)
    evaluar_entidades(nlp)
    evaluar_semantica(nlp)
    evaluar_operaciones_multiples(nlp)
    evaluar_relaciones(nlp)
    evaluar_contrato(nlp)
    evaluar_conjunto_independiente(nlp)
    evaluar_regresiones_fase_2(nlp)


if __name__ == "__main__":
    main()
