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
        correcto = all(obtenido.get(campo) == valor for campo, valor in esperado.items())
        if correcto:
            aciertos += 1
            print(f"[OK] {texto!r}")
        else:
            print(f"[FALLO] {texto!r}")
            print(f"        esperados: {esperado}")
            print(f"        obtenido:  {obtenido}")
    print(f"\nResultado semántico: {aciertos}/{len(pruebas)} frases correctas")
    return aciertos


def main():
    import spacy

    nlp = spacy.load("custom_ner_model")
    evaluar_ner(nlp, PRUEBAS_NUEVAS, "PRUEBAS NER", detallado=True)
    evaluar_semantica(nlp)


if __name__ == "__main__":
    main()
