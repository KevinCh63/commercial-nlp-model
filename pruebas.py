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


# Conjunto externo a train_data: ejemplos nuevos anotados para medir
# generalización real. No se usa para entrenar ni se comparte con los fixtures
# históricos; las frases se auditan contra el entrenamiento antes de entrenar.
PRUEBAS_NER_INDEPENDIENTE = [
    _crear_prueba("Por 37 dólares recibieron ocho sacos de quinua", "COMPUESTA", [("37 dólares", "PRICE"), ("ocho sacos", "QUANTITY"), ("quinua", "PRODUCT")]),
    _crear_prueba("El municipio desembolsó 642 dólares para el mantenimiento vial", "CONCEPTOS", [("642 dólares", "PRICE"), ("mantenimiento vial", "CONCEPT")]),
    _crear_prueba("Ayer vendí, sin intermediarios, once bidones de aceite vegetal por 48 euros", "VARIACION", [("once bidones", "QUANTITY"), ("aceite vegetal", "PRODUCT"), ("48 euros", "PRICE")]),
    _crear_prueba("¿Cuánto pagó Rosa por tres metros de cable? 26 dólares", "ORDEN", [("tres metros", "QUANTITY"), ("cable", "PRODUCT"), ("26 dólares", "PRICE")]),
    _crear_prueba("Recibimos $58 por las canastas artesanales", "MONEDA", [("$58", "PRICE"), ("canastas artesanales", "PRODUCT")]),
    _crear_prueba("Con 72 euros compraron las láminas galvanizadas", "ORDEN", [("72 euros", "PRICE"), ("láminas galvanizadas", "PRODUCT")]),
    _crear_prueba("Nueve kilos de cacao se colocaron a 33 dólares", "PASIVA", [("Nueve kilos", "QUANTITY"), ("cacao", "PRODUCT"), ("33 dólares", "PRICE")]),
    _crear_prueba("Para reparar el techo, gastamos 215 dólares", "CONCEPTOS", [("reparar el techo", "CONCEPT"), ("215 dólares", "PRICE")]),
    _crear_prueba("El proveedor facturó 19,75 dólares por dos frascos de esencia", "DECIMALES", [("19,75 dólares", "PRICE"), ("dos frascos", "QUANTITY"), ("esencia", "PRODUCT")]),
    _crear_prueba("Se pagaron 84 pesos por el transporte escolar", "MONEDA", [("84 pesos", "PRICE"), ("transporte escolar", "CONCEPT")]),
    _crear_prueba("La cooperativa vendió mallas y redes por 1.250 dólares", "CANTIDAD", [("mallas", "PRODUCT"), ("redes", "PRODUCT"), ("1.250 dólares", "PRICE")]),
    _crear_prueba("Entregué tres docenas de empanadas al puesto por 60 dólares", "CANTIDAD", [("tres docenas", "QUANTITY"), ("empanadas", "PRODUCT"), ("60 dólares", "PRICE")]),
    _crear_prueba("Con €22 compró 1.5 kg de nuez pecana", "FORMATOS", [("€22", "PRICE"), ("1.5 kg", "QUANTITY"), ("nuez pecana", "PRODUCT")]),
    _crear_prueba("Factura de repuestos: 39 USD, cancelados hoy", "CONCEPTOS", [("repuestos", "CONCEPT"), ("39 USD", "PRICE")]),
    _crear_prueba("Vendí tres cajas de azulejos por un total de 125 dólares", "ALCANCE", [("tres cajas", "QUANTITY"), ("azulejos", "PRODUCT"), ("125 dólares", "PRICE")]),
]


# Conjunto de desarrollo: puede guiar iteraciones de entrenamiento, pero no se
# incorpora a train_data ni se confunde con el conjunto independiente histórico.
PRUEBAS_NER_VALIDACION = [
    _crear_prueba("Con 46 euros adquirieron cinco rollos de papel reciclado", "DESARROLLO", [("46 euros", "PRICE"), ("cinco rollos", "QUANTITY"), ("papel reciclado", "PRODUCT")]),
    _crear_prueba("A la caja ingresaron US$ 89 por la venta de panela", "DESARROLLO", [("US$ 89", "PRICE"), ("panela", "PRODUCT")]),
    _crear_prueba("Se invirtieron 315 dólares en la refacción del local", "DESARROLLO", [("315 dólares", "PRICE"), ("refacción del local", "CONCEPT")]),
    _crear_prueba("Tres bidones de cloro costaron 17,60 euros", "DESARROLLO", [("Tres bidones", "QUANTITY"), ("cloro", "PRODUCT"), ("17,60 euros", "PRICE")]),
    _crear_prueba("El comercio recibió 120 pesos por quince sacos de arroz", "DESARROLLO", [("120 pesos", "PRICE"), ("quince sacos", "QUANTITY"), ("arroz", "PRODUCT")]),
    _crear_prueba("Pagué 8 dólares por la compostura de la bomba hidráulica", "DESARROLLO", [("8 dólares", "PRICE"), ("compostura de la bomba hidráulica", "CONCEPT")]),
    _crear_prueba("Doce baldes de pintura blanca salieron a 76 dólares", "DESARROLLO", [("Doce baldes", "QUANTITY"), ("pintura blanca", "PRODUCT"), ("76 dólares", "PRICE")]),
    _crear_prueba("En materiales, el taller gastó 58 dólares", "DESARROLLO", [("materiales", "CONCEPT"), ("58 dólares", "PRICE")]),
    _crear_prueba("La harina integral se vendió a 22 USD la funda", "DESARROLLO", [("harina integral", "PRODUCT"), ("22 USD", "PRICE")]),
    _crear_prueba("Compramos 2,5 kg de nuez moscada por 31 euros", "DESARROLLO", [("2,5 kg", "QUANTITY"), ("nuez moscada", "PRODUCT"), ("31 euros", "PRICE")]),
    _crear_prueba("Veinticuatro cajas de focos LED se entregaron por $94", "DESARROLLO", [("Veinticuatro cajas", "QUANTITY"), ("focos LED", "PRODUCT"), ("$94", "PRICE")]),
    _crear_prueba("Pagaron 13 euros para instalar la cerradura nueva", "DESARROLLO", [("13 euros", "PRICE"), ("instalar la cerradura nueva", "CONCEPT")]),
]


# Evaluación final reservada. No se importa desde el entrenamiento ni se usa
# para elegir ejemplos, épocas o parámetros; se consulta una vez al cerrar la
# candidata para medir generalización adicional.
PRUEBAS_NER_FINAL = [
    _crear_prueba("De la cosecha salieron 26 arrobas de frijol rojo, valoradas en 318 euros", "FINAL", [("26 arrobas", "QUANTITY"), ("frijol rojo", "PRODUCT"), ("318 euros", "PRICE")]),
    _crear_prueba("El taller cargó a caja 67,80 USD por el servicio de soldadura", "FINAL", [("67,80 USD", "PRICE"), ("servicio de soldadura", "CONCEPT")]),
    _crear_prueba("Recibieron 1.240 pesos por vender barro cocido", "FINAL", [("1.240 pesos", "PRICE"), ("barro cocido", "PRODUCT")]),
    _crear_prueba("Cinco rollos de lámina galvanizada fueron entregados por 112 dólares", "FINAL", [("Cinco rollos", "QUANTITY"), ("lámina galvanizada", "PRODUCT"), ("112 dólares", "PRICE")]),
    _crear_prueba("Pagó 27 euros en mantenimiento preventivo de las cámaras", "FINAL", [("27 euros", "PRICE"), ("mantenimiento preventivo de las cámaras", "CONCEPT")]),
    _crear_prueba("Bajaron del camión 18 sacos de quinua orgánica; el total fue $405", "FINAL", [("18 sacos", "QUANTITY"), ("quinua orgánica", "PRODUCT"), ("$405", "PRICE")]),
    _crear_prueba("Compró cuarenta metros de cable coaxial a 83 euros", "FINAL", [("cuarenta metros", "QUANTITY"), ("cable coaxial", "PRODUCT"), ("83 euros", "PRICE")]),
    _crear_prueba("El recaudo fue 51 dólares y correspondió a la venta de cerámicas pintadas", "FINAL", [("51 dólares", "PRICE"), ("cerámicas pintadas", "PRODUCT")]),
    _crear_prueba("Se reservaron 625 euros para la reparación integral del techo", "FINAL", [("625 euros", "PRICE"), ("reparación integral del techo", "CONCEPT")]),
    _crear_prueba("Doce kilogramos de ají seco costaron 72,30 dólares", "FINAL", [("Doce kilogramos", "QUANTITY"), ("ají seco", "PRODUCT"), ("72,30 dólares", "PRICE")]),
    _crear_prueba("Hubo un gasto de USD 96 en limpieza del ducto", "FINAL", [("USD 96", "PRICE"), ("limpieza del ducto", "CONCEPT")]),
    _crear_prueba("La farmacia entregó tres cajas de gasas estériles a cambio de 29 euros", "FINAL", [("tres cajas", "QUANTITY"), ("gasas estériles", "PRODUCT"), ("29 euros", "PRICE")]),
    _crear_prueba("Ingresaron 2.300 dólares por quince lotes de madera de pino", "FINAL", [("2.300 dólares", "PRICE"), ("quince lotes", "QUANTITY"), ("madera de pino", "PRODUCT")]),
    _crear_prueba("Del monto, €14,50 correspondieron al envío urgente", "FINAL", [("€14,50", "PRICE"), ("envío urgente", "CONCEPT")]),
    _crear_prueba("En la subasta se colocaron siete máscaras de barro por 214 dólares", "FINAL", [("siete", "QUANTITY"), ("máscaras de barro", "PRODUCT"), ("214 dólares", "PRICE")]),
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


def medir_ner(nlp, pruebas):
    """Métricas de entidad con coincidencia exacta de límites y etiqueta."""
    from collections import Counter

    etiquetas = ("QUANTITY", "PRODUCT", "PRICE", "CONCEPT")
    conteos = {etiqueta: {"tp": 0, "fp": 0, "fn": 0} for etiqueta in etiquetas}
    casos, errores_por_tipo = [], Counter()
    for texto, anotaciones in pruebas:
        esperadas = {(inicio, fin, etiqueta) for inicio, fin, etiqueta in anotaciones["entities"]}
        predichas = {(ent.start_char, ent.end_char, ent.label_) for ent in nlp(texto).ents}
        for etiqueta in etiquetas:
            reales = {ent for ent in esperadas if ent[2] == etiqueta}
            predichas_etiqueta = {ent for ent in predichas if ent[2] == etiqueta}
            conteos[etiqueta]["tp"] += len(reales & predichas_etiqueta)
            conteos[etiqueta]["fp"] += len(predichas_etiqueta - reales)
            conteos[etiqueta]["fn"] += len(reales - predichas_etiqueta)
        emparejadas = esperadas & predichas
        sin_pred = set(predichas - emparejadas)
        errores = []
        for real in sorted(esperadas - emparejadas):
            inicio, fin, etiqueta = real
            solapadas = [p for p in sin_pred if max(inicio, p[0]) < min(fin, p[1])]
            if not solapadas:
                error = {"esperado": texto[inicio:fin], "etiqueta_esperada": etiqueta, "predicho": None, "etiqueta_predicha": None, "tipo": "entidad_omitida"}
            else:
                def iou(pred):
                    interseccion = max(0, min(fin, pred[1]) - max(inicio, pred[0]))
                    union = max(fin, pred[1]) - min(inicio, pred[0])
                    return interseccion / union if union else 0
                pred = max(solapadas, key=iou)
                sin_pred.remove(pred)
                mismos_limites = (inicio, fin) == pred[:2]
                if mismos_limites and etiqueta != pred[2]:
                    tipo = "etiqueta_incorrecta"
                elif etiqueta == pred[2]:
                    tipo = "limites_incorrectos_o_entidad_parcial"
                else:
                    tipo = "confusion_de_etiqueta_y_limites"
                error = {"esperado": texto[inicio:fin], "etiqueta_esperada": etiqueta, "predicho": texto[pred[0]:pred[1]], "etiqueta_predicha": pred[2], "tipo": tipo}
            errores.append(error)
            errores_por_tipo[error["tipo"]] += 1
        for inicio, fin, etiqueta in sorted(sin_pred):
            error = {"esperado": None, "etiqueta_esperada": None, "predicho": texto[inicio:fin], "etiqueta_predicha": etiqueta, "tipo": "falsa_entidad"}
            errores.append(error)
            errores_por_tipo[error["tipo"]] += 1
        casos.append({
            "texto": texto,
            "coincidencia_exacta_de_frase": esperadas == predichas,
            "esperadas": [{"fragmento": texto[s:e], "etiqueta": l, "inicio": s, "fin": e} for s, e, l in sorted(esperadas)],
            "predichas": [{"fragmento": texto[s:e], "etiqueta": l, "inicio": s, "fin": e} for s, e, l in sorted(predichas)],
            "errores": errores,
        })
    for stats in conteos.values():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        stats["precision"] = tp / (tp + fp) if tp + fp else 0.0
        stats["recall"] = tp / (tp + fn) if tp + fn else 0.0
        stats["f1"] = 2 * stats["precision"] * stats["recall"] / (stats["precision"] + stats["recall"]) if stats["precision"] + stats["recall"] else 0.0
    micro = {clave: sum(stats[clave] for stats in conteos.values()) for clave in ("tp", "fp", "fn")}
    micro["precision"] = micro["tp"] / (micro["tp"] + micro["fp"]) if micro["tp"] + micro["fp"] else 0.0
    micro["recall"] = micro["tp"] / (micro["tp"] + micro["fn"]) if micro["tp"] + micro["fn"] else 0.0
    micro["f1"] = 2 * micro["precision"] * micro["recall"] / (micro["precision"] + micro["recall"]) if micro["precision"] + micro["recall"] else 0.0
    macro_f1 = sum(stats["f1"] for stats in conteos.values()) / len(etiquetas)
    return {
        "unidad": "entidad con límites y etiqueta exactos; no es métrica por token",
        "frases": len(casos),
        "coincidencia_exacta_de_frase": {"correctas": sum(c["coincidencia_exacta_de_frase"] for c in casos), "total": len(casos)},
        "micro_entidad": micro,
        "macro_f1_por_etiqueta": macro_f1,
        "por_etiqueta": conteos,
        "tipos_de_error": dict(errores_por_tipo),
        "casos": casos,
    }


def evaluar_validacion_datos(nlp):
    """Prueba el validador de anotaciones y la detección de fuga textual."""
    from entrenamiento import auditar_solapamiento, train_data, validar_train_data

    casos_invalidos = [
        ("etiqueta", [("Caja", {"entities": [(0, 4, "MONEY")]})], "etiqueta"),
        ("offset fuera de rango", [("Caja", {"entities": [(0, 9, "PRODUCT")]})], "fuera de rango"),
        ("entidad vacía", [("Caja", {"entities": [(2, 2, "PRODUCT")]})], "fuera de rango"),
        ("fragmento vacío", [("a b", {"entities": [(1, 2, "PRODUCT")]})], "produce una entidad vacía"),
        ("solapamiento", [("4 cajas", {"entities": [(0, 1, "QUANTITY"), (0, 6, "PRODUCT")]})], "solapan"),
        ("duplicado exacto", [("Caja", {"entities": []}), ("Caja", {"entities": []})], "duplicado"),
        ("alineación de token", [("caja", {"entities": [(1, 4, "PRODUCT")]})], "no está alineado"),
    ]
    aciertos = 0
    print("\nVALIDACIÓN DE ANOTACIONES Y FUGA DE DATOS")
    print("-" * 60)
    for nombre, datos, fragmento_esperado in casos_invalidos:
        try:
            validar_train_data(nlp, datos)
            correcto = False
            detalle = "no rechazó los datos inválidos"
        except ValueError as exc:
            correcto = fragmento_esperado in str(exc)
            detalle = str(exc).splitlines()[-1]
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] {nombre}: {detalle}")

    try:
        validar_train_data(nlp, train_data)
        exactos_hist, cercanos_hist = auditar_solapamiento(train_data, PRUEBAS_NUEVAS)
        exactos_ind, cercanos_ind = auditar_solapamiento(train_data, PRUEBAS_NER_INDEPENDIENTE)
        correcto = bool(exactos_hist) and bool(cercanos_hist) and not exactos_ind and not cercanos_ind
    except Exception as exc:
        correcto = False
        exactos_hist, cercanos_hist, exactos_ind, cercanos_ind = [], [], [], []
        print(f"[FALLO] auditoría de conjuntos: {type(exc).__name__}: {exc}")
    aciertos += int(correcto)
    print(f"[{'OK' if correcto else 'FALLO'}] separación: legado exactos={len(exactos_hist)}, cercanos={len(cercanos_hist)} | independiente exactos={len(exactos_ind)}, cercanos={len(cercanos_ind)}")
    total = len(casos_invalidos) + 1
    print(f"Resultado validación de datos: {aciertos}/{total}")
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
    {"categoria": "entradas_dificiles", "texto": "Vendi cinco cajas por dies dólares", "operaciones": [_operacion_esperada(None, "vender", "INGRESO", "cinco cajas", None, None, None, "NO_ESPECIFICADO")], "relaciones": []},
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


def evaluar_regresiones_fase_4(nlp):
    from interpretacion import interpretar_texto, interpretar_operaciones, interpretar_operaciones_con_relaciones

    print("\nREGRESIONES Y CONSISTENCIA DEL CONTRATO — FASE 4")
    print("-" * 60)
    casos = [
        "Compré 3 y vendí 4 por 20 dólares",
        "Compré 3 manzanas por 10 dólares y vendí 4 por 20 dólares",
        "Compré 3 manzanas por 10 dólares y vendí 4 naranjas por 20 dólares",
    ]
    aciertos = 0
    for texto in casos:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        ops = resultado["operaciones"]
        correcto = len(ops) == 2 and ops[1]["producto"] != "por"
        correcto = correcto and [op["cantidad"] for op in ops] == ["3", "4"]
        if texto == casos[0]:
            correcto = correcto and ops[0]["producto"] is None and ops[1]["producto"] is None
        elif texto == casos[1]:
            correcto = correcto and ops[0]["producto"] == "manzanas" and ops[1]["producto"] is None
        else:
            correcto = correcto and ops[0]["producto"] == "manzanas" and ops[1]["producto"] == "naranjas"
        correcto = correcto and all(
            resultado["texto"][meta["inicio"]:meta["fin"]] == meta["texto"]
            and meta["texto"].casefold() in op["texto"].casefold()
            for op in ops for campo in op["campos"].values()
            if (meta := campo.get("evidencia")) and meta.get("inicio") is not None and meta.get("fin") is not None
        )
        if correcto:
            aciertos += 1
            print(f"[OK] asociación y offsets: {texto!r}")
        else:
            print(f"[FALLO] asociación y offsets: {texto!r} | operaciones={ops}")

    relaciones = [
        ("Compré papel por 100 dólares y aboné 10 dólares de esa compra", 1, "CONFIRMADA", 0),
        ("Compré papel por 100 dólares y compré fundas por 50 dólares y aboné 10 dólares", 2, None, None),
        ("Compré papel por 100 dólares y compré fundas por 50 dólares y aboné 10 dólares de esa compra", 2, "AMBIGUA", None),
        ("Compré papel por 100 dólares y aboné 10 dólares de esa compra y aboné 20 dólares de esa compra", 2, "CONFIRMADA", 0),
        ("Compré papel por 100 dólares y aboné 10 dólares de la compra anterior", 1, "CONFIRMADA", 0),
        ("Compré papel por 100 dólares y compré fundas por 50 dólares y aboné 10 dólares de la compra anterior", 2, "CONFIRMADA", 1),
    ]
    for texto, compras, estado_relacion, destino in relaciones:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        if estado_relacion is None:
            correcto = resultado["relaciones"] == []
        else:
            correctas = [r for r in resultado["relaciones"] if r["estado"] == estado_relacion]
            correcto = bool(correctas) and all(r["operacion_destino"] == destino for r in correctas)
        correcto = correcto and len({op["id"] for op in resultado["operaciones"]}) == len(resultado["operaciones"])
        correcto = correcto and all(
            resultado["operaciones"][r["operacion_origen"]]["id"] == r["operacion_origen_id"]
            and (
                r["operacion_destino"] is None
                or resultado["operaciones"][r["operacion_destino"]]["id"] == r["operacion_destino_id"]
            )
            for r in resultado["relaciones"]
        )
        if correcto:
            aciertos += 1
            print(f"[OK] relación: {texto!r}")
        else:
            print(f"[FALLO] relación: {texto!r} | {resultado['relaciones']}")

    genericos = {
        "Compré producto por 10 dólares": "GENERICA",
        "Vendí mercadería por 20 dólares": "GENERICA",
        "Compré artículos": "GENERICA",
    }
    for texto, tipo_esperado in genericos.items():
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        op = resultado["operaciones"][0] if resultado["operaciones"] else {}
        correcto = op.get("campos", {}).get("producto", {}).get("interpretacion") == tipo_esperado
        if correcto:
            aciertos += 1
            print(f"[OK] producto genérico preservado: {texto!r}")
        else:
            print(f"[FALLO] producto genérico: {texto!r} | {op.get('campos', {}).get('producto')}")

    texto_unidades = "Compré 3 cajas"
    salida_unidades = interpretar_operaciones_con_relaciones(texto_unidades, nlp=nlp)
    correcto = bool(salida_unidades["operaciones"]) and salida_unidades["operaciones"][0]["producto"] is None
    aciertos += int(correcto)
    print(f"[{'OK' if correcto else 'FALLO'}] unidad sin producto: {texto_unidades!r}")

    for texto, esperado in [("", "error"), ("   ", "error"), ("xyz blorf", "NO_RECONOCIDO"), ("Vendí", "PARCIAL")]:
        try:
            if esperado == "error":
                interpretar_texto(texto, nlp=nlp)
                correcto = False
            else:
                correcto = interpretar_operaciones_con_relaciones(texto, nlp=nlp)["estado"] == esperado
        except ValueError:
            correcto = esperado == "error"
        if esperado == "error":
            try:
                interpretar_operaciones(texto, nlp=nlp)
                correcto = False
            except ValueError:
                pass
            wrapper = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
            correcto = correcto and wrapper["estado"] == "NO_RECONOCIDO" and wrapper["operaciones"] == []
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] entrada {texto!r} -> {esperado}")
    print(f"\nResultado fase 4: {aciertos}/17 casos correctos")
    return aciertos


def evaluar_endurecimiento_semantico(nlp):
    from interpretacion import interpretar_operaciones_con_relaciones, interpretar_texto

    print("\nENDURECIMIENTO SEMÁNTICO Y LINGÜÍSTICO")
    print("-" * 60)
    casos = [
        ("vendi 3 libras de tomate por 5 dolares", "vender", "USD", "3 libras", "tomate", "5 dolares"),
        ("VENDI 3 LIBRAS DE TOMATE POR 5 DOLARES", "vender", "USD", "3 LIBRAS", "TOMATE", "5 DOLARES"),
        ("Vendí 3 libras de tomate por 5 euros", "vender", "EUR", "3 libras", "tomate", "5 euros"),
        ("Vendí 3 libras de tomate por 5 pesos", "vender", "PESO", "3 libras", "tomate", "5 pesos"),
        ("pague 5 dolares", "pagar", "USD", None, None, "5 dolares"),
        ("gaste 5 euros", "gastar", "EUR", None, None, "5 euros"),
        ("abone 5 pesos", "abonar", "PESO", None, None, "5 pesos"),
        ("compre 3 cajas de tomate por 10 dolares", "comprar", "USD", "3 cajas", "tomate", "10 dolares"),
        ("pago 5 dolares", "pagar", "USD", None, None, "5 dolares"),
        ("Compré algo", "comprar", None, None, "algo", None),
        ("Compré algunas cosas", "comprar", None, "algunas", "cosas", None),
        ("Vendí varios productos", "vender", None, "varios", "productos", None),
        ("Compré 3 cajas de tomate", "comprar", None, "3 cajas", "tomate", None),
        ("Vendí 5 camisetas por 30 dólares", "vender", "USD", "5", "camisetas", "30 dólares"),
    ]
    aciertos = 0
    for texto, accion, moneda, cantidad, producto, precio in casos:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        op = resultado["operaciones"][0] if resultado["operaciones"] else {}
        campos_correctos = (op.get("accion"), op.get("moneda"), op.get("cantidad"), op.get("producto"), op.get("precio")) == (accion, moneda, cantidad, producto, precio)
        if producto in {"algo", "algunas cosas", "productos"}:
            campos_correctos = campos_correctos and op.get("campos", {}).get("producto", {}).get("interpretacion") == "GENERICA"
            estado_esperado = "AMBIGUO" if cantidad in {"varios", "varias", "algunos", "algunas", "muchos", "muchas"} else "PARCIAL"
            campos_correctos = campos_correctos and op.get("estado") == estado_esperado
        correcto = len(resultado["operaciones"]) == 1 and campos_correctos
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] {texto!r} | real={((op.get('accion'), op.get('moneda'), op.get('cantidad'), op.get('producto'), op.get('precio'), op.get('estado')))}")

    modalidades = [
        ("No compré nada", "NEGADA", "comprar"),
        ("No vendí nada", "NEGADA", "vender"),
        ("No pagué", "NEGADA", "pagar"),
        ("No quiero comprar", "INTENCION", "comprar"),
        ("Quiero comprar tomate mañana", "INTENCION", "comprar"),
        ("Tal vez compré tomate", "INCIERTA", "comprar"),
        ("Tal vez compré mercadería por 10 dólares", "INCIERTA", "comprar"),
        ("No compré tomate", "NEGADA", "comprar"),
        ("Sí compré tomate", "AFIRMADA", "comprar"),
    ]
    for texto, modalidad, accion in modalidades:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        op = resultado["operaciones"][0] if resultado["operaciones"] else {}
        analisis = interpretar_texto(texto, nlp=nlp)
        correcto = analisis.get("modalidad") == modalidad and analisis.get("accion") == accion
        correcto = correcto and (len(resultado["operaciones"]) == (1 if modalidad == "AFIRMADA" else 0))
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] admisión modalidad {texto!r} | análisis={analisis.get('modalidad')}, operaciones={len(resultado['operaciones'])}")

    relacion_negada = interpretar_operaciones_con_relaciones(
        "No compré papel por 10 dólares y aboné 2 dólares de esa compra", nlp=nlp
    )
    relaciones_negadas = relacion_negada.get("relaciones", [])
    correcto = (
        len(relaciones_negadas) == 1
        and relaciones_negadas[0].get("estado") == "AMBIGUA"
        and relaciones_negadas[0].get("operacion_destino_id") is None
        and len(relacion_negada["operaciones"]) == 1
        and relacion_negada["operaciones"][0].get("accion") == "abonar"
    )
    aciertos += int(correcto)
    print(f"[{'OK' if correcto else 'FALLO'}] relación a compra negada | real={relaciones_negadas}")

    estados = ["Vendí 2 linternas por 12 dólares", "Por 20 dólares vendí 2 lámparas"]
    for texto in estados:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        op = resultado["operaciones"][0]
        ambiguo = any(meta.get("estado") == "AMBIGUO" for meta in op["campos"].values())
        correcto = not (ambiguo and op["estado"] == "VALIDO")
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] coherencia {texto!r} | op={op['estado']}, campos={[m['estado'] for m in op['campos'].values()]}")
    finanzas = [
        ("Vendí 5 dólares por caja", "UNITARIO", "USD"),
        ("Vendí 5 dólares en total", "TOTAL", "USD"),
        ("Vendí 5", "NO_ESPECIFICADO", None),
        ("Vendí diez dólares", "NO_ESPECIFICADO", "USD"),
        ("Vendí cinco dólares", "NO_ESPECIFICADO", "USD"),
        ("Vendí 5 dolares", "NO_ESPECIFICADO", "USD"),
        ("Vendí 5 DÓLARES", "NO_ESPECIFICADO", "USD"),
        ("Vendí 5 euros", "NO_ESPECIFICADO", "EUR"),
        ("Vendí 5 pesos", "NO_ESPECIFICADO", "PESO"),
        ("Vendi cinco cajas por dies dólares", "NO_ESPECIFICADO", None),
        ("Vendi cinco cajas por un importe desconocido", "NO_ESPECIFICADO", None),
    ]
    for texto, ambito, moneda in finanzas:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        op = resultado["operaciones"][0] if resultado["operaciones"] else {}
        precio_meta = op.get("campos", {}).get("precio", {})
        correcto = op.get("moneda") == moneda and precio_meta.get("ambito") == ambito
        if "dies" in texto or "importe desconocido" in texto:
            correcto = correcto and op.get("precio") is None and "precio" in op.get("faltantes", [])
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] importe {texto!r} | real={op.get('precio')}, {op.get('moneda')}, {precio_meta.get('ambito')}")
    total = len(casos) + len(modalidades) + 1 + len(estados) + len(finanzas)
    print(f"\nResultado endurecimiento: {aciertos}/{total}")
    return aciertos


def evaluar_admision_operaciones(nlp):
    from interpretacion import interpretar_operaciones_con_relaciones, interpretar_texto

    print("\nADMISIÓN DE OPERACIONES AFIRMADAS")
    print("-" * 60)
    aceptadas = [
        ("Compré 3 cajas de tomate por 10 dólares", ["comprar"]),
        ("Vendí tomate por 5 dólares", ["vender"]),
        ("María compró 4 productos", ["comprar"]),
        ("Aboné 10 dólares de esa compra", ["abonar"]),
        ("Compré tomate y después vendí cebollas", ["comprar", "vender"]),
        ("Tal vez compré tomate, pero ayer vendí cebollas", ["vender"]),
        ("No compré tomate y vendí cebollas", ["vender"]),
    ]
    rechazadas = [
        "Tal vez compré tomate", "Quizá vendí cajas", "Creo que compré tomate",
        "No compré tomate", "Nunca vendí cebollas", "No quiero comprar tomate",
        "Quiero comprar tomate mañana", "Podría comprar tomate",
        "Si compro tomate, pagaré después", "Compraré tomate mañana",
    ]
    aciertos = 0
    for texto, acciones_esperadas in aceptadas:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        acciones_reales = [op["accion"] for op in resultado["operaciones"]]
        correcto = acciones_reales == acciones_esperadas
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] afirmadas {texto!r} | esperado={acciones_esperadas}, real={acciones_reales}")
    for texto in rechazadas:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        correcto = resultado["operaciones"] == []
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] no ejecutada {texto!r} | operaciones={len(resultado['operaciones'])}")

    incompletas = [
        ("Compré tomate", "PARCIAL", None, "tomate", None),
        ("Compré cinco cajas por dies dólares", "PARCIAL", "cinco cajas", None, None),
        ("Vendi cinco cajas por diez dólares", "AMBIGUO", "cinco cajas", None, "diez dólares"),
        ("Vendí varios productos", "AMBIGUO", "varios", "productos", None),
        ("Compré varias cosas", "AMBIGUO", "varias", "cosas", None),
    ]
    for texto, estado, cantidad, producto, precio in incompletas:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        op = resultado["operaciones"][0] if len(resultado["operaciones"]) == 1 else {}
        correcto = (op.get("estado"), op.get("cantidad"), op.get("producto"), op.get("precio")) == (estado, cantidad, producto, precio)
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] campos explícitos {texto!r} | real={(op.get('estado'), op.get('cantidad'), op.get('producto'), op.get('precio'))}")

    texto = "Compraron tomate"
    resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
    op = resultado["operaciones"][0] if resultado["operaciones"] else {}
    correcto = op.get("cantidad") is None and op.get("producto") == "tomate" and op.get("agente") == "TERCERA_PERSONA_PLURAL"
    aciertos += int(correcto)
    print(f"[{'OK' if correcto else 'FALLO'}] cantidad/producto/agente {texto!r} | real={(op.get('cantidad'), op.get('producto'), op.get('agente'))}")

    for texto, agente in [("Compramos tomate", "NOSOTROS"), ("María compró tomate", "María")]:
        resultado = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
        op = resultado["operaciones"][0] if resultado["operaciones"] else {}
        correcto = op.get("agente") == agente
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] atribución explícita {texto!r} | agente={op.get('agente')}")

    lectura = interpretar_texto("Quizá vendí cajas", nlp=nlp)
    correcto = lectura.get("producto") is None
    aciertos += int(correcto)
    print(f"[{'OK' if correcto else 'FALLO'}] no atribuir verbo como producto | producto={lectura.get('producto')}")

    total = len(aceptadas) + len(rechazadas) + len(incompletas) + 4
    print(f"\nResultado admisión: {aciertos}/{total}")
    return aciertos


def evaluar_regresiones_fase_2(nlp):
    print("\nREGRESIONES DE LOS FALLOS ADVERSARIALES INICIALES")
    return evaluar_conjunto_independiente(nlp, PRUEBAS_REGRESION_FASE_2)


def evaluar_matriz_modalidad(nlp):
    """Valida cláusula por cláusula qué operaciones afirmadas admite el contrato."""
    from interpretacion import interpretar_operaciones_con_relaciones

    casos = [
        ("No compré tomate, pero vendí cebollas", ["vender"], [("vender", "cebollas")]),
        ("Tal vez compré tomate, pero ayer vendí cebollas", ["vender"], [("vender", "cebollas")]),
        ("Creo que compré tomate y después pagué 5 dólares", ["pagar"], [("pagar", None)]),
        ("No compré tomate y tampoco pagué", [], []),
        ("No compré tomate, pero sí aboné 5 dólares de una deuda anterior", ["abonar"], [("abonar", None)]),
        ("Compré tomate y no vendí cebollas", ["comprar"], [("comprar", "tomate")]),
        ("Si compro tomate, pagaré mañana", [], []),
        ("Quiero comprar tomate, pero ayer compré arroz", ["comprar"], [("comprar", "arroz")]),
        ("Mañana compraré tomate y hoy vendí arroz", ["vender"], [("vender", "arroz")]),
        ("No quiero comprar tomate; pagué una deuda ayer", ["pagar"], [("pagar", None)]),
    ]
    errores = 0
    print("\nMATRIZ DE DIEZ FRASES: MODALIDAD Y ADMISIÓN")
    print("-" * 60)
    for texto, acciones_esperadas, valores_esperados in casos:
        try:
            salida = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
            operaciones = salida["operaciones"]
            acciones = [op.get("accion") for op in operaciones]
            campos = [(op.get("accion"), op.get("producto")) for op in operaciones]
            correcto = acciones == acciones_esperadas and campos == valores_esperados
            # Los campos incompletos deben seguir ausentes y los identificadores no deben repetirse.
            correcto = correcto and len({op.get("id") for op in operaciones}) == len(operaciones)
            if texto == casos[2][0]:
                correcto = correcto and operaciones[0].get("precio") == "5 dólares" and operaciones[0].get("producto") is None
            if texto == casos[4][0]:
                correcto = correcto and operaciones[0].get("precio") == "5 dólares"
            if texto == casos[9][0]:
                correcto = correcto and operaciones[0].get("producto") is None
        except Exception as exc:
            correcto, operaciones = False, []
            print(f"[FALLO] {texto!r} | excepción={type(exc).__name__}: {exc}")
        errores += int(not correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] {texto!r} | acciones esperadas={acciones_esperadas}, obtenidas={[op.get('accion') for op in operaciones]} | ops={[(op.get('accion'), op.get('agente'), op.get('producto'), op.get('cantidad'), op.get('precio')) for op in operaciones]}")
    print(f"Resultado matriz modalidad: {len(casos) - errores}/{len(casos)} | fallos={errores}")
    return errores


def evaluar_relaciones_evidencia(nlp):
    """Comprueba integridad de relaciones e índices de evidencia sobre el original."""
    from interpretacion import interpretar_operaciones_con_relaciones

    casos = [
        ("Compré papel y aboné 2 dólares de esa compra", "confirmada"),
        ("No compré papel y aboné 2 dólares de esa compra", "no_destino_no_admitido"),
        ("Tal vez compré papel y aboné 2 dólares de esa compra", "no_destino_no_admitido"),
        ("Compré café y compré té; aboné 2 dólares de esa compra", "ambigua"),
        ("Compré papel y luego pagué 2 dólares", "sin_relacion"),
        ("Compré 3 cajas de té; después aboné 2 dólares de esa compra", "offsets"),
        ("COMPRÉ  2 cajas de café, y luego aboné 1 dólar de esa compra", "offsets"),
        ("Compré té. Pagué 2 dólares de la compra anterior", "offsets"),
    ]
    errores = 0
    print("\nRELACIONES, EVIDENCIA, OFFSETS E IDENTIFICADORES")
    print("-" * 60)
    for texto, tipo in casos:
        correcto = True
        try:
            salida = interpretar_operaciones_con_relaciones(texto, nlp=nlp)
            ops, relaciones = salida["operaciones"], salida["relaciones"]
            ids = {op.get("id") for op in ops}
            correcto = len(ids) == len(ops) and None not in ids
            for rel in relaciones:
                correcto = correcto and rel.get("operacion_origen_id") in ids
                if rel.get("estado") == "CONFIRMADA":
                    destino = next((op for op in ops if op.get("id") == rel.get("operacion_destino_id")), None)
                    correcto = correcto and destino is not None and destino.get("accion") == "comprar"
                if rel.get("operacion_destino_id") is not None:
                    correcto = correcto and rel["operacion_destino_id"] in ids
            if tipo == "confirmada":
                correcto = correcto and len(relaciones) == 1 and relaciones[0].get("estado") == "CONFIRMADA"
            elif tipo == "no_destino_no_admitido":
                correcto = correcto and all(r.get("estado") != "CONFIRMADA" for r in relaciones)
                correcto = correcto and not any(op.get("accion") == "comprar" for op in ops)
            elif tipo == "ambigua":
                correcto = correcto and len(relaciones) == 1 and relaciones[0].get("estado") == "AMBIGUA" and relaciones[0].get("operacion_destino_id") is None
            elif tipo == "sin_relacion":
                correcto = correcto and relaciones == []
            elif tipo == "offsets":
                campos = [meta for op in ops for meta in op.get("campos", {}).values()]
                for meta in campos:
                    evidencia = meta.get("evidencia")
                    if evidencia and evidencia.get("inicio") is not None and evidencia.get("fin") is not None:
                        inicio, fin = evidencia["inicio"], evidencia["fin"]
                        correcto = correcto and 0 <= inicio <= fin <= len(texto)
                        correcto = correcto and texto[inicio:fin] == evidencia.get("texto")
        except Exception as exc:
            correcto = False
            print(f"[FALLO] {texto!r} | excepción={type(exc).__name__}: {exc}")
        errores += int(not correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] {tipo}: {texto!r} | ops={[(o.get('id'), o.get('accion')) for o in salida.get('operaciones', [])] if 'salida' in locals() else []} | relaciones={salida.get('relaciones', []) if 'salida' in locals() else []}")
        if 'salida' in locals():
            del salida
    print(f"Resultado relaciones/evidencia: {len(casos) - errores}/{len(casos)} | fallos={errores}")
    return errores


def evaluar_interfaz_independiente(nlp):
    """Comprueba inicialización, tipos de entrada, límites y estado residual."""
    from pathlib import Path
    from interpretacion import (
        MODELO_BASE, MODELO_ENTRENADO, cargar_modelo, interpretar_operaciones_con_relaciones,
        interpretar_texto, serializar_resultado,
    )

    casos = []
    try:
        ruta_inexistente = Path(__file__).resolve().parent / "__modelo_ausente_para_regresion__"
        try:
            cargar_modelo(ruta_inexistente)
            casos.append(("modelo ausente", False, "se cargó sin existir"))
        except RuntimeError as exc:
            casos.append(("modelo ausente", "No se encontró" in str(exc) or "No se pudo" in str(exc), str(exc)))

        modelo_base = cargar_modelo(MODELO_BASE)
        modelo_custom = cargar_modelo(Path(__file__).resolve().parent / MODELO_ENTRENADO)
        casos.append(("cache por ruta", modelo_base is not modelo_custom and "ner" in modelo_custom.pipe_names, f"base={modelo_base.meta.get('name')}, custom labels={len(modelo_custom.get_pipe('ner').labels)}"))

        vacio = interpretar_operaciones_con_relaciones("", nlp=nlp)
        casos.append(("texto vacío estructurado", vacio.get("estado") == "NO_RECONOCIDO" and vacio.get("operaciones") == [], str(vacio)))
        invalido = interpretar_operaciones_con_relaciones(42, nlp=nlp)
        casos.append(("tipo inválido", invalido.get("estado") == "NO_RECONOCIDO" and bool(invalido.get("advertencias")), str(invalido)))

        muestra = interpretar_operaciones_con_relaciones(
            "Compré papel y aboné 2 dólares de esa compra", nlp=nlp
        )
        contrato_correcto = [op.get("id") for op in muestra["operaciones"]] == ["op_1", "op_2"]
        contrato_correcto = contrato_correcto and len(muestra["relaciones"]) == 1
        if muestra["relaciones"]:
            contrato_correcto = contrato_correcto and (
                muestra["relaciones"][0].get("estado") == "CONFIRMADA"
                and muestra["relaciones"][0].get("operacion_destino_id") == "op_1"
            )
        casos.append(("ejemplo público de contrato", contrato_correcto, str(muestra["relaciones"])))

        try:
            interpretar_texto("   ", nlp=nlp)
            casos.append(("API analítica vacía", False, "no lanzó ValueError"))
        except ValueError as exc:
            casos.append(("API analítica vacía", True, str(exc)))

        demasiado_largo = "x" * (nlp.max_length + 1)
        try:
            interpretar_operaciones_con_relaciones(demasiado_largo, nlp=nlp)
            casos.append(("texto extenso", False, "no respetó max_length"))
        except ValueError as exc:
            casos.append(("texto extenso", "máximo" in str(exc), str(exc)))

        frase = "Compré té por 7 dólares; después aboné 2 dólares de esa compra"
        primera = serializar_resultado(interpretar_operaciones_con_relaciones(frase, nlp=nlp))
        interpretar_operaciones_con_relaciones("Vendi cinco cajas por dies dólares", nlp=nlp)
        segunda = serializar_resultado(interpretar_operaciones_con_relaciones(frase, nlp=nlp))
        casos.append(("Unicode, repetición y estado residual", primera == segunda and "té" in primera, "salidas repetidas iguales" if primera == segunda else "las salidas difieren"))
    except Exception as exc:
        casos.append(("error inesperado", False, f"{type(exc).__name__}: {exc}"))

    aciertos = 0
    print("\nINTERFAZ INDEPENDIENTE, CARGA Y ERRORES")
    print("-" * 60)
    for nombre, correcto, detalle in casos:
        aciertos += int(correcto)
        print(f"[{'OK' if correcto else 'FALLO'}] {nombre}: {detalle}")
    total = 8
    # Un error inesperado puede interrumpir la construcción; cuenta los restantes como fallidos.
    aciertos = min(aciertos, total)
    print(f"Resultado interfaz: {aciertos}/{total}")
    return aciertos


def ejecutar_baterias(nlp, baterias):
    """Ejecuta todas las baterías y distingue fallos de casos no ejecutados."""
    aprobadas = fallos = no_ejecutadas = errores_ejecucion = 0
    for nombre, ejecutar, total in baterias:
        try:
            aciertos = ejecutar(nlp)
            if type(aciertos) is not int or not 0 <= aciertos <= total:
                raise TypeError(f"la batería devolvió {type(aciertos).__name__}={aciertos!r}, fuera de 0..{total}")
            aprobadas += aciertos
            fallos += total - aciertos
        except Exception as exc:
            errores_ejecucion += 1
            no_ejecutadas += total
            print(f"[EXCEPCIÓN] batería {nombre}: {type(exc).__name__}: {exc}")
    return aprobadas, fallos, no_ejecutadas, errores_ejecucion


def evaluar_segmentacion_ner(nlp):
    """Verifica proyección acotada, offsets y límites semánticos de segmentos."""
    from interpretacion import interpretar_operaciones_con_relaciones

    pruebas = [
        {
            "texto": "Compré cajas, pagué el transporte y vendí productos",
            "operaciones": [("comprar", None), ("pagar", None), ("vender", "productos")],
            "diagnostico": "proyeccion_sin_ancla_local",
            "span": "cajas",
        },
        {
            "texto": "Compré tomate y no vendí cebollas",
            "operaciones": [("comprar", "tomate")],
            "diagnostico": None,
            "span": None,
        },
        {
            "texto": "Mañana compraré tomate y hoy vendí arroz",
            "operaciones": [("vender", "arroz")],
            "diagnostico": None,
            "span": None,
        },
        {
            "texto": "Vendí tomate por 6 dólares y vendí fundas por 4 dólares",
            "operaciones": [("vender", "tomate"), ("vender", "fundas")],
            "diagnostico": None,
            "span": None,
        },
    ]
    aciertos = 0
    for caso in pruebas:
        resultado = interpretar_operaciones_con_relaciones(caso["texto"], nlp=nlp)
        operaciones = [(op.get("accion"), op.get("producto")) for op in resultado["operaciones"]]
        correcto = operaciones == caso["operaciones"] and "diagnosticos_ner" in resultado
        if caso["diagnostico"]:
            span_completo = next((
                ent for ent in nlp(caso["texto"]).ents
                if ent.text.casefold() == caso["span"].casefold()
            ), None)
            spans = [
                d["span"] for d in resultado["diagnosticos_ner"]
                if d.get("tipo") == caso["diagnostico"]
            ]
            if span_completo is not None:
                correcto = correcto and any(
                    caso["texto"][span["inicio"]:span["fin"]] == caso["span"]
                    for span in spans
                )
        if caso["texto"].startswith("Vendí tomate por") and len(resultado["operaciones"]) == 2:
            evidencia = resultado["operaciones"][1]["campos"]["producto"]["evidencia"]
            correcto = correcto and (
                evidencia is not None
                and caso["texto"][evidencia["inicio"]:evidencia["fin"]] == "fundas"
            )
        aciertos += int(correcto)
        estado = "OK" if correcto else "FALLO"
        print(f"[{estado}] segmentación NER {caso['texto']!r} | ops={operaciones} | diagnósticos={len(resultado.get('diagnosticos_ner', []))}")
    print(f"Resultado segmentación NER: {aciertos}/{len(pruebas)}")
    return aciertos


def construir_baterias():
    """Devuelve la matriz única de baterías usada por pruebas y promoción."""
    return [
        ("NER", lambda modelo: evaluar_ner(modelo, PRUEBAS_NUEVAS, "PRUEBAS NER", detallado=True), len(PRUEBAS_NUEVAS)),
        ("NER independiente", lambda modelo: evaluar_ner(modelo, PRUEBAS_NER_INDEPENDIENTE, "EVALUACIÓN NER INDEPENDIENTE"), len(PRUEBAS_NER_INDEPENDIENTE)),
        ("validación de datos", evaluar_validacion_datos, 8),
        ("interfaz independiente", evaluar_interfaz_independiente, 8),
        ("entidades", evaluar_entidades, len(PRUEBAS_ENTIDADES)),
        ("semántica", evaluar_semantica, len(PRUEBAS_SEMANTICAS)),
        ("operaciones múltiples", evaluar_operaciones_multiples, len(PRUEBAS_OPERACIONES_MULTIPLES)),
        ("relaciones", evaluar_relaciones, len(PRUEBAS_RELACIONES)),
        ("contrato", evaluar_contrato, len(PRUEBAS_CONTRATO)),
        ("adversarial", lambda modelo: len(EVALUACION_ADVERSARIAL) - len(evaluar_conjunto_independiente(modelo)[2]), len(EVALUACION_ADVERSARIAL)),
        ("fase 2", lambda modelo: len(PRUEBAS_REGRESION_FASE_2) - len(evaluar_regresiones_fase_2(modelo)[2]), len(PRUEBAS_REGRESION_FASE_2)),
        ("fase 4", evaluar_regresiones_fase_4, 17),
        ("endurecimiento", evaluar_endurecimiento_semantico, 37),
        ("admisión", evaluar_admision_operaciones, 26),
        ("matriz modalidad", lambda modelo: 10 - evaluar_matriz_modalidad(modelo), 10),
        ("relaciones/evidencia", lambda modelo: 8 - evaluar_relaciones_evidencia(modelo), 8),
        ("segmentación NER segura", evaluar_segmentacion_ner, 4),
    ]


def main():
    baterias = construir_baterias()
    total = sum(total for _, _, total in baterias)
    from interpretacion import cargar_modelo
    try:
        nlp = cargar_modelo()
    except (RuntimeError, OSError) as exc:
        print(f"[EXCEPCIÓN] carga del modelo: {type(exc).__name__}: {exc}")
        print(f"\nTOTAL FINAL: {total} | aprobaciones=0 | fallos=0 | no ejecutadas={total} | errores de ejecución=1")
        raise SystemExit(1) from exc
    aprobadas, fallos, no_ejecutadas, errores_ejecucion = ejecutar_baterias(nlp, baterias)
    print(f"\nTOTAL FINAL: {total} | aprobaciones={aprobadas} | fallos={fallos} | no ejecutadas={no_ejecutadas} | errores de ejecución={errores_ejecucion}")
    if fallos or no_ejecutadas or errores_ejecucion:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
