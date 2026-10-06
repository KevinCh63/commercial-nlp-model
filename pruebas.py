PRUEBAS_NUEVAS = [
    ("vendí 4 camisas por 25 dólares", {"entities": [(6, 7, "QUANTITY"), (8, 15, "PRODUCT"), (20, 30, "PRICE")]}),
    ("compramos 8 productos por 60 dólares", {"entities": [(10, 11, "QUANTITY"), (12, 21, "PRODUCT"), (26, 36, "PRICE")]}),
    ("Juan pagó 15 dólares", {"entities": [(10, 20, "PRICE")]}),
    ("gasté 40 dólares en electricidad", {"entities": [(6, 16, "PRICE"), (20, 32, "CONCEPT")]}),
    ("vendieron 6 cajas de frutas por 30 dólares", {"entities": [(10, 17, "QUANTITY"), (21, 27, "PRODUCT"), (32, 42, "PRICE")]}),
    ("El Mateo vendió 5 latas de sardina de 2.50$", {"entities": [(16, 23, "QUANTITY"), (27, 34, "PRODUCT"), (38, 43, "PRICE")]}),
    ("Carlos quedó debiendo 10", {"entities": [(22, 24, "PRICE")]}),
]


def evaluar_ner(nlp, pruebas, titulo):

    print(f"\n{titulo}")
    print("-" * 60)

    aciertos = 0

    for texto, anotaciones in pruebas:
        doc = nlp(texto)

        esperado = sorted(anotaciones["entities"])
        obtenido = sorted((e.start_char, e.end_char, e.label_) for e in doc.ents)

        correcto = esperado == obtenido
        if correcto:
            aciertos += 1

        marca = "OK  " if correcto else "FALLO"
        print(f"[{marca}] {texto!r}")
        if not correcto:
            print(f"        esperado: {esperado}")
            print(f"        obtenido: {obtenido}")

    print(f"Resultado: {aciertos}/{len(pruebas)} frases correctas")
    return aciertos
