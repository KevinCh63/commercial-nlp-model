"""Familias de entrenamiento anotadas por construcción, sin importar pruebas."""
import random


def generar_datos():
    datos = []

    def agregar(partes):
        texto, entidades = "", []
        for parte in partes:
            fragmento, etiqueta = parte if isinstance(parte, tuple) else (parte, None)
            inicio = len(texto)
            texto += fragmento
            if etiqueta:
                entidades.append((inicio, len(texto), etiqueta))
        datos.append((texto, {"entities": entidades}))

    rng = random.Random(816)
    numeros = ["dos", "tres", "cinco", "ocho", "once", "quince", "veinte", "4", "9", "17", "1.5", "2,5"]
    unidades = ["sacos", "bidones", "frascos", "docenas", "metros", "kg", "kilos", "cajas", "botellas", "paquetes", "rollos", "baldes"]
    productos = ["avena", "cebada", "arroz integral", "jabón líquido", "aceite vegetal", "café molido", "cuerda trenzada", "nuez pecana", "lona impermeable", "pintura blanca", "canastas tejidas", "láminas metálicas", "vasos reciclables", "papel", "fundas", "tomate", "cebollas", "mercadería"]
    conceptos = ["mantenimiento preventivo", "transporte escolar", "limpieza del depósito", "reparación del motor", "reparar la puerta", "instalar el grifo", "publicidad", "alquiler", "mantenimiento vial", "servicio técnico"]
    for i in range(360):
        cantidad = f"{rng.choice(numeros)} {rng.choice(unidades)}"
        producto = rng.choice(productos)
        valor = str(rng.randrange(7, 980))
        precio = rng.choice([f"{valor} dólares", f"{valor} euros", f"{valor} pesos", f"${valor}", f"€{valor}", f"{valor},50 USD"])
        q, p, m = (cantidad, "QUANTITY"), (producto, "PRODUCT"), (precio, "PRICE")
        variantes = [
            ["El almacén compró ", q, " de ", p, " por ", m],
            ["Durante la jornada, por ", m, " recibieron ", q, " de ", p],
            ["Ayer vendimos, sin comisión, ", q, " de ", p, " por ", m],
            ["¿Cuánto pagó Inés por ", q, " de ", p, "? ", m],
            ["Entregamos ", q, " de ", p, " al negocio por ", m],
            ["Con ", m, " adquirió ", q, " de ", p],
            ["Vendieron ", q, " de ", p, " por un total de ", m],
            [q, " de ", p, " se colocaron a ", m],
            ["Con ", m, " compraron las ", p],
            ["Recibieron ", m, " por las ", p],
        ]
        agregar(variantes[i % len(variantes)])
    for i in range(100):
        concepto = (rng.choice(conceptos), "CONCEPT")
        precio = (f"{rng.randrange(11, 870)} dólares", "PRICE")
        agregar(rng.choice([
            ["La asociación de comerciantes desembolsó ayer ", precio, " para el ", concepto],
            ["Para ", concepto, ", gastaron ", precio],
            ["Mediante una transferencia se pagaron ayer ", precio, " por el ", concepto],
            ["Factura de ", concepto, ": ", precio, ", cancelados ayer"],
            ["Gastamos ", precio, " en ", concepto],
        ]))
    for i in range(100):
        producto = (rng.choice(productos), "PRODUCT")
        otro = (rng.choice(productos), "PRODUCT")
        precio = (f"{rng.randrange(2, 90)} dólares", "PRICE")
        agregar(rng.choice([
            ["Compramos ", producto, " y abonamos ", precio, " de esa compra"],
            ["No compré ", producto, ", pero vendí ", otro],
            ["Tal vez compré ", producto, ", pero ayer vendí ", otro],
            ["Compré ", producto, " y no vendí ", otro],
            ["Vendí ", producto, " y ", otro, " por ", precio],
            ["Compré ", producto, " por ", precio, " y aboné de la compra anterior"],
            ["Quiero comprar ", producto, ", pero ayer compré ", otro],
        ]))
    for texto in ["No compré ni pagué", "de esa compra", "de la compra anterior", "de una deuda anterior", "No hay operaciones hoy", "Compré y pagué", "un total de", "Mañana pagaré", "Creo que ya aboné"]:
        agregar([texto])
    # Evita duplicados generados por combinaciones equivalentes.
    return list({texto.casefold(): (texto, anotaciones) for texto, anotaciones in datos}.values())
