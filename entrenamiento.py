import random
import hashlib
from datetime import datetime, timezone
import json

import spacy
from spacy.training.example import Example
from spacy.util import fix_random_seed, minibatch

from pruebas import PRUEBAS_NUEVAS, evaluar_ner

MODELO_BASE = "es_core_news_md"
MODELO_GUARDADO = "custom_ner_model"
EPOCAS = 100
DROP = 0.3
SEMILLA = 42


def _crear_dato(texto, entidades):
    anotaciones = []
    for fragmento, etiqueta in entidades:
        inicio = texto.index(fragmento)
        anotaciones.append((inicio, inicio + len(fragmento), etiqueta))
    return texto, {"entities": anotaciones}

train_data = [
    _crear_dato("vendí 7 botellas de agua por 12 dólares", [("7 botellas", "QUANTITY"), ("agua", "PRODUCT"), ("12 dólares", "PRICE")]),
    _crear_dato("El cliente abonó 8 dólares", [("8 dólares", "PRICE")]),
    ("vendí 3 libras de tomate a 2 dólares", {"entities": [(6, 14, "QUANTITY"), (18, 24, "PRODUCT"), (27, 36, "PRICE")]}),
    ("vendí 3 cajas de frutas a 15 dólares", {"entities": [(6, 13, "QUANTITY"), (17, 23, "PRODUCT"), (26, 36, "PRICE")]}),
    ("vendimos 10 productos por 50 dólares", {"entities": [(9, 11, "QUANTITY"), (12, 21, "PRODUCT"), (26, 36, "PRICE")]}),
    ("vendí 5 camisetas por 30 dólares", {"entities": [(6, 7, "QUANTITY"), (8, 17, "PRODUCT"), (22, 32, "PRICE")]}),
    ("vendí 2 camisetas por 15 USD", {"entities": [(6, 7, "QUANTITY"), (8, 17, "PRODUCT"), (22, 28, "PRICE")]}),
    ("compré 5 camisetas por 40 dólares", {"entities": [(7, 8, "QUANTITY"), (9, 18, "PRODUCT"), (23, 33, "PRICE")]}),
    ("compré 3 camisetas", {"entities": [(7, 8, "QUANTITY"), (9, 18, "PRODUCT")]}),
    ("María compró un producto por 10 dólares", {"entities": [(13, 15, "QUANTITY"), (16, 24, "PRODUCT"), (29, 39, "PRICE")]}),
    ("Juan me pagó 20 dólares", {"entities": [(13, 23, "PRICE")]}),
    ("Carlos me abonó 5", {"entities": [(16, 17, "PRICE")]}),
    ("Carlos quedó debiendo 10", {"entities": [(22, 24, "PRICE")]}),
    ("gasté 5 dólares en transporte", {"entities": [(6, 15, "PRICE"), (19, 29, "CONCEPT")]}),
    ("gasté 30 dólares en transporte", {"entities": [(6, 16, "PRICE"), (20, 30, "CONCEPT")]}),
    ("gasté 50 dólares en publicidad", {"entities": [(6, 16, "PRICE"), (20, 30, "CONCEPT")]}),
    ("gasté 30 dólares en alquiler", {"entities": [(6, 16, "PRICE"), (20, 28, "CONCEPT")]}),
    ("gastamos 20 en materiales", {"entities": [(9, 11, "PRICE"), (15, 25, "CONCEPT")]}),
    ("gastamos 20 dólares en materiales", {"entities": [(9, 19, "PRICE"), (23, 33, "CONCEPT")]}),
    ("gastamos 25 dólares en electricidad", {"entities": [(9, 19, "PRICE"), (23, 35, "CONCEPT")]}),
    ("El Pepe vendió 6 botellas de miel a 18 dólares", {"entities": [(15, 25, "QUANTITY"), (29, 33, "PRODUCT"), (36, 46, "PRICE")]}),
    ("gasté 25 dólares en el alquiler", {"entities": [(6, 16, "PRICE"), (23, 31, "CONCEPT")]}),
    ("Ayer vendió María 5 bolsas de café a 15 dólares", {"entities": [(18, 26, "QUANTITY"), (30, 34, "PRODUCT"), (37, 47, "PRICE")]}),
    ("Se vendieron 3 botellas de jugo por 14 dólares", {"entities": [(13, 14, "QUANTITY"), (15, 23, "PRODUCT"), (36, 46, "PRICE")]}),
    ("Se compraron 2 paquetes de servilletas por 16 dólares", {"entities": [(13, 14, "QUANTITY"), (15, 23, "PRODUCT"), (43, 53, "PRICE")]}),
    ("Se gastaron 12 dólares en limpieza", {"entities": [(12, 22, "PRICE"), (26, 34, "CONCEPT")]}),
    ("compré 4 lámparas por 30 dólares", {"entities": [(7, 8, "QUANTITY"), (9, 17, "PRODUCT"), (22, 32, "PRICE")]}),
    ("vendimos 8 focos por 25 dólares", {"entities": [(9, 10, "QUANTITY"), (11, 16, "PRODUCT"), (21, 31, "PRICE")]}),
    ("Adquirí 3 cuadernos por 12 dólares", {"entities": [(8, 9, "QUANTITY"), (10, 19, "PRODUCT"), (24, 34, "PRICE")]}),
    ("ofreció 7 botellas de jugo por 21 dólares", {"entities": [(8, 9, "QUANTITY"), (10, 18, "PRODUCT"), (31, 41, "PRICE")]}),
    ("Por 18 dólares vendimos 5 paquetes de galletas", {"entities": [(4, 14, "PRICE"), (24, 25, "QUANTITY"), (26, 34, "PRODUCT")]}),
    ("12 dólares pagó Marta por 2 cables", {"entities": [(0, 10, "PRICE"), (26, 27, "QUANTITY"), (28, 34, "PRODUCT")]}),
    ("Por 16 dólares, compró Luis 3 carpetas", {"entities": [(4, 14, "PRICE"), (28, 29, "QUANTITY"), (30, 38, "PRODUCT")]}),
    ("vendí tres carpetas por 15 dólares", {"entities": [(6, 10, "QUANTITY"), (11, 19, "PRODUCT"), (24, 34, "PRICE")]}),
    ("compramos ocho vasos por 20 dólares", {"entities": [(10, 14, "QUANTITY"), (15, 20, "PRODUCT"), (25, 35, "PRICE")]}),
    ("compré 6 linternas", {"entities": [(7, 8, "QUANTITY"), (9, 18, "PRODUCT")]}),
    _crear_dato("vendió 9 botellas de aceite por 22 dólares", [("9 botellas", "QUANTITY"), ("aceite", "PRODUCT"), ("22 dólares", "PRICE")]),
    _crear_dato("Se vendieron 6 cajas de galletas por 24 dólares", [("6 cajas", "QUANTITY"), ("galletas", "PRODUCT"), ("24 dólares", "PRICE")]),
    _crear_dato("María adquirió 2 cajas de cartón por 11 dólares", [("2 cajas", "QUANTITY"), ("cartón", "PRODUCT"), ("11 dólares", "PRICE")]),
    _crear_dato("compré cables por 8", [("cables", "PRODUCT"), ("8", "PRICE")]),
    _crear_dato("Se vendió una cartera", [("una", "QUANTITY"), ("cartera", "PRODUCT")]),
    _crear_dato("vendió 11 vasos por 9 euros", [("11", "QUANTITY"), ("vasos", "PRODUCT"), ("9 euros", "PRICE")]),
    _crear_dato("compró 4 mochilas por 18 dólares", [("4", "QUANTITY"), ("mochilas", "PRODUCT"), ("18 dólares", "PRICE")]),
    _crear_dato("vendió 7 lámparas por 16 dólares", [("7", "QUANTITY"), ("lámparas", "PRODUCT"), ("16 dólares", "PRICE")]),
    _crear_dato("La persona abonó 6 dólares", [("6 dólares", "PRICE")]),
]


def validar_train_data(nlp, datos):
    errores = []
    for texto, anotaciones in datos:
        doc = nlp.make_doc(texto)
        vistos = []
        for inicio, fin, etiqueta in anotaciones["entities"]:
            fragmento = texto[inicio:fin]
            if not (0 <= inicio < fin <= len(texto)):
                errores.append(f"{texto!r}: {etiqueta} {inicio}:{fin} fuera de rango (longitud {len(texto)})")
                continue
            if doc.char_span(inicio, fin) is None:
                errores.append(f"{texto!r}: {etiqueta} {inicio}:{fin} -> {fragmento!r} no está alineado a los límites de los tokens")
            for otro_inicio, otro_fin, otra_etiqueta in vistos:
                if inicio < otro_fin and otro_inicio < fin:
                    errores.append(f"{texto!r}: {etiqueta} y {otra_etiqueta} se solapan")
            vistos.append((inicio, fin, etiqueta))
    if errores:
        raise ValueError("train_data inválido:\n  - " + "\n  - ".join(errores))


def entrenar_ner(nlp, datos):
    ner = nlp.get_pipe("ner")
    for _, anotaciones in datos:
        for _, _, etiqueta in anotaciones["entities"]:
            if etiqueta not in ner.labels:
                ner.add_label(etiqueta)
    otros_pipes = [p for p in nlp.pipe_names if p != "ner"]
    with nlp.disable_pipes(*otros_pipes):
        nlp.initialize()
        ejemplos = [Example.from_dict(nlp.make_doc(texto), anotaciones) for texto, anotaciones in datos]
        for epoca in range(EPOCAS):
            random.shuffle(ejemplos)
            losses = {}
            for lote in minibatch(ejemplos, size=8):
                nlp.update(lote, drop=DROP, losses=losses)
            if (epoca + 1) % 20 == 0 or epoca == 0:
                perdidas = {k: round(float(v), 3) for k, v in losses.items()}
                print(f"Época {epoca + 1:3d}: losses={perdidas}")
    return nlp


def comprobar_guardado(nlp_entrenado, ruta=MODELO_GUARDADO):
    nlp_entrenado.to_disk(ruta)
    with open(f"{ruta}/proyecto_metadata.json", "w", encoding="utf-8") as archivo:
        huella_datos = hashlib.sha256(
            json.dumps(train_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        json.dump({
            "fecha_entrenamiento_utc": datetime.now(timezone.utc).isoformat(),
            "modelo_base": MODELO_BASE,
            "epocas": EPOCAS,
            "dropout": DROP,
            "semilla": SEMILLA,
            "entidades": sorted(nlp_entrenado.get_pipe("ner").labels),
            "cantidad_datos_entrenamiento": len(train_data),
            "huella_datos_entrenamiento": huella_datos,
        }, archivo, ensure_ascii=False, indent=2)
    recargado = spacy.load(ruta)
    print(f"\nModelo guardado en {ruta!r} y recargado correctamente.")
    print(f"Pipes del modelo guardado: {recargado.pipe_names}")
    print(f"Etiquetas NER: {list(recargado.get_pipe('ner').labels)}")
    return recargado


def main():
    fix_random_seed(SEMILLA)
    nlp = spacy.load(MODELO_BASE)
    validar_train_data(nlp, train_data)
    validar_train_data(nlp, PRUEBAS_NUEVAS)
    print(f"Datos validados: {len(train_data)} de entrenamiento, {len(PRUEBAS_NUEVAS)} de prueba | pipes base = {nlp.pipe_names}\n")
    entrenar_ner(nlp, train_data)
    recargado = comprobar_guardado(nlp)
    evaluar_ner(recargado, train_data, "ENTRENAMIENTO (sí están en train_data)")
    evaluar_ner(recargado, PRUEBAS_NUEVAS, "GENERALIZACIÓN (NO están en train_data)")
    from pruebas import evaluar_semantica
    evaluar_semantica(recargado)
    from interpretacion import mostrar_bateria_de_pruebas
    mostrar_bateria_de_pruebas(recargado)


if __name__ == "__main__":
    main()
