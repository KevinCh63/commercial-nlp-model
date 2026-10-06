import random

import spacy
from spacy.training.example import Example
from spacy.util import fix_random_seed, minibatch

from pruebas import PRUEBAS_NUEVAS, evaluar_ner

MODELO_BASE = "es_core_news_md"
MODELO_GUARDADO = "custom_ner_model"
EPOCAS = 100
DROP = 0.3
SEMILLA = 42

train_data = [
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
    from interpretacion import mostrar_bateria_de_pruebas
    mostrar_bateria_de_pruebas(recargado)


if __name__ == "__main__":
    main()
