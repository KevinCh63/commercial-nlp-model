import random
import hashlib
from difflib import SequenceMatcher
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import uuid
import contextlib
import io

import spacy
from spacy.training.example import Example
from spacy.util import fix_random_seed, minibatch

from pruebas import PRUEBAS_NER_VALIDACION, evaluar_ner, medir_ner

MODELO_BASE = "es_core_news_md"
MODELO_GUARDADO = "custom_ner_model"
EPOCAS = 100
DROP = 0.3
SEMILLA = 42
ETIQUETAS_PERSONALIZADAS = {"QUANTITY", "PRODUCT", "PRICE", "CONCEPT"}


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

# Ampliación guiada por familias de error observadas (límites multi-token,
# productos compuestos, monedas/símbolos y conceptos de gasto), no por copia de
# las frases de evaluación independiente.
DATOS_COMPLEMENTARIOS = [
    _crear_dato("Durante la feria entraron 63 euros por nueve paquetes de cacao.", [("63 euros", "PRICE"), ("nueve paquetes", "QUANTITY"), ("cacao", "PRODUCT")]),
    _crear_dato("La administración abonó 275 dólares para reparar el montacargas.", [("275 dólares", "PRICE"), ("reparar el montacargas", "CONCEPT")]),
    _crear_dato("Salieron veinte frascos de salsa picante a 44 USD.", [("veinte frascos", "QUANTITY"), ("salsa picante", "PRODUCT"), ("44 USD", "PRICE")]),
    _crear_dato("Se cobró $71 por ocho bandejas de queso fresco.", [("$71", "PRICE"), ("ocho bandejas", "QUANTITY"), ("queso fresco", "PRODUCT")]),
    _crear_dato("Marta adquirió 2 rollos de malla plástica por 37 euros.", [("2 rollos", "QUANTITY"), ("malla plástica", "PRODUCT"), ("37 euros", "PRICE")]),
    _crear_dato("El servicio de agua costó 56 dólares en el trimestre.", [("servicio de agua", "CONCEPT"), ("56 dólares", "PRICE")]),
    _crear_dato("Recibimos treinta y cuatro pesos por cuatro bolsas de lentejas.", [("treinta y cuatro pesos", "PRICE"), ("cuatro bolsas", "QUANTITY"), ("lentejas", "PRODUCT")]),
    _crear_dato("Con 12,5 dólares se pagaron los repuestos del horno.", [("12,5 dólares", "PRICE"), ("repuestos del horno", "CONCEPT")]),
    _crear_dato("Ocho docenas de vasos biodegradables valieron 105 euros.", [("Ocho docenas", "QUANTITY"), ("vasos biodegradables", "PRODUCT"), ("105 euros", "PRICE")]),
    _crear_dato("La compra de alambre galvanizado ascendió a USD 82.", [("alambre galvanizado", "PRODUCT"), ("USD 82", "PRICE")]),
    _crear_dato("El puesto vendió cerámica artesanal por 190 dólares.", [("cerámica artesanal", "PRODUCT"), ("190 dólares", "PRICE")]),
    _crear_dato("Pagamos $23,40 por el envío de herramientas.", [("$23,40", "PRICE"), ("envío de herramientas", "CONCEPT")]),
    _crear_dato("Nos llegaron diecisiete unidades de aceite de oliva, facturadas en 68 euros.", [("diecisiete unidades", "QUANTITY"), ("aceite de oliva", "PRODUCT"), ("68 euros", "PRICE")]),
    _crear_dato("Se asignaron 540 pesos a pintar las paredes del almacén.", [("540 pesos", "PRICE"), ("pintar las paredes del almacén", "CONCEPT")]),
    _crear_dato("Compró tres metros de cuerda náutica por 16 euros.", [("tres metros", "QUANTITY"), ("cuerda náutica", "PRODUCT"), ("16 euros", "PRICE")]),
    _crear_dato("Por 29 USD salieron seis paquetes de semillas orgánicas.", [("29 USD", "PRICE"), ("seis paquetes", "QUANTITY"), ("semillas orgánicas", "PRODUCT")]),
    _crear_dato("Hubo un ingreso de 48 euros por nueve cajones de naranjas.", [("48 euros", "PRICE"), ("nueve cajones", "QUANTITY"), ("naranjas", "PRODUCT")]),
    _crear_dato("La reparación de la vitrina requirió 112 dólares.", [("reparación de la vitrina", "CONCEPT"), ("112 dólares", "PRICE")]),
    _crear_dato("Se compraron treinta y cinco latas de atún en 210 euros.", [("treinta y cinco latas", "QUANTITY"), ("atún", "PRODUCT"), ("210 euros", "PRICE")]),
    _crear_dato("Facturaron 4,25 euros por el alquiler del carrito.", [("4,25 euros", "PRICE"), ("alquiler del carrito", "CONCEPT")]),
    _crear_dato("Transferí 325 USD para adquirir cartuchos de tinta.", [("325 USD", "PRICE"), ("cartuchos de tinta", "PRODUCT")]),
    _crear_dato("El comedor recibió catorce kilos de queso por 98 dólares.", [("catorce kilos", "QUANTITY"), ("queso", "PRODUCT"), ("98 dólares", "PRICE")]),
    _crear_dato("Hubo que pagar 21 euros por trasladar el congelador.", [("21 euros", "PRICE"), ("trasladar el congelador", "CONCEPT")]),
    _crear_dato("La bodega despachó 3 cajas de frascos ámbar por 47 euros.", [("3 cajas", "QUANTITY"), ("frascos ámbar", "PRODUCT"), ("47 euros", "PRICE")]),
    _crear_dato("Se reservaron 65 dólares para cambiar el cableado.", [("65 dólares", "PRICE"), ("cambiar el cableado", "CONCEPT")]),
    _crear_dato("Luisa gastó 39 euros en la instalación de paneles solares.", [("39 euros", "PRICE"), ("instalación de paneles solares", "CONCEPT")]),
    _crear_dato("La recaudación por venta de miel de abeja fue 83 USD.", [("miel de abeja", "PRODUCT"), ("83 USD", "PRICE")]),
    _crear_dato("Llegaron cinco envases de detergente neutro, facturados en 32 euros.", [("cinco envases", "QUANTITY"), ("detergente neutro", "PRODUCT"), ("32 euros", "PRICE")]),
    _crear_dato("Limpieza: se cargaron 44 dólares al presupuesto.", [("Limpieza", "CONCEPT"), ("44 dólares", "PRICE")]),
]
train_data.extend(DATOS_COMPLEMENTARIOS)


def validar_train_data(nlp, datos):
    errores = []
    if not isinstance(datos, (list, tuple)) or not datos:
        raise ValueError("Se esperaba una lista no vacía de ejemplos NER.")
    textos_vistos = set()
    for indice, ejemplo in enumerate(datos):
        if not isinstance(ejemplo, (list, tuple)) or len(ejemplo) != 2:
            errores.append(f"ejemplo {indice}: se esperaba (texto, anotaciones)")
            continue
        texto, anotaciones = ejemplo
        if not isinstance(texto, str) or not texto.strip():
            errores.append(f"ejemplo {indice}: el texto debe ser una cadena no vacía")
            continue
        clave_texto = texto.casefold()
        if clave_texto in textos_vistos:
            errores.append(f"ejemplo {indice}: texto duplicado {texto!r}")
        textos_vistos.add(clave_texto)
        if not isinstance(anotaciones, dict) or not isinstance(anotaciones.get("entities"), (list, tuple)):
            errores.append(f"{texto!r}: falta una lista de entidades")
            continue
        try:
            doc = nlp.make_doc(texto)
        except ValueError as exc:
            errores.append(f"{texto!r}: no se pudo tokenizar: {exc}")
            continue
        vistos = []
        for indice_entidad, entidad in enumerate(anotaciones["entities"]):
            if not isinstance(entidad, (list, tuple)) or len(entidad) != 3:
                errores.append(f"{texto!r}: entidad {indice_entidad} debe ser (inicio, fin, etiqueta)")
                continue
            inicio, fin, etiqueta = entidad
            if not isinstance(etiqueta, str) or etiqueta not in ETIQUETAS_PERSONALIZADAS:
                errores.append(f"{texto!r}: etiqueta {etiqueta!r} no permitida; use {sorted(ETIQUETAS_PERSONALIZADAS)}")
            if isinstance(inicio, bool) or isinstance(fin, bool) or not isinstance(inicio, int) or not isinstance(fin, int):
                errores.append(f"{texto!r}: offsets deben ser enteros, recibidos {inicio!r}:{fin!r}")
                continue
            if not (0 <= inicio < fin <= len(texto)):
                errores.append(f"{texto!r}: {etiqueta} {inicio}:{fin} fuera de rango (longitud {len(texto)})")
                continue
            fragmento = texto[inicio:fin]
            if not fragmento.strip():
                errores.append(f"{texto!r}: {etiqueta} {inicio}:{fin} produce una entidad vacía")
            if doc.char_span(inicio, fin) is None:
                errores.append(f"{texto!r}: {etiqueta} {inicio}:{fin} -> {fragmento!r} no está alineado a los límites de los tokens")
            for otro_inicio, otro_fin, otra_etiqueta in vistos:
                if inicio < otro_fin and otro_inicio < fin:
                    errores.append(f"{texto!r}: {etiqueta} y {otra_etiqueta} se solapan")
            vistos.append((inicio, fin, etiqueta))
    if errores:
        raise ValueError("train_data inválido:\n  - " + "\n  - ".join(errores))


def auditar_solapamiento(datos_entrenamiento, datos_evaluacion, umbral=0.82):
    """Devuelve pares exactos y textualmente cercanos entre dos conjuntos."""
    normalizar = lambda texto: " ".join(re.findall(r"\w+", texto.casefold(), flags=re.UNICODE))
    entrenamiento = [normalizar(texto) for texto, _ in datos_entrenamiento]
    exactos, cercanos = [], []
    for indice_eval, (texto, _) in enumerate(datos_evaluacion):
        normalizado = normalizar(texto)
        for indice_train, referencia in enumerate(entrenamiento):
            similitud = SequenceMatcher(None, normalizado, referencia).ratio()
            if similitud == 1.0:
                exactos.append((indice_eval, indice_train, similitud))
            elif similitud >= umbral:
                cercanos.append((indice_eval, indice_train, similitud))
    return exactos, cercanos


def entrenar_ner(nlp, datos):
    fix_random_seed(SEMILLA)
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
    destino = Path(ruta)
    if not destino.is_absolute():
        destino = Path(__file__).resolve().parent / destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    candidato = destino.with_name(f".{destino.name}.candidate-{uuid.uuid4().hex}")
    respaldo = destino.with_name(f".{destino.name}.previous-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")
    nlp_entrenado.to_disk(candidato)
    with open(candidato / "proyecto_metadata.json", "w", encoding="utf-8") as archivo:
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
    # Comprueba que el artefacto temporal carga antes de mover el modelo vigente.
    # La comprobación usa también el cargador real consumido por el intérprete.
    from interpretacion import cargar_modelo
    cargar_modelo(candidato)
    tenia_modelo = destino.exists()
    if tenia_modelo:
        destino.rename(respaldo)
    try:
        candidato.rename(destino)
    except Exception:
        if tenia_modelo and respaldo.exists() and not destino.exists():
            respaldo.rename(destino)
        raise
    recargado = cargar_modelo(destino)
    print(f"\nModelo guardado en {str(destino)!r} y recargado correctamente.")
    if tenia_modelo:
        print(f"Modelo previo preservado en {str(respaldo)!r}.")
    print(f"Pipes del modelo guardado: {recargado.pipe_names}")
    print(f"Etiquetas NER: {list(recargado.get_pipe('ner').labels)}")
    return recargado


def medir_baterias(nlp):
    """Mide cada batería histórica con la misma matriz que `pruebas.py`."""
    from pruebas import construir_baterias

    resultados = {}
    for nombre, ejecutar, total in construir_baterias():
        try:
            # Las baterías ya detallan sus fallos en pruebas.py; aquí interesa
            # comparar el resultado por grupo sin duplicar toda esa salida.
            with contextlib.redirect_stdout(io.StringIO()):
                aprobadas = ejecutar(nlp)
            if type(aprobadas) is not int or not 0 <= aprobadas <= total:
                raise TypeError(f"resultado inválido {aprobadas!r}/{total}")
            resultados[nombre] = {"aprobadas": aprobadas, "total": total, "error": None}
        except Exception as exc:
            resultados[nombre] = {"aprobadas": 0, "total": total, "error": f"{type(exc).__name__}: {exc}"}
    return resultados


def main():
    fix_random_seed(SEMILLA)
    nlp = spacy.load(MODELO_BASE)
    validar_train_data(nlp, train_data)
    validar_train_data(nlp, PRUEBAS_NER_VALIDACION)
    exactos, cercanos = auditar_solapamiento(train_data, PRUEBAS_NER_VALIDACION)
    if exactos or cercanos:
        raise ValueError(f"El conjunto de desarrollo se solapa con entrenamiento: exactos={len(exactos)}, cercanos={len(cercanos)}")
    print(f"Datos validados: {len(train_data)} entrenamiento, {len(PRUEBAS_NER_VALIDACION)} desarrollo | pipes base = {nlp.pipe_names}\n")
    ruta_existente = Path(__file__).resolve().parent / MODELO_GUARDADO
    from interpretacion import cargar_modelo
    baseline = cargar_modelo(ruta_existente) if ruta_existente.exists() else spacy.load(MODELO_BASE)
    metricas_base_desarrollo = medir_ner(baseline, PRUEBAS_NER_VALIDACION)
    baterias_base = medir_baterias(baseline)
    entrenar_ner(nlp, train_data)
    evaluar_ner(nlp, train_data, "ENTRENAMIENTO (no es generalización)")
    ruta_candidata_eval = Path(__file__).resolve().parent / f".{MODELO_GUARDADO}.evaluation-{uuid.uuid4().hex}"
    nlp.to_disk(ruta_candidata_eval)
    try:
        candidata_cargada = cargar_modelo(ruta_candidata_eval)
        metricas_candidatas_desarrollo = medir_ner(candidata_cargada, PRUEBAS_NER_VALIDACION)
        baterias_candidata = medir_baterias(candidata_cargada)
    finally:
        import shutil
        shutil.rmtree(ruta_candidata_eval, ignore_errors=True)
    print("\nComparación desarrollo (micro-F1 y exactitud de frase, baseline -> candidata):", (
        metricas_base_desarrollo["micro_entidad"]["f1"], metricas_base_desarrollo["coincidencia_exacta_de_frase"]["correctas"]
    ), "->", (
        metricas_candidatas_desarrollo["micro_entidad"]["f1"], metricas_candidatas_desarrollo["coincidencia_exacta_de_frase"]["correctas"]
    ))
    f1_base = metricas_base_desarrollo["micro_entidad"]["f1"]
    f1_candidata = metricas_candidatas_desarrollo["micro_entidad"]["f1"]
    frases_base = metricas_base_desarrollo["coincidencia_exacta_de_frase"]["correctas"]
    frases_candidata = metricas_candidatas_desarrollo["coincidencia_exacta_de_frase"]["correctas"]
    regresiones = [
        nombre for nombre, base in baterias_base.items()
        if baterias_candidata[nombre]["error"] is not None
        or baterias_candidata[nombre]["aprobadas"] < base["aprobadas"]
    ]
    if f1_candidata < f1_base or frases_candidata < frases_base or regresiones:
        print("Candidata rechazada; se conserva el modelo vigente.")
        if f1_candidata < f1_base or frases_candidata < frases_base:
            print("Retrocede en desarrollo NER.")
        if regresiones:
            print("Baterías con regresión:", ", ".join(regresiones))
        raise SystemExit(1)
    if f1_candidata == f1_base and frases_candidata == frases_base:
        print("Candidata sin mejora en desarrollo; el modelo existente se conserva sin sobrescribirlo.")
        return
    print("Baterías sin regresión; promoción habilitada por mejora en desarrollo.")
    recargado = comprobar_guardado(nlp)
    from interpretacion import mostrar_bateria_de_pruebas
    mostrar_bateria_de_pruebas(recargado)


if __name__ == "__main__":
    main()
