from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

import spacy

MODELO_ENTRENADO = "custom_ner_model"
MODELO_BASE = "es_core_news_md"
VERBOS_EGRESO = {"gastar", "comprar", "adquirir", "pagar", "abonar", "desembolsar", "entregar"}
VERBOS_INGRESO = {"vender", "cobrar", "recibir", "ingresar", "entregar"}
VERBOS_DIRECCIONALES = {"pagar", "abonar", "entregar"}
ACCIONES_CANONICAS = {"adquirir": "comprar", "desembolsar": "gastar", "salir": "gastar", "entrar": "recibir"}
LEMAS_CORREGIDOS = {"vendar": "vender"}
FORMAS_SIN_DIACRITICO = {"compre": "comprar", "vendi": "vender", "pague": "pagar", "gaste": "gastar", "abone": "abonar", "compro": "comprar", "pago": "pagar"}
PERSONA_POR_RASGOS = {
    ("1", "Sing"): "YO", ("1", "Plur"): "NOSOTROS", ("2", "Sing"): "TU",
    ("2", "Plur"): "USTEDES", ("3", "Sing"): "TERCERA_PERSONA",
    ("3", "Plur"): "TERCERA_PERSONA_PLURAL",
}
FORMAS_NO_RECONOCIDAS = {
    "gasté": ("gastar", "1", "Sing"), "vendí": ("vender", "1", "Sing"),
    "vendimos": ("vender", "1", "Plur"), "pagué": ("pagar", "1", "Sing"),
    "compré": ("comprar", "1", "Sing"), "adquirí": ("adquirir", "1", "Sing"),
    "gastamos": ("gastar", "1", "Plur"), "compramos": ("comprar", "1", "Plur"),
    "quedé": ("quedar", "1", "Sing"), "quedamos": ("quedar", "1", "Plur"),
    "vendió": ("vender", "3", "Sing"), "compró": ("comprar", "3", "Sing"),
    "pagó": ("pagar", "3", "Sing"), "pagamos": ("pagar", "1", "Plur"),
    "pagué": ("pagar", "1", "Sing"), "cobró": ("cobrar", "3", "Sing"),
    "recibió": ("recibir", "3", "Sing"), "gastó": ("gastar", "3", "Sing"),
    "aboné": ("abonar", "1", "Sing"), "abonó": ("abonar", "3", "Sing"),
    "entregué": ("entregar", "1", "Sing"),
}
CLAVES_SEMANTICAS = (
    "agente", "contraparte", "accion", "tipo", "cantidad", "producto", "concepto",
    "precio", "moneda", "estado", "faltantes", "advertencias",
)
_nlp = None
_nlp_ruta = None
_NUMERO = r"(?:\d+(?:[.,]\d+)?|un(?:a|o)?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|varios|varias)"
_MONEDAS = r"(?:d[oó]lar(?:es)?|usd|euro(?:s)?|eur|peso(?:s)?|\$)"
_UNIDADES = r"(?:botellas?|cajas?|paquetes?|bolsas?|latas?|vasos?|unidades?|libras?|kilos?|kilogramos?|gramos?|metros?|litros?)"
_UNIDADES_PRODUCTO = {"vaso", "vasos"}
_MONEDAS_NUMERO = r"(?:d[oó]lar(?:es)?|usd|euro(?:s)?|eur|peso(?:s)?)"
_PRODUCTOS_GENERICOS = {"producto", "productos", "artículo", "artículos", "articulo", "articulos", "mercadería", "mercaderia", "mercancía", "mercancia", "artículos varios", "articulos varios"}
_REFERENCIAS_GENERICAS = {"algo", "nada", "cosa", "cosas", "alguna cosa", "algunas cosas"}
_CUANTIFICADORES_VAGOS = {"varios", "varias", "algunos", "algunas", "muchos", "muchas"}


def cargar_modelo(ruta: str = MODELO_ENTRENADO):
    """Carga y cachea el modelo solicitado; no sustituye un modelo ausente."""
    global _nlp, _nlp_ruta
    if not isinstance(ruta, (str, Path)) or not str(ruta).strip():
        raise ValueError("ruta debe ser un nombre o una ruta de modelo no vacía")
    nombre = str(ruta)
    candidato = Path(nombre)
    if not candidato.is_absolute() and (nombre == MODELO_ENTRENADO or "/" in nombre or "\\" in nombre):
        candidato = Path(__file__).resolve().parent / candidato
        objetivo = candidato.resolve()
        clave = str(objetivo)
    elif candidato.is_absolute():
        objetivo = candidato.resolve()
        clave = str(objetivo)
    else:
        objetivo, clave = nombre, nombre
    if _nlp is not None and _nlp_ruta == clave:
        return _nlp
    try:
        modelo = spacy.load(str(objetivo))
    except (OSError, ValueError, ImportError) as exc:
        if nombre == MODELO_ENTRENADO:
            detalle = f"No se encontró o no se pudo abrir el modelo personalizado {str(objetivo)!r}. Entrénelo con `python entrenamiento.py` o indique una ruta válida."
        else:
            detalle = f"No se pudo cargar el modelo {str(objetivo)!r}. Verifique que esté instalado y que sea compatible con spaCy {spacy.__version__}."
        raise RuntimeError(detalle) from exc
    _nlp, _nlp_ruta = modelo, clave
    return _nlp


def rasgo_morfologico(token, nombre: str):
    if token is None:
        return None
    valores = token.morph.get(nombre)
    return valores[0] if valores else None


def rasgos_de_persona(token):
    if token is None:
        return None, None
    correccion = FORMAS_NO_RECONOCIDAS.get(token.text.casefold())
    if correccion and token.lemma_.casefold() == token.text.casefold():
        return correccion[1], correccion[2]
    return rasgo_morfologico(token, "Person"), rasgo_morfologico(token, "Number")


def interpretar_persona(token):
    rasgos = rasgos_de_persona(token)
    return None if None in rasgos else PERSONA_POR_RASGOS.get(rasgos)


def encontrar_verbo_principal(doc):
    comercial = next((t for t in doc if normalizar_accion(t) in VERBOS_EGRESO | VERBOS_INGRESO), None)
    if comercial is not None:
        return comercial
    raices = [t for t in doc if t.dep_ == "ROOT"]
    verbo = next((t for t in raices if t.pos_ in ("VERB", "AUX")), None)
    verbo = verbo or next((t for t in doc if t.pos_ == "VERB"), None)
    return verbo or next((t for t in doc if t.text.casefold() in FORMAS_NO_RECONOCIDAS | FORMAS_SIN_DIACRITICO), None)


def describir_verbo(token):
    if token is None:
        return None
    _, numero = rasgos_de_persona(token)
    return {"texto": token.text, "lema": token.lemma_, "pos": token.pos_,
            "persona": interpretar_persona(token), "numero": numero, "dependencia": token.dep_}


def pronombre_agente(token):
    if token is None or token.pos_ != "PRON":
        return None
    return PERSONA_POR_RASGOS.get((rasgo_morfologico(token, "Person"), rasgo_morfologico(token, "Number")))


def encontrar_agente(verbo):
    if verbo is None:
        return None
    if verbo.text.lower() in FORMAS_SIN_DIACRITICO:
        return None
    sujeto = next((h for h in verbo.children if h.dep_ in ("nsubj", "nsubj:pass")), None)
    if any(h.dep_ in ("expl:pass", "expl:impers") for h in verbo.children):
        return None
    if sujeto is not None:
        if sujeto.lemma_.lower() in {"dólar", "dolares", "dólares", "euro", "euros", "dinero", "plata", "importe", "cantidad"}:
            return None
        persona, _ = rasgos_de_persona(verbo)
        if sujeto.pos_ != "PRON" and persona in ("1", "2"):
            return interpretar_persona(verbo)
        return pronombre_agente(sujeto) or sujeto.text
    if verbo.text.lower() in FORMAS_NO_RECONOCIDAS:
        return interpretar_persona(verbo)
    return interpretar_persona(verbo)


def extraer_entidades(doc):
    return [{"texto": e.text, "etiqueta": e.label_, "inicio": e.start_char, "fin": e.end_char} for e in doc.ents]


def normalizar_accion(verbo):
    if verbo is None:
        return None
    forma = FORMAS_NO_RECONOCIDAS.get(verbo.text.casefold())
    lema = forma[0] if forma else FORMAS_SIN_DIACRITICO.get(verbo.text.casefold(), verbo.lemma_.casefold())
    lema = LEMAS_CORREGIDOS.get(lema, lema)
    return ACCIONES_CANONICAS.get(lema, lema)


def detectar_modalidad(doc):
    palabras = [t.text.casefold() for t in doc]
    accion = next((t for t in doc if normalizar_accion(t) in VERBOS_EGRESO | VERBOS_INGRESO), None)
    antes = [t.text.casefold() for t in doc if accion is not None and t.i < accion.i]
    contexto = " ".join(antes)
    if re.search(r"\b(?:tal vez|quizá|quizas|puede que|a lo mejor|creo que|supongo que|me parece que|es posible que|probablemente|posiblemente|si)\b", contexto):
        return "INCIERTA"
    if any(t.morph.get("Mood") == ["Cnd"] and (accion is None or t.i < accion.i) for t in doc):
        return "INCIERTA"
    if (
        accion is not None and accion.morph.get("Tense") == ["Fut"]
        and accion.text.casefold() not in FORMAS_NO_RECONOCIDAS
    ):
        return "INTENCION"
    if any(p in {"quiero", "quería", "quisiera", "deseo", "planeo"} for p in palabras):
        return "INTENCION"
    if any(p in {"no", "nunca", "jamas", "jamás"} for p in antes):
        return "NEGADA"
    return "AFIRMADA"


def modalidad_contextual(texto: str, inicio_operacion: int):
    """Aplica solo los marcadores de modalidad propios de la cláusula actual."""
    prefijo = texto[:inicio_operacion]
    verbos_previos = list(re.finditer(
        r"\b(?:compr(?:é|e|ó|o|aron|amos|ará|arán|aría)|vend(?:í|i|ió|io|ieron|emos|erá|erán)|"
        r"pag(?:ué|ue|ó|o|aron|amos|ará|arán)|abon(?:é|e|ó|o|aron|amos)|gast(?:é|e|ó|o|aron|amos))\b",
        prefijo, re.IGNORECASE,
    ))
    # Si existe una acción previa, toma solo el material que la conecta con la
    # acción actual. Así «creo que compré X y después pagué» no contagia la
    # incertidumbre de la primera cláusula, pero «y tampoco pagué» sí conserva
    # la negación explícita de la segunda.
    clausula = prefijo[verbos_previos[-1].end():] if verbos_previos else prefijo
    clausula = re.split(r";|\bpero\b|\bsin embargo\b|\baunque\b", clausula, flags=re.IGNORECASE)[-1]
    if re.search(r"\b(?:tal vez|quizá|quizas|puede que|a lo mejor|probablemente|posiblemente|si)\b", clausula, re.IGNORECASE):
        return "INCIERTA"
    if re.search(r"\b(?:no|nunca|jam[aá]s|tampoco)\b", clausula, re.IGNORECASE):
        return "NEGADA"
    return "AFIRMADA"


def detectar_accion(doc, verbo):
    accion_verbal = normalizar_accion(verbo)
    if accion_verbal in VERBOS_INGRESO | VERBOS_EGRESO:
        return accion_verbal
    if any(t.lemma_.lower() == "deber" or t.text.lower() in ("debiendo", "debió") for t in doc):
        return "deber"
    return accion_verbal


def dinero_me_llega(verbo):
    return verbo is not None and any(
        h.dep_ in ("iobj", "dative") and h.lemma_.lower() in ("yo", "nosotros") for h in verbo.children
    )


def clasificar_tipo(accion, verbo):
    if accion == "entregar" and dinero_me_llega(verbo):
        return "INGRESO"
    if accion in VERBOS_INGRESO - {"entregar"}:
        return "INGRESO"
    if accion in VERBOS_EGRESO:
        return "INGRESO" if accion in VERBOS_DIRECCIONALES and dinero_me_llega(verbo) else "EGRESO"
    return None


def _roles_desde_entidades(entidades):
    roles = {}
    for ent in entidades:
        etiqueta, texto = ent["etiqueta"], ent["texto"]
        if etiqueta == "QUANTITY": roles.setdefault("cantidad", texto)
        elif etiqueta == "PRICE": roles.setdefault("precio", texto)
        elif etiqueta == "CONCEPT": roles.setdefault("concepto", texto)
        elif etiqueta == "PRODUCT": roles.setdefault("producto", texto)
    return roles


def _entidades_comerciales(doc, entidades, accion):
    texto, roles = doc.text, _roles_desde_entidades(entidades)
    cantidad_detectada = roles.get("cantidad")
    patron_cantidad = rf"(?:{_NUMERO}|algunos|algunas|muchos|muchas)(?:\s+{_UNIDADES})?"
    if cantidad_detectada and not re.fullmatch(patron_cantidad, cantidad_detectada.strip(), re.IGNORECASE):
        cantidad_simple = re.fullmatch(rf"(?P<numero>{_NUMERO})\s+(?P<final>[\wáéíóúüñ]+)", cantidad_detectada, re.IGNORECASE)
        entidad_cantidad = next((e for e in entidades if e["etiqueta"] == "QUANTITY" and e["texto"] == cantidad_detectada), None)
        tokens_cantidad = [t for t in doc if entidad_cantidad and entidad_cantidad["inicio"] <= t.idx < entidad_cantidad["fin"]]
        if cantidad_simple and tokens_cantidad and tokens_cantidad[-1].pos_ in {"ADP", "CCONJ", "DET", "PRON", "PUNCT"}:
            roles["cantidad"] = cantidad_simple.group("numero")
            cantidad_detectada = roles["cantidad"]
        token_candidato = next((t for t in doc if t.text.casefold() == cantidad_detectada.casefold()), None)
        if cantidad_detectada and not re.fullmatch(patron_cantidad, cantidad_detectada.strip(), re.IGNORECASE) and (
            "producto" not in roles and accion in {"comprar", "vender"}
            and token_candidato is not None and token_candidato.pos_ in {"NOUN", "PROPN"}
        ):
            roles["producto"] = cantidad_detectada
        if not re.fullmatch(patron_cantidad, cantidad_detectada.strip(), re.IGNORECASE):
            roles.pop("cantidad", None)
    cantidad = roles.get("cantidad")
    cantidad_simple = re.fullmatch(rf"(?P<numero>{_NUMERO})\s+(?P<final>[\wáéíóúüñ]+)", cantidad or "", re.IGNORECASE)
    if cantidad_simple:
        entidad_cantidad = next((e for e in entidades if e["etiqueta"] == "QUANTITY" and e["texto"] == cantidad), None)
        tokens_cantidad = [t for t in doc if entidad_cantidad and entidad_cantidad["inicio"] <= t.idx < entidad_cantidad["fin"]]
        token_final = tokens_cantidad[-1] if tokens_cantidad else None
        if token_final is not None and token_final.pos_ in {"ADP", "CCONJ", "DET", "PRON", "PUNCT"}:
            # Correct an overlong NER quantity span while retaining the original entity in `entidades`.
            roles["cantidad"] = cantidad_simple.group("numero")
    if roles.get("precio") and not re.fullmatch(rf"{_NUMERO}(?:\s+{_MONEDAS})?", roles["precio"], re.IGNORECASE):
        roles.pop("precio", None)
    if (roles.get("cantidad") or "").casefold() in {"la", "las", "el", "los", "esa", "esta", "aquella", "dicha", "primera", "segunda", "tercera"}:
        roles.pop("cantidad", None)
    if (roles.get("cantidad") or "").casefold() in VERBOS_EGRESO | VERBOS_INGRESO | set(FORMAS_SIN_DIACRITICO) | set(FORMAS_NO_RECONOCIDAS):
        roles.pop("cantidad", None)
    if (roles.get("concepto") or "").casefold() in {"además", "también", "luego", "después", "pero", "importe", "precio", "cantidad"}:
        roles.pop("concepto", None)
    cuantificador_generico = re.search(
        r"\b(?P<cuantificador>varios|varias|algunos|algunas|muchos|muchas)\s+(?P<generico>productos|art[ií]culos|cosas)\b",
        texto, re.IGNORECASE,
    )
    generico_contexto = re.search(r"\b(?P<generico>algo|nada|alguna\s+cosa|algunas\s+cosas|varias\s+cosas|muchas\s+cosas)\b", texto, re.IGNORECASE)
    if cuantificador_generico and accion in {"comprar", "vender"}:
        roles["cantidad"] = cuantificador_generico.group("cuantificador")
        roles["producto"] = cuantificador_generico.group("generico")
    elif generico_contexto and accion in {"comprar", "vender"}:
        roles["producto"] = " ".join(generico_contexto.group("generico").split())
        if generico_contexto.group("generico").casefold() in {"algo", "nada", "alguna cosa"}:
            roles.pop("cantidad", None)
    elif (roles.get("producto") or "").casefold() in _CUANTIFICADORES_VAGOS:
        roles.pop("producto", None)
    if (roles.get("producto") or "").casefold() == "nada":
        roles.pop("producto", None)
    if (roles.get("producto") or "").casefold() in VERBOS_EGRESO | VERBOS_INGRESO | set(FORMAS_SIN_DIACRITICO) | set(FORMAS_NO_RECONOCIDAS):
        roles.pop("producto", None)
    if accion in {"pagar", "abonar"} and re.search(r"\b(?:de|para|a)\s+(?:(?:esa|esta|aquella|la|una|dicha)\s+)?(?:(?:primera|segunda|tercera)\s+)?(?:compra|adquisici[oó]n|factura|deuda)\s+de\s+[\wáéíóúüñ]+", texto, re.IGNORECASE):
        roles.pop("producto", None)
    if accion == "recibir" and "producto" not in roles:
        venta_producto = re.search(r"\bventa\s+de\s+(?P<producto>[\wáéíóúüñ]+)", texto, re.IGNORECASE)
        if venta_producto:
            roles["producto"] = venta_producto.group("producto")
    if re.search(r"\b(?:de|para)\s+(?:(?:esa|esta|aquella|la|dicha)\s+)?(?:(?:primera|segunda|tercera)\s+)?(?:compra|adquisici[oó]n|factura|deuda)\b", texto, re.IGNORECASE):
        if (roles.get("cantidad") or "").lower() in {"esa", "esta", "aquella", "dicha", "primera", "segunda", "tercera"}:
            roles.pop("cantidad", None)
        if (roles.get("concepto") or "").lower() in {"compra", "adquisición", "adquisicion", "factura", "deuda"}:
            roles.pop("concepto", None)
    importe_moneda = re.search(rf"\b(?P<importe>{_NUMERO})\s+(?P<moneda>{_MONEDAS_NUMERO})\b", texto, re.IGNORECASE)
    if importe_moneda:
        if roles.get("producto", "").lower() in {"dólar", "dólares", "dolar", "dolares", "euro", "euros", "usd", "eur"}:
            roles.pop("producto", None)
        if roles.get("cantidad") and roles["cantidad"].lower() == importe_moneda.group("importe").lower():
            roles.pop("cantidad", None)
    deuda = re.search(r"\b(?:deuda|saldo)\b", texto, re.IGNORECASE)
    if deuda:
        roles.pop("producto", None)
        roles.pop("cantidad", None)
        if "deuda" in (roles.get("precio") or "").casefold() or "saldo" in (roles.get("precio") or "").casefold():
            roles.pop("precio", None)
    patron = re.search(
        rf"(?P<cantidad>{_NUMERO}\s+{_UNIDADES})\s+de\s+(?P<producto>[\wáéíóúüñ]+(?:\s+(?!por|a|con|en|para)[\wáéíóúüñ]+)?)",
        texto, re.IGNORECASE,
    )
    if patron:
        roles["cantidad"], roles["producto"] = patron.group("cantidad"), patron.group("producto").strip()
    else:
        con_unidad = re.search(rf"(?P<cantidad>{_NUMERO})\s+(?P<unidad>{_UNIDADES})\b", texto, re.IGNORECASE)
        if con_unidad and not deuda:
            unidad = con_unidad.group("unidad").lower()
            if unidad in _UNIDADES_PRODUCTO:
                roles["cantidad"] = con_unidad.group("cantidad")
                roles["producto"] = unidad
            else:
                roles["cantidad"] = con_unidad.group(0)
                roles.pop("producto", None)
        elif "producto" not in roles and not deuda:
            simple = re.search(
                rf"(?P<cantidad>{_NUMERO})\s+(?!{_MONEDAS}\b)(?P<producto>(?!{_MONEDAS}\b)[\wáéíóúüñ]+)",
                texto, re.IGNORECASE,
            )
            if simple and simple.group("producto").lower() not in {"deuda", "saldo"}:
                candidato = simple.group("producto")
                token = next((t for t in doc if t.idx == simple.start("producto")), None)
                # A rule-created product must be a noun, proper noun or adjective.
                # Keep NER entities untouched; this filter only guards the fallback rule.
                if token is not None and token.pos_ in {"NOUN", "PROPN", "ADJ"}:
                    roles["cantidad"], roles["producto"] = simple.group("cantidad"), candidato
    importes = list(re.finditer(rf"(?P<importe>{_NUMERO}\s+{_MONEDAS})", texto, re.IGNORECASE))
    if importes:
        # Prefer a complete numeric-currency span over a malformed partial NER PRICE span.
        roles["precio"] = importes[-1].group("importe").strip()
    elif "precio" not in roles and not re.search(r"\bimporte\s+desconocido\b", texto, re.IGNORECASE):
            importe_suelto = re.search(rf"\b(?:por|a)\s+(?P<importe>{_NUMERO})\b", texto, re.IGNORECASE)
            if importe_suelto:
                roles["precio"] = importe_suelto.group("importe").strip()
            elif accion == "deber":
                importe_deuda = re.search(rf"\b(?:debiendo|debe|debió)\s+(?P<importe>{_NUMERO})\b", texto, re.IGNORECASE)
                if importe_deuda:
                    roles["precio"] = importe_deuda.group("importe").strip()
    if re.search(r"\bimporte\s+desconocido\b", texto, re.IGNORECASE):
        roles.pop("precio", None)
    if accion == "gastar" and "concepto" not in roles:
        concepto = re.search(r"\b(?:en|para)\s+(?:el|la|los|las)?\s*([\wáéíóúüñ]+(?:\s+[\wáéíóúüñ]+)?)", texto, re.IGNORECASE)
        if concepto:
            roles["concepto"] = concepto.group(1).strip()
    if accion in {"pagar", "abonar", "entregar"}:
        contraparte = re.search(r"\b(?:a|al)\s+(?:el\s+)?([\wáéíóúüñ]+)", texto, re.IGNORECASE)
        if contraparte:
            roles["contraparte"] = contraparte.group(1)
            roles.pop("concepto", None)
    return roles


def _moneda(precio):
    if not precio:
        return None
    moneda = re.search(_MONEDAS, precio, re.IGNORECASE)
    if not moneda:
        return None
    token = moneda.group(0).casefold()
    if token in {"usd", "dólar", "dólares", "dolar", "dolares", "$"}: return "USD"
    if token in {"euro", "euros", "eur"}: return "EUR"
    if token in {"peso", "pesos"}: return "PESO"
    return None


def validar_resultado(resultado: dict[str, Any]):
    faltantes, advertencias = [], []
    accion = resultado.get("accion")
    if accion is None: faltantes.append("accion")
    if accion in {"vender", "comprar"} and not resultado.get("producto"): faltantes.append("producto")
    if accion in VERBOS_INGRESO | VERBOS_EGRESO and not resultado.get("precio"):
        faltantes.append("precio")
    if resultado.get("agente") is None: advertencias.append("No se pudo determinar el agente.")
    if accion == "deber": advertencias.append("La deuda expresa una obligación; no demuestra un pago realizado.")
    texto = resultado.get("texto", "")
    if len(re.findall(r"\b(?:comprar|compró|compré|compramos|vender|vendió|vendí|pagar|pagó|pagué|cobrar|cobró|recibir|recibió)\b", texto, re.IGNORECASE)) > 1:
        advertencias.append("La frase contiene varias acciones; el esquema actual solo representa una operación.")
    if re.search(r"\bcada\s+(?:uno|una)\b", texto, re.IGNORECASE):
        advertencias.append("El importe parece unitario; el esquema actual no separa precio unitario de total.")
    if accion in {"pagar", "abonar"} and re.search(r"\bdeuda\b", texto, re.IGNORECASE):
        advertencias.append("Se identifica el pago, pero el saldo previo y el saldo restante no están representados.")
    if resultado.get("tipo") is None and accion not in (None, "deber"):
        advertencias.append("La clasificación económica no es concluyente.")
    if accion is None:
        estado = "NO_RECONOCIDO"
    elif advertencias or faltantes:
        estado = "AMBIGUO" if resultado.get("tipo") is None else "PARCIAL"
    else:
        estado = "VALIDO"
    resultado.update(faltantes=faltantes, advertencias=advertencias, estado=estado)
    if resultado.get("modalidad") == "NEGADA":
        resultado["estado"] = "PARCIAL"
        resultado["advertencias"].append("La acción está negada; no representa una operación positiva ejecutada.")
    if resultado.get("modalidad") == "INTENCION":
        resultado["estado"] = "PARCIAL"
        resultado["advertencias"].append("El texto expresa intención, no una operación ejecutada.")
    if resultado.get("modalidad") == "INCIERTA":
        resultado["estado"] = "AMBIGUO"
        resultado["advertencias"].append("El texto expresa incertidumbre sobre la operación.")
    return resultado


def interpretar_texto(
    texto: str, nlp=None, _offset_base: int = 0,
    _entidades_override=None, _doc_override=None,
):
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("texto debe ser una cadena no vacía")
    nlp = nlp if nlp is not None else cargar_modelo()
    if len(texto) > nlp.max_length:
        raise ValueError(f"texto excede el máximo del modelo ({nlp.max_length} caracteres)")
    doc = _doc_override if _doc_override is not None else nlp(texto)
    if doc.text != texto:
        raise ValueError("_doc_override debe corresponder exactamente al texto")
    verbo = None
    verbo = encontrar_verbo_principal(doc)
    detalle, accion = describir_verbo(verbo), detectar_accion(doc, verbo)
    modalidad = detectar_modalidad(doc)
    entidades, roles = (
        extraer_entidades(doc) if _entidades_override is None else _entidades_override,
        None,
    )
    roles = _entidades_comerciales(doc, entidades, accion)
    resultado = {
        "texto": texto, "agente": encontrar_agente(verbo), "contraparte": roles.pop("contraparte", None), "modalidad": modalidad,
        "accion": accion, "tipo": clasificar_tipo(accion, verbo), "cantidad": roles.get("cantidad"),
        "producto": roles.get("producto"), "concepto": roles.get("concepto"), "precio": roles.get("precio"),
        "moneda": _moneda(roles.get("precio")), "entidades": entidades,
        "verbo": detalle["texto"] if detalle else None, "lema": detalle["lema"] if detalle else None,
        "pos": detalle["pos"] if detalle else None, "persona": interpretar_persona(verbo),
        "numero": detalle["numero"] if detalle else None,
        "dependencias": [(t.text, t.dep_, t.head.text) for t in doc],
    }
    if _offset_base:
        resultado["_offset_base"] = _offset_base
    return validar_resultado(resultado)


def _proyectar_entidades_segmento(doc_completo, segmento, doc_segmento):
    """Recupera spans completos solo con ancla local y registra conflictos."""
    entidades_locales = extraer_entidades(doc_segmento)
    seleccionadas = list(doc_segmento.ents)
    diagnosticos = []
    for entidad in doc_completo.ents:
        inicio, fin = entidad.start_char, entidad.end_char
        contenido = segmento["inicio"] <= inicio and fin <= segmento["fin"]
        interseca = inicio < segmento["fin"] and segmento["inicio"] < fin
        registro = {
            "texto": entidad.text, "etiqueta": entidad.label_,
            "inicio": inicio, "fin": fin,
        }
        if not contenido:
            if interseca:
                diagnosticos.append({
                    "tipo": "span_cruza_limite_segmento", "span": registro,
                    "segmento": {"inicio": segmento["inicio"], "fin": segmento["fin"]},
                    "resolucion": "sin_proyeccion; se conserva la inferencia local",
                })
            continue
        span_local = doc_segmento.char_span(
            inicio - segmento["inicio"], fin - segmento["inicio"],
            label=entidad.label_, alignment_mode="strict",
        )
        if span_local is None:
            diagnosticos.append({
                "tipo": "span_no_alineable", "span": registro,
                "resolucion": "sin_proyeccion; se conserva la inferencia local",
            })
            continue
        if not entidades_locales:
            diagnosticos.append({
                "tipo": "proyeccion_sin_ancla_local", "span": registro,
                "resolucion": "proyeccion_omitida; el segmento no tiene otra entidad local",
            })
            continue
        solapados = [
            local for local in seleccionadas
            if max(local.start_char, span_local.start_char) < min(local.end_char, span_local.end_char)
        ]
        if solapados:
            if not any(
                local.start_char == span_local.start_char
                and local.end_char == span_local.end_char
                and local.label_ == span_local.label_
                for local in solapados
            ):
                diagnosticos.append({
                    "tipo": "solape_local_proyectado", "span": registro,
                    "locales": [{
                        "texto": local.text, "etiqueta": local.label_,
                        "inicio": local.start_char + segmento["inicio"],
                        "fin": local.end_char + segmento["inicio"],
                    } for local in solapados],
                    "resolucion": "proyeccion_omitida; se conserva la inferencia local",
                })
            continue
        seleccionadas.append(span_local)
        diagnosticos.append({
            "tipo": "proyeccion_con_ancla_local", "span": registro,
            "resolucion": "span_completo_agregado_sin_solape",
        })
    entidades_finales = [
        {"texto": span.text, "etiqueta": span.label_,
         "inicio": span.start_char, "fin": span.end_char}
        for span in sorted(seleccionadas, key=lambda item: (item.start_char, item.end_char))
    ]
    return entidades_finales, diagnosticos


def _interpretar_operaciones_detalle(texto: str, nlp=None):

    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("texto debe ser una cadena no vacía")
    nlp = nlp if nlp is not None else cargar_modelo()
    if len(texto) > nlp.max_length:
        raise ValueError(f"texto excede el máximo del modelo ({nlp.max_length} caracteres)")
    doc = nlp(texto)
    acciones = []
    for token in doc:
        if token.pos_ not in ("VERB", "AUX") and token.text.casefold() not in (FORMAS_NO_RECONOCIDAS.keys() | FORMAS_SIN_DIACRITICO.keys()):
            continue
        accion = normalizar_accion(token)
        if accion in VERBOS_INGRESO | VERBOS_EGRESO:
            acciones.append(token)
    if not acciones:
        es_deuda = any(
            token.lemma_.lower() == "deber" or token.text.lower() in {"debiendo", "debió"}
            for token in doc
        )
        if es_deuda:
            operacion = interpretar_texto(texto, nlp=nlp)
            return ([operacion] if operacion.get("modalidad") == "AFIRMADA" else []), []
        return [], []
    if len(acciones) <= 1:
        operacion = interpretar_texto(texto, nlp=nlp)
        return ([operacion] if operacion.get("modalidad") == "AFIRMADA" else []), []

    limites = [0] + [token.idx for token in acciones[1:]] + [len(texto)]
    segmentos, diagnosticos_ner = [], []
    for inicio, fin in zip(limites, limites[1:]):
        bruto = texto[inicio:fin]
        segmento = bruto.strip(" ,; y")
        if segmento:
            recorte = bruto.find(segmento)
            inicio_real = inicio + recorte
            fin_real = inicio_real + len(segmento)
            doc_segmento = nlp(segmento)
            entidades, diagnosticos = _proyectar_entidades_segmento(
                doc, {"inicio": inicio_real, "fin": fin_real}, doc_segmento,
            )
            diagnosticos_ner.extend(diagnosticos)
            operacion = interpretar_texto(
                segmento, nlp=nlp, _offset_base=inicio_real,
                _entidades_override=entidades, _doc_override=doc_segmento,
            )
            if operacion.get("modalidad") == "AFIRMADA":
                heredada = modalidad_contextual(texto, inicio_real)
                if heredada != "AFIRMADA":
                    operacion["modalidad"] = heredada
                    operacion["estado"] = "AMBIGUO"
            if operacion.get("modalidad") == "AFIRMADA":
                segmentos.append(operacion)
    # Elimina diagnósticos repetidos cuando un span toca más de un segmento.
    unicos = {}
    for diagnostico in diagnosticos_ner:
        clave = json.dumps(diagnostico, ensure_ascii=False, sort_keys=True)
        unicos.setdefault(clave, diagnostico)
    return segmentos, list(unicos.values())


def interpretar_operaciones(texto: str, nlp=None):
    """Interpreta cláusulas afirmadas y conserva diagnóstico de NER por separado."""
    operaciones, _ = _interpretar_operaciones_detalle(texto, nlp=nlp)
    return operaciones


def interpretar_operaciones_con_relaciones(texto: str, nlp=None):
    if not isinstance(texto, str):
        return {
            "texto": None, "estado": "NO_RECONOCIDO", "operaciones": [], "relaciones": [],
            "diagnosticos_ner": [], "advertencias": ["La entrada debe ser una cadena de texto."], "ambiguedades": [],
        }
    if not texto.strip():
        return {
            "texto": texto if isinstance(texto, str) else None,
            "estado": "NO_RECONOCIDO", "operaciones": [], "relaciones": [],
            "diagnosticos_ner": [],
            "advertencias": ["No hay texto suficiente para interpretar."],
            "ambiguedades": [],
        }
    operaciones, diagnosticos_ner = _interpretar_operaciones_detalle(texto, nlp=nlp)
    etiquetas_campo = {
        "cantidad": {"QUANTITY"}, "producto": {"PRODUCT"},
        "concepto": {"CONCEPT"}, "precio": {"PRICE"},
        "agente": {"PER", "PERSON"},
    }
    for indice, operacion in enumerate(operaciones):
        operacion["id"] = f"op_{indice + 1}"
        campos = {}
        for campo in ("agente", "accion", "tipo", "cantidad", "producto", "concepto", "precio"):
            valor = operacion.get(campo)
            evidencia = None
            origen = None
            estado_campo = "AUSENTE"
            if valor is not None:
                estado_campo = "IDENTIFICADO"
                objetivo = operacion.get("verbo") if campo in {"accion", "agente", "tipo"} else str(valor)
                match = re.search(re.escape(objetivo), operacion.get("texto", ""), re.IGNORECASE) if objetivo else None
                if match:
                    base = operacion.get("_offset_base", 0)
                    evidencia = {"texto": match.group(0), "inicio": base + match.start(), "fin": base + match.end()}
                    estado_ner = any(
                        entidad["texto"].casefold() == match.group(0).casefold()
                        and entidad["etiqueta"] in etiquetas_campo.get(campo, set())
                        for entidad in operacion.get("entidades", [])
                    )
                    origen = "NER" if estado_ner else ("MORFOLOGIA" if campo == "agente" else "REGLA")
                elif campo in {"agente", "tipo"}:
                    origen = "MORFOLOGIA" if campo == "agente" else "REGLA"
                    verbo_match = re.search(re.escape(operacion.get("verbo", "")), operacion.get("texto", ""), re.IGNORECASE) if operacion.get("verbo") else None
                    base = operacion.get("_offset_base", 0)
                    evidencia = {
                        "texto": operacion.get("verbo"),
                        "inicio": base + verbo_match.start() if verbo_match else None,
                        "fin": base + verbo_match.end() if verbo_match else None,
                    }
                    estado_campo = "INFERIDO"
            elif campo in etiquetas_campo:
                entidades_candidatas = [
                    entidad for entidad in operacion.get("entidades", [])
                    if entidad["etiqueta"] in etiquetas_campo[campo]
                    and entidad["texto"].casefold() not in {"esa", "este", "esta", "dicha", "primera", "segunda", "tercera"}
                    and not (
                        campo == "concepto"
                        and entidad["texto"].casefold() in {"compra", "adquisición", "adquisicion", "factura", "deuda"}
                        and re.search(r"\b(?:de|para|a)\s+(?:(?:esa|esta|aquella|la|una|dicha)\s+)?(?:(?:primera|segunda|tercera)\s+)?(?:compra|adquisici[oó]n|factura|deuda)\b", operacion.get("texto", ""), re.IGNORECASE)
                    )
                ]
                if entidades_candidatas:
                    estado_campo = "NO_INTERPRETADO"
                    evidencia = {
                        "texto": entidades_candidatas[0]["texto"],
                        "inicio": operacion.get("_offset_base", 0) + entidades_candidatas[0]["inicio"],
                        "fin": operacion.get("_offset_base", 0) + entidades_candidatas[0]["fin"],
                    }
                    origen = "NER"
            if campo == "tipo" and valor is not None:
                estado_campo = "INFERIDO"
            if campo == "agente" and valor in set(PERSONA_POR_RASGOS.values()):
                estado_campo = "INFERIDO"
            campos[campo] = {"valor": valor, "estado": estado_campo, "origen": origen, "evidencia": evidencia}
            if campo == "producto" and valor is not None:
                normalizado = " ".join(valor.casefold().split())
                campos[campo]["interpretacion"] = "GENERICA" if normalizado in _PRODUCTOS_GENERICOS | _REFERENCIAS_GENERICAS | _CUANTIFICADORES_VAGOS else "TEXTO_ESPECIFICO"
            if campo == "precio":
                texto_op = operacion.get("texto", "")
                if valor is not None:
                    if re.search(r"\b(?:cada\s+(?:uno|una)|por\s+(?:caja|unidad|kilo|libra|litro))\b", texto_op, re.IGNORECASE):
                        campos[campo]["ambito"] = "UNITARIO"
                    elif re.search(r"\btotal\b", texto_op, re.IGNORECASE):
                        campos[campo]["ambito"] = "TOTAL"
                    elif operacion.get("cantidad"):
                        campos[campo]["ambito"] = "DESCONOCIDO"
                        campos[campo]["estado"] = "AMBIGUO"
                    else:
                        campos[campo]["ambito"] = "NO_ESPECIFICADO"
                else:
                    campos[campo]["ambito"] = "NO_ESPECIFICADO"
            if campo == "cantidad" and isinstance(valor, str) and valor.casefold() in _CUANTIFICADORES_VAGOS:
                campos[campo]["estado"] = "AMBIGUO"
        operacion["campos"] = campos
        if any(meta["estado"] == "AMBIGUO" for meta in campos.values()):
            operacion["estado"] = "AMBIGUO"
        elif campos["producto"].get("valor") is not None and campos["producto"].get("interpretacion") == "GENERICA":
            operacion["estado"] = "PARCIAL"
            if "producto" not in operacion["faltantes"]:
                operacion["faltantes"].append("producto")
            operacion["advertencias"].append("La referencia al producto es genérica y no identifica un artículo concreto.")
        if operacion.get("modalidad") == "NEGADA":
            operacion["estado"] = "PARCIAL"
        elif operacion.get("modalidad") == "INTENCION":
            operacion["estado"] = "PARCIAL"
        elif operacion.get("modalidad") == "INCIERTA":
            operacion["estado"] = "AMBIGUO"
        operacion.pop("_offset_base", None)
    relaciones = []
    referencias = re.compile(
        r"\b(?:de|para|a)\s+(?:(?:esa|esta|aquella|la|una|dicha)\s+)?"
        r"(?:(?P<ordinal>primera|segunda|tercera)\s+)?"
        r"(?P<entidad>compra|adquisici[oó]n|factura|deuda)"
        r"(?:\s+(?:anterior|previa))?"
        r"(?:\s+de\s+(?P<producto>[\wáéíóúüñ]+))?",
        re.IGNORECASE,
    )
    for indice_origen, operacion in enumerate(operaciones):
        if operacion.get("accion") not in {"pagar", "abonar"} or operacion.get("modalidad") != "AFIRMADA":
            continue
        texto_operacion = operacion.get("texto", "")
        referencia = referencias.search(texto_operacion)
        deuda_explicita = re.search(r"\b(?:una|la|esa|aquella|dicha)\s+deuda(?:\s+(?:anterior|previa))?\b", texto_operacion, re.IGNORECASE)
        if not referencia and not deuda_explicita:
            continue
        if deuda_explicita and not referencia:
            relaciones.append({
                "operacion_origen": indice_origen,
                "operacion_destino": None,
                "operacion_origen_id": operacion["id"],
                "operacion_destino_id": None,
                "tipo": "ABONO_DE" if operacion["accion"] == "abonar" else "PAGO_DE",
                "evidencia": deuda_explicita.group(0),
                "estado": "AMBIGUA",
            })
            continue
        candidatos = [
            indice for indice, previa in enumerate(operaciones[:indice_origen])
            if previa.get("accion") == "comprar" and previa.get("modalidad") == "AFIRMADA"
        ]
        destino = None
        estado = "AMBIGUA"
        ordinal = referencia.group("ordinal")
        producto = referencia.group("producto")
        if producto:
            coincidencias = [
                indice for indice in candidatos
                if (operaciones[indice].get("producto") or "").casefold() == producto.casefold()
            ]
            if len(coincidencias) == 1:
                destino, estado = coincidencias[0], "CONFIRMADA"
        elif ordinal:
            posicion = {"primera": 0, "segunda": 1, "tercera": 2}[ordinal.lower()]
            if len(candidatos) > posicion:
                destino, estado = candidatos[posicion], "CONFIRMADA"
        elif referencia.group("entidad").lower() in {"compra", "adquisición"} and re.search(r"\banterior\b", referencia.group(0), re.IGNORECASE) and candidatos:
            # “La compra anterior” identifies the immediately preceding purchase,
            # using explicit ordering language rather than mere proximity.
            destino, estado = candidatos[-1], "CONFIRMADA"
        elif len(candidatos) == 1 and referencia.group("entidad").lower() in {"compra", "adquisición"}:
            destino, estado = candidatos[0], "CONFIRMADA"
        relaciones.append({
            "operacion_origen": indice_origen,
            "operacion_destino": destino,
            "operacion_origen_id": operaciones[indice_origen]["id"],
            "operacion_destino_id": operaciones[destino]["id"] if destino is not None else None,
            "tipo": "ABONO_DE" if operacion["accion"] == "abonar" else "PAGO_DE",
            "evidencia": referencia.group(0),
            "estado": estado,
        })
    ambiguedades = []
    advertencias = []
    for operacion in operaciones:
        advertencias.extend(operacion.get("advertencias", []))
        cantidad_meta = operacion.get("campos", {}).get("cantidad", {})
        if cantidad_meta.get("estado") == "AMBIGUO":
            ambiguedades.append({
                "operacion_id": operacion["id"], "campo": "cantidad",
                "evidencia": cantidad_meta.get("evidencia", {}).get("texto"),
                "motivo": "La cantidad está expresada de forma vaga y no determina un valor numérico.",
            })
        producto_meta = operacion.get("campos", {}).get("producto", {})
        if producto_meta.get("interpretacion") == "GENERICA":
            advertencias.append("La referencia al producto es genérica y no identifica un artículo concreto.")
        precio_meta = operacion.get("campos", {}).get("precio", {})
        if precio_meta.get("ambito") == "DESCONOCIDO":
            ambiguedades.append({
                "operacion_id": operacion["id"], "campo": "precio",
                "evidencia": operacion.get("precio"),
                "motivo": "El importe está reconocido, pero el texto no especifica si es unitario o total.",
            })
        if re.search(r"\bcada\s+(?:uno|una)\b", operacion.get("texto", ""), re.IGNORECASE):
            ambiguedades.append({
                "operacion_id": operacion["id"], "campo": "precio",
                "evidencia": "cada " + re.search(r"\bcada\s+(?:uno|una)\b", operacion["texto"], re.IGNORECASE).group(0).split()[-1],
                "motivo": "El importe puede ser unitario; no se puede obtener el total con seguridad.",
            })
    for relacion in relaciones:
        if relacion["estado"] == "AMBIGUA":
            ambiguedades.append({
                "operacion_id": relacion["operacion_origen_id"], "campo": "relacion",
                "evidencia": relacion["evidencia"],
                "motivo": "La referencia textual no identifica una única operación destino.",
            })
    for relacion in relaciones:
        if relacion["estado"] == "AMBIGUA":
            operaciones[relacion["operacion_origen"]]["estado"] = "AMBIGUO"
    for operacion in operaciones:
        if operacion.get("modalidad") == "INCIERTA":
            ambiguedades.append({
                "operacion_id": operacion["id"], "campo": "modalidad",
                "evidencia": operacion.get("texto"),
                "motivo": "La frase no describe una operación afirmada y ejecutada.",
            })
    if ambiguedades:
        estado_general = "AMBIGUO"
    elif any(op.get("estado") == "PARCIAL" for op in operaciones):
        estado_general = "PARCIAL"
    elif not operaciones:
        estado_general = "NO_RECONOCIDO"
    else:
        estado_general = "VALIDO"
    if any(rel["estado"] == "AMBIGUA" for rel in relaciones):
        advertencias.append("Hay referencias entre operaciones cuyo destino permanece ambiguo.")
    if not operaciones and texto.strip():
        advertencias.append("No se reconoció una acción comercial que pueda convertirse en operación.")
    return {
        "texto": texto,
        "estado": estado_general,
        "operaciones": operaciones,
        "relaciones": relaciones,
        "diagnosticos_ner": diagnosticos_ner,
        "advertencias": list(dict.fromkeys(advertencias)),
        "ambiguedades": ambiguedades,
    }


def resultado_semantico(resultado):
    return {k: resultado[k] for k in CLAVES_SEMANTICAS if k in resultado}


def serializar_resultado(resultado) -> str:
    return json.dumps(resultado, ensure_ascii=False, sort_keys=True)


def mostrar_analisis(resultado):
    print("Texto:", resultado["texto"])
    print("Resultado:", resultado_semantico(resultado))
    print("Entidades:", [(e["texto"], e["etiqueta"]) for e in resultado["entidades"]])
    print("-" * 10)


def mostrar_bateria_de_pruebas(nlp=None):
    for texto in ["vendí 5 camisetas por 30 dólares", "compré cables por 12", "Gasté 5 dólares en transporte", "Pagué 10 dólares al proveedor", "Carlos quedó debiendo 10 dólares"]:
        mostrar_analisis(interpretar_texto(texto, nlp=nlp))


if __name__ == "__main__":
    mostrar_bateria_de_pruebas()
