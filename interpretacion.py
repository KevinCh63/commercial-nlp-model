from __future__ import annotations

import re
import json
from typing import Any

import spacy

MODELO_ENTRENADO = "custom_ner_model"
MODELO_BASE = "es_core_news_md"
VERBOS_EGRESO = {"gastar", "comprar", "adquirir", "pagar", "abonar", "desembolsar"}
VERBOS_INGRESO = {"vender", "cobrar", "recibir", "ingresar", "entregar"}
VERBOS_DIRECCIONALES = {"pagar", "abonar"}
ACCIONES_CANONICAS = {"adquirir": "comprar", "desembolsar": "gastar", "entregar": "recibir"}
LEMAS_CORREGIDOS = {"vendar": "vender"}
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
}
CLAVES_SEMANTICAS = (
    "agente", "contraparte", "accion", "tipo", "cantidad", "producto", "concepto",
    "precio", "moneda", "estado", "faltantes", "advertencias",
)
_nlp = None
_NUMERO = r"(?:\d+(?:[.,]\d+)?|un(?:a|o)?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|varios|varias)"
_MONEDAS = r"(?:d[oó]lares?|usd|euros?|eur|\$)"


def cargar_modelo(ruta: str = MODELO_ENTRENADO):
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        _nlp = spacy.load(ruta)
    except OSError:
        print(f"[aviso] No existe {ruta!r}; se usa {MODELO_BASE!r}.")
        _nlp = spacy.load(MODELO_BASE)
    return _nlp


def rasgo_morfologico(token, nombre: str):
    if token is None:
        return None
    valores = token.morph.get(nombre)
    return valores[0] if valores else None


def rasgos_de_persona(token):
    if token is None:
        return None, None
    correccion = FORMAS_NO_RECONOCIDAS.get(token.text.lower())
    if correccion and token.lemma_.lower() == token.text.lower():
        return correccion[1], correccion[2]
    return rasgo_morfologico(token, "Person"), rasgo_morfologico(token, "Number")


def interpretar_persona(token):
    rasgos = rasgos_de_persona(token)
    return None if None in rasgos else PERSONA_POR_RASGOS.get(rasgos)


def encontrar_verbo_principal(doc):
    raices = [t for t in doc if t.dep_ == "ROOT"]
    verbo = next((t for t in raices if t.pos_ in ("VERB", "AUX")), None)
    verbo = verbo or next((t for t in doc if t.pos_ == "VERB"), None)
    return verbo or next((t for t in doc if t.text.lower() in FORMAS_NO_RECONOCIDAS), None)


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
    sujeto = next((h for h in verbo.children if h.dep_ in ("nsubj", "nsubj:pass")), None)
    if any(h.dep_ in ("expl:pass", "expl:impers") for h in verbo.children):
        return None
    if sujeto is not None:
        persona, _ = rasgos_de_persona(verbo)
        if sujeto.pos_ != "PRON" and persona in ("1", "2"):
            return interpretar_persona(verbo)
        return pronombre_agente(sujeto) or sujeto.text
    return interpretar_persona(verbo)


def extraer_entidades(doc):
    return [{"texto": e.text, "etiqueta": e.label_, "inicio": e.start_char, "fin": e.end_char} for e in doc.ents]


def normalizar_accion(verbo):
    if verbo is None:
        return None
    forma = FORMAS_NO_RECONOCIDAS.get(verbo.text.lower())
    lema = forma[0] if forma else verbo.lemma_.lower()
    lema = LEMAS_CORREGIDOS.get(lema, lema)
    return ACCIONES_CANONICAS.get(lema, lema)


def detectar_accion(doc, verbo):
    if any(t.lemma_.lower() == "deber" or t.text.lower() in ("debiendo", "debió") for t in doc):
        return "deber"
    return normalizar_accion(verbo)


def dinero_me_llega(verbo):
    return verbo is not None and any(
        h.dep_ in ("iobj", "dative") and h.lemma_.lower() in ("yo", "nosotros") for h in verbo.children
    )


def clasificar_tipo(accion, verbo):
    if accion in VERBOS_INGRESO:
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
    """Usa NER como candidato y reglas de estructura para corregir sus roles."""
    texto, roles = doc.text, _roles_desde_entidades(entidades)
    patron = re.search(
        rf"(?P<cantidad>{_NUMERO}\s+(?:botellas?|cajas?|paquetes?|bolsas?|latas?|vasos?|unidades?))\s+de\s+(?P<producto>[\wáéíóúüñ]+(?:\s+(?!por|a|con|en|para)[\wáéíóúüñ]+)?)",
        texto, re.IGNORECASE,
    )
    if patron:
        roles["cantidad"], roles["producto"] = patron.group("cantidad"), patron.group("producto").strip()
    else:
        simple = re.search(
            rf"(?P<cantidad>{_NUMERO})\s+(?!{_MONEDAS}\b)(?P<producto>(?!{_MONEDAS}\b)[\wáéíóúüñ]+)",
            texto, re.IGNORECASE,
        )
        if simple and "producto" not in roles:
            roles["cantidad"], roles["producto"] = simple.group("cantidad"), simple.group("producto")
    if "precio" not in roles:
        importes = list(re.finditer(rf"(?P<importe>{_NUMERO}\s+{_MONEDAS})", texto, re.IGNORECASE))
        if importes:
            roles["precio"] = importes[-1].group("importe").strip()
        else:
            importe_suelto = re.search(rf"\b(?:por|a)\s+(?P<importe>{_NUMERO})\b", texto, re.IGNORECASE)
            if importe_suelto:
                roles["precio"] = importe_suelto.group("importe").strip()
            elif accion == "deber":
                importe_deuda = re.search(rf"\b(?:debiendo|debe|debió)\s+(?P<importe>{_NUMERO})\b", texto, re.IGNORECASE)
                if importe_deuda:
                    roles["precio"] = importe_deuda.group("importe").strip()
    if accion == "gastar" and "concepto" not in roles:
        concepto = re.search(r"\b(?:en|para)\s+(?:el|la|los|las)?\s*([\wáéíóúüñ]+(?:\s+[\wáéíóúüñ]+)?)", texto, re.IGNORECASE)
        if concepto:
            roles["concepto"] = concepto.group(1).strip()
    if accion in {"pagar", "abonar"}:
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
    return "USD" if moneda.group(0).lower() in {"usd", "dólar", "dólares", "$"} else "EUR"


def validar_resultado(resultado: dict[str, Any]):
    faltantes, advertencias = [], []
    accion = resultado.get("accion")
    if accion is None: faltantes.append("accion")
    if accion in {"vender", "comprar"} and not resultado.get("producto"): faltantes.append("producto")
    if resultado.get("agente") is None: advertencias.append("No se pudo determinar el agente.")
    if accion == "deber": advertencias.append("La deuda expresa una obligación; no demuestra un pago realizado.")
    if resultado.get("tipo") is None and accion not in (None, "deber"):
        advertencias.append("La clasificación económica no es concluyente.")
    if accion is None:
        estado = "NO_RECONOCIDO"
    elif advertencias or faltantes:
        estado = "AMBIGUO" if resultado.get("tipo") is None else "PARCIAL"
    else:
        estado = "VALIDO"
    resultado.update(faltantes=faltantes, advertencias=advertencias, estado=estado)
    return resultado


def interpretar_texto(texto: str, nlp=None):
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("texto debe ser una cadena no vacía")
    nlp = nlp if nlp is not None else cargar_modelo()
    doc, verbo = nlp(texto), None
    verbo = encontrar_verbo_principal(doc)
    detalle, accion = describir_verbo(verbo), detectar_accion(doc, verbo)
    entidades, roles = extraer_entidades(doc), None
    roles = _entidades_comerciales(doc, entidades, accion)
    resultado = {
        "texto": texto, "agente": encontrar_agente(verbo), "contraparte": roles.pop("contraparte", None),
        "accion": accion, "tipo": clasificar_tipo(accion, verbo), "cantidad": roles.get("cantidad"),
        "producto": roles.get("producto"), "concepto": roles.get("concepto"), "precio": roles.get("precio"),
        "moneda": _moneda(roles.get("precio")), "entidades": entidades,
        "verbo": detalle["texto"] if detalle else None, "lema": detalle["lema"] if detalle else None,
        "pos": detalle["pos"] if detalle else None, "persona": interpretar_persona(verbo),
        "numero": detalle["numero"] if detalle else None,
        "dependencias": [(t.text, t.dep_, t.head.text) for t in doc],
    }
    return validar_resultado(resultado)


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
