import spacy

MODELO_ENTRENADO = "custom_ner_model"
MODELO_BASE = "es_core_news_md"


VERBOS_EGRESO = {"gastar", "comprar", "pagar", "abonar"}
VERBOS_INGRESO = {"vender", "cobrar", "recibir", "ingresar"}

# Person + Number del verbo -> quién realiza la acción.
PERSONA_POR_RASGOS = {
    ("1", "Sing"): "YO",
    ("1", "Plur"): "NOSOTROS",
    ("2", "Sing"): "TU",
    ("2", "Plur"): "USTEDES",
    ("3", "Sing"): "TERCERA_PERSONA",
    ("3", "Plur"): "TERCERA_PERSONA_PLURAL",
}


FORMAS_NO_RECONOCIDAS = {
    "gasté": {"lema": "gastar", "persona": "1", "numero": "Sing"},
    "vendimos": {"lema": "vender", "persona": "1", "numero": "Plur"},
    "hablé": {"lema": "hablar", "persona": "1", "numero": "Sing"},
    "caminé": {"lema": "caminar", "persona": "1", "numero": "Sing"},
    "viajé": {"lema": "viajar", "persona": "1", "numero": "Sing"},
}

LEMAS_CORREGIDOS = {
    "vendar": "vender"
}

CLAVES_SEMANTICAS = (
    "agente", "accion", "tipo", "cantidad", "producto", "concepto", "precio",
)

_nlp = None


def cargar_modelo(ruta=MODELO_ENTRENADO):
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        _nlp = spacy.load(ruta)
    except OSError:
        print(f"[aviso] No existe {ruta!r}. Ejecuta `python entrenamiento.py` primero. "
              f"Uso {MODELO_BASE!r}.")
        _nlp = spacy.load(MODELO_BASE)
    return _nlp


def rasgo_morfologico(token, nombre):

    if token is None:
        return None
    valores = token.morph.get(nombre)
    return valores[0] if valores else None


def forma_no_reconocida(token):

    return token is not None and token.lemma_ == token.text


def rasgos_de_persona(token):

    if token is None:
        return None, None
    persona = rasgo_morfologico(token, "Person")
    numero = rasgo_morfologico(token, "Number")
    if forma_no_reconocida(token):
        correccion = FORMAS_NO_RECONOCIDAS.get(token.text.lower())
        if correccion is not None:
            return correccion["persona"], correccion["numero"]
    return persona, numero


def interpretar_persona(token):

    rasgos = rasgos_de_persona(token)
    if None in rasgos:
        return None
    return PERSONA_POR_RASGOS.get(rasgos)


DEPENDENCIAS_DE_OBJETO = {"obj", "obl", "iobj", "dative", "attr", "ccomp", "xcomp"}


def encontrar_verbo_principal(doc):

    raices = [t for t in doc if t.dep_ == "ROOT"]
    candidatos = [t for t in raices if t.pos_ in ("VERB", "AUX")]
    if candidatos:
        return candidatos[0]
    for raiz in raices:
        if any(hijo.dep_ in DEPENDENCIAS_DE_OBJETO for hijo in raiz.children):
            return raiz
    return next((t for t in doc if t.pos_ == "VERB"), None)


def describir_verbo(token):
    if token is None:
        return None
    _, numero = rasgos_de_persona(token)
    return {
        "texto": token.text,
        "lema": token.lemma_,
        "pos": token.pos_,
        "persona": interpretar_persona(token),
        "numero": numero,
        "dependencia": token.dep_,
    }


def pronombre_agente(token):

    if token is None or token.pos_ != "PRON":
        return None
    persona = rasgo_morfologico(token, "Person")
    numero = rasgo_morfologico(token, "Number")
    if persona is None or numero is None:
        return None
    return PERSONA_POR_RASGOS.get((persona, numero))


def encontrar_agente(verbo):

    if verbo is None:
        return None
    sujeto = next(
        (h for h in verbo.children if h.dep_ in ("nsubj", "nsubj:pass")),
        None,
    )
    if sujeto is not None:
        persona, _ = rasgos_de_persona(verbo)
        # En español un verbo conjugado en 1ª/2ª persona no admite sujeto
        # nominal: si aparece, el parser se equivocó y confiamos en la morfología.
        if sujeto.pos_ != "PRON" and persona in ("1", "2"):
            return interpretar_persona(verbo)
        return pronombre_agente(sujeto) or sujeto.text
    return interpretar_persona(verbo)


def extraer_entidades(doc):

    return [
        {"texto": ent.text, "etiqueta": ent.label_,
         "inicio": ent.start_char, "fin": ent.end_char}
        for ent in doc.ents
    ]


def normalizar_accion(verbo):

    if verbo is None:
        return None
    if forma_no_reconocida(verbo):
        correccion = FORMAS_NO_RECONOCIDAS.get(verbo.text.lower())
        if correccion is not None:
            return correccion["lema"]
    return LEMAS_CORREGIDOS.get(verbo.lemma_, verbo.lemma_)



VERBOS_DIRECCIONALES = {"pagar", "abonar"}


def dinero_me_llega(verbo):

    if verbo is None:
        return False
    for hijo in verbo.children:
        if hijo.dep_ in ("iobj", "dative") and hijo.lemma_ == "yo":
            return True
    return False


def clasificar_tipo(accion, verbo):

    if accion is None:
        return None
    if accion in VERBOS_INGRESO:
        return "INGRESO"
    if accion in VERBOS_EGRESO:
        if accion in VERBOS_DIRECCIONALES and dinero_me_llega(verbo):
            return "INGRESO"
        return "EGRESO"
    return None


def _roles_desde_entidades(entidades):

    roles = {}
    for ent in entidades:
        etiqueta, texto = ent["etiqueta"], ent["texto"]
        if etiqueta == "QUANTITY":
            roles.setdefault("cantidad", texto)
        elif etiqueta == "PRICE":
            roles.setdefault("precio", texto)
        elif etiqueta == "CONCEPT":
            roles.setdefault("concepto", texto)
        elif etiqueta == "PRODUCT":
            roles.setdefault("producto", texto)
    return roles


def interpretar_texto(texto, nlp=None):

    nlp = nlp if nlp is not None else cargar_modelo()
    doc = nlp(texto)

    verbo = encontrar_verbo_principal(doc)
    detalle = describir_verbo(verbo)
    accion = normalizar_accion(verbo)
    tipo = clasificar_tipo(accion, verbo)
    entidades = extraer_entidades(doc)

    resultado = {
        "texto": texto,
        "agente": encontrar_agente(verbo),
        "accion": accion,
        "tipo": tipo,
        "entidades": entidades,
        "verbo": detalle["texto"] if detalle else None,
        "lema": detalle["lema"] if detalle else None,
        "pos": detalle["pos"] if detalle else None,
        "persona": interpretar_persona(verbo),
        "numero": detalle["numero"] if detalle else None,
        "dependencias": [(t.text, t.dep_, t.head.text) for t in doc],
    }
    resultado.update(_roles_desde_entidades(entidades))
    return resultado


def resultado_semantico(resultado):

    return {k: resultado[k] for k in CLAVES_SEMANTICAS if k in resultado}


def mostrar_analisis(resultado):
    deps = " ; ".join(
        f"{texto} --{dep}--> {cabeza}"
        for texto, dep, cabeza in resultado["dependencias"]
    ) or "(sin dependencias)"
    entidades = [(e["texto"], e["etiqueta"]) for e in resultado["entidades"]]

    print("Texto:", resultado["texto"])
    print("Verbo:", resultado["verbo"])
    print("Lema:", resultado["lema"])
    print("Persona:", resultado["persona"])
    print("Agente:", resultado["agente"])
    print("Entidades:", entidades)
    print("Dependencias:", deps)
    print("Resultado:", resultado_semantico(resultado))
    print("-" * 10)


def mostrar_bateria_de_pruebas(nlp=None):
    textos = [
        "vendí 5 camisetas por 30 dólares",
        "compré 5 camisetas por 40 dólares",
        "Juan me pagó 20 dólares",
        "gasté 30 dólares en transporte",
        "gastamos 20 dólares en materiales",
        "vendimos 10 productos por 50 dólares",
        "vendí 3 libras de tomate a 2 dólares",
        "María compró un producto por 10 dólares",
        "vendí 2 camisetas por 15 USD",
        "Carlos me abonó 5",
        "Carlos quedó debiendo 10",
        "El Mateo vendió 5 latas de sardina de 2.50$"
    ]
    print("\n" + "=" * 78)
    print("BATERÍA DE INTERPRETACIÓN")
    print("=" * 78)
    for texto in textos:
        mostrar_analisis(interpretar_texto(texto, nlp=nlp))


if __name__ == "__main__":
    mostrar_bateria_de_pruebas()
