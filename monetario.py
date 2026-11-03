"""Lectura monetaria conservadora: nunca usa float ni extrae subimportes."""
from decimal import Decimal, InvalidOperation
import re

NUMERO = r"[+−-]?\d+(?:[.,]+\d+|[ \u00a0]+\d+)*"
BASE_PALABRAS = r"(?:un(?:o|a)?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|veinte|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|cien|ciento|mil)"
PALABRAS = rf"{BASE_PALABRAS}(?:\s+(?:y\s+)?{BASE_PALABRAS})*"
MONEDA = r"(?:US\$|USD|EUR|\$|€|d[oó]lar(?:es)?|euros?|pesos?|soles?|GBP|libras\s+esterlinas)"
PATRON = re.compile(
    rf"(?<![\w.,+−-])(?:{MONEDA}\s*{NUMERO}|{NUMERO}\s*{MONEDA}|{PALABRAS}\s+{MONEDA})(?!\w|[.,]\d)",
    re.IGNORECASE,
)
VALORES = dict(zip(
    ['un', 'uno', 'una', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez', 'once', 'doce', 'trece', 'catorce', 'quince', 'veinte', 'treinta', 'cuarenta', 'cincuenta', 'sesenta', 'setenta', 'ochenta', 'noventa'],
    [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50, 60, 70, 80, 90],
))


def moneda_de(texto):
    match = re.search(MONEDA, texto or '', re.IGNORECASE)
    if not match:
        return None
    moneda = match.group().casefold()
    if moneda in {'usd', 'us$', '$', 'dólar', 'dólares', 'dolar', 'dolares'}:
        return 'USD'
    if moneda in {'eur', '€', 'euro', 'euros'}:
        return 'EUR'
    if moneda in {'peso', 'pesos'}:
        return 'PESO'
    return None


def normalizar_importe(texto):
    """Devuelve valores JSON: decimal exacto como cadena y original intacto."""
    salida = {'original': texto, 'valor': None, 'moneda': moneda_de(texto),
              'estado': 'AUSENTE' if not texto else 'IDENTIFICADO', 'motivos': []}
    if not texto:
        return salida
    if not (PATRON.fullmatch(texto) or re.fullmatch(NUMERO, texto) or re.fullmatch(PALABRAS, texto, re.IGNORECASE)):
        salida['estado'] = 'AMBIGUO'
        salida['motivos'] = ['FORMATO_MONETARIO_INVALIDO']
        return salida
    numero = re.search(NUMERO, texto)
    bruto = numero.group().replace('−', '-') if numero else None
    if bruto is None:
        palabra = re.sub(MONEDA, '', texto, flags=re.IGNORECASE).strip().casefold()
        bruto = str(VALORES[palabra]) if palabra in VALORES else None
        partes_palabra = palabra.split(' y ')
        if len(partes_palabra) == 2:
            decena, unidad = (VALORES.get(p) for p in partes_palabra)
            if decena is not None and unidad is not None and decena >= 20 and decena % 10 == 0 and 1 <= unidad <= 9:
                bruto = str(decena + unidad)
    if bruto is None:
        salida['motivos'].append('NUMERO_NO_NORMALIZABLE')
    else:
        signo = bruto[0] if bruto[0] in '+-' else ''
        cuerpo = bruto.lstrip('+-')
        if ' ' in cuerpo or '\u00a0' in cuerpo:
            if re.fullmatch(r'\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d{1,2})?', cuerpo):
                cuerpo = cuerpo.replace(' ', '').replace('\u00a0', '')
            else:
                salida['motivos'].append('FORMATO_MONETARIO_INVALIDO')
        if '.' in cuerpo and ',' in cuerpo:
            decimal = '.' if cuerpo.rfind('.') > cuerpo.rfind(',') else ','
            miles = ',' if decimal == '.' else '.'
            patron = rf"\d{{1,3}}(?:{re.escape(miles)}\d{{3}})+{re.escape(decimal)}\d{{1,2}}"
            if re.fullmatch(patron, cuerpo):
                cuerpo = cuerpo.replace(miles, '').replace(decimal, '.')
            else:
                salida['motivos'].append('FORMATO_MONETARIO_INVALIDO')
        elif '.' in cuerpo or ',' in cuerpo:
            separador = '.' if '.' in cuerpo else ','
            partes = cuerpo.split(separador)
            if len(partes) == 2 and 1 <= len(partes[1]) <= 2:
                cuerpo = cuerpo.replace(separador, '.')
            elif len(partes) == 2 and len(partes[1]) == 3:
                salida['motivos'].append('SEPARADOR_AMBIGUO')
            elif len(partes) > 2 and 1 <= len(partes[0]) <= 3 and all(len(p) == 3 for p in partes[1:]):
                cuerpo = ''.join(partes)
            else:
                salida['motivos'].append('FORMATO_MONETARIO_INVALIDO')
        if not salida['motivos']:
            try:
                valor = Decimal(signo + cuerpo)
                salida['valor'] = format(valor, 'f')
                if valor <= 0:
                    salida['motivos'].append('IMPORTE_NO_POSITIVO')
            except InvalidOperation:
                salida['motivos'].append('FORMATO_MONETARIO_INVALIDO')
    if salida['moneda'] is None:
        salida['motivos'].append('MONEDA_NO_DETERMINADA')
    elif salida['moneda'] == 'PESO':
        salida['motivos'].append('MONEDA_PESO_NO_ESPECIFICA')
    if salida['motivos']:
        salida['estado'] = 'AMBIGUO'
    return salida


def buscar_importes(texto):
    return [{'texto': m.group(), 'inicio': m.start(), 'fin': m.end()} for m in PATRON.finditer(texto)]
