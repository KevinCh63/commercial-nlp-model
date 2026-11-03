"""Comprobación de propuestas, sin persistencia ni confirmación de movimientos."""
import re


def anotar_revision(resultado):
    bloqueos = []
    if not resultado['operaciones']:
        bloqueos.append({'codigo': 'SIN_OPERACIONES', 'operacion_id': None})
    if re.search(r'\b(?:devolv\w*|devoluci[oó]n|descuent\w*|fiad\w*)\b', resultado.get('texto') or '', re.IGNORECASE):
        bloqueos.append({'codigo': 'ACCION_FUERA_DE_ALCANCE', 'operacion_id': None})
    for operacion in resultado['operaciones']:
        motivos = set(operacion.get('incidencias', []))
        importe = operacion.get('importe', {})
        motivos.update(importe.get('motivos', []))
        if importe.get('valor') is None:
            motivos.add('IMPORTE_NO_NORMALIZADO')
        if not importe.get('moneda'):
            motivos.add('MONEDA_NO_DETERMINADA')
        if operacion.get('estado') != 'VALIDO':
            motivos.add('LECTURA_INCOMPLETA_O_AMBIGUA')
        if operacion.get('accion') in {'comprar', 'vender'} and not operacion.get('cantidad'):
            motivos.add('CANTIDAD_NO_DETERMINADA')
        if operacion.get('campos', {}).get('precio', {}).get('ambito') == 'UNITARIO':
            motivos.add('TOTAL_NO_EXPLICITADO')
        if operacion.get('tipo') is None:
            motivos.add('TIPO_NO_DETERMINADO')
        bloqueos.extend({'codigo': codigo, 'operacion_id': operacion.get('id')} for codigo in sorted(motivos))
    resultado['revision'] = {
        'requiere_revision': True,
        'registrable': False,
        'bloqueos': bloqueos,
        'motivo': 'La interpretación es una propuesta; este componente no confirma ni registra movimientos.',
    }
    return resultado
