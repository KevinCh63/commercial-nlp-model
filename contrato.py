"""Contrato público v1: valores monetarios exactos serializados como cadenas."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

VERSION_ESQUEMA = '1.0.0'
Estado = Literal['VALIDO', 'PARCIAL', 'AMBIGUO', 'NO_RECONOCIDO']


class Contrato(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Evidencia(Contrato):
    texto: str
    inicio: int = Field(ge=0)
    fin: int = Field(gt=0)


class Importe(Contrato):
    original: str | None
    valor: str | None = Field(pattern=r'^[-+]?\d+(?:\.\d+)?$')
    moneda: Literal['USD', 'EUR', 'PESO'] | None
    estado: Literal['AUSENTE', 'IDENTIFICADO', 'AMBIGUO']
    motivos: list[str]


class Operacion(Contrato):
    id: str
    texto: str
    agente: str | None
    contraparte: str | None
    accion: str | None
    tipo: Literal['INGRESO', 'EGRESO'] | None
    cantidad: str | None
    producto: str | None
    concepto: str | None
    precio: str | None
    moneda: Literal['USD', 'EUR', 'PESO'] | None
    estado: Estado
    importe: Importe
    ambito_precio: Literal['UNITARIO', 'TOTAL', 'DESCONOCIDO', 'NO_ESPECIFICADO']
    evidencias: dict[str, Evidencia | None]
    productos_mencionados: Evidencia | None
    incidencias: list[str]
    faltantes: list[str]
    advertencias: list[str]


class Relacion(Contrato):
    operacion_origen_id: str
    operacion_destino_id: str | None
    tipo: Literal['ABONO_DE', 'PAGO_DE']
    evidencia: str
    estado: Literal['CONFIRMADA', 'AMBIGUA']


class Bloqueo(Contrato):
    codigo: str
    operacion_id: str | None


class Revision(Contrato):
    requiere_revision: Literal[True]
    registrable: Literal[False]
    bloqueos: list[Bloqueo]
    motivo: str


class Versiones(Contrato):
    componente: str
    modelo: str
    reglas: str
    esquema: Literal['1.0.0']


class Propuesta(Contrato):
    texto: str
    estado: Estado
    versiones: Versiones
    operaciones: list[Operacion]
    relaciones: list[Relacion]
    revision: Revision
    advertencias: list[str]


def construir_propuesta(resultado, versiones):
    operaciones = []
    for op in resultado['operaciones']:
        datos = {k: op[k] for k in Operacion.model_fields if k in op}
        datos['ambito_precio'] = op['campos']['precio']['ambito']
        datos['evidencias'] = {k: v['evidencia'] for k, v in op['campos'].items()}
        datos['productos_mencionados'] = op.get('productos_mencionados')
        operaciones.append(Operacion.model_validate(datos))
    relaciones = [Relacion.model_validate({k: r[k] for k in Relacion.model_fields}) for r in resultado['relaciones']]
    propuesta = Propuesta(texto=resultado['texto'], estado=resultado['estado'], versiones=Versiones(**versiones),
                          operaciones=operaciones, relaciones=relaciones, revision=Revision(**resultado['revision']),
                          advertencias=resultado['advertencias'])
    for op in propuesta.operaciones:
        for evidencia in [*op.evidencias.values(), op.productos_mencionados]:
            if evidencia and (not 0 <= evidencia.inicio < evidencia.fin <= len(propuesta.texto)
                              or propuesta.texto[evidencia.inicio:evidencia.fin] != evidencia.texto):
                raise ValueError('La evidencia no coincide con el texto original')
    return propuesta
