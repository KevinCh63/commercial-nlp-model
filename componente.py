"""Componente local de inferencia; no entrena, publica ni registra operaciones."""
from pathlib import Path
import hashlib
import json
import os
import threading
import spacy
from contrato import VERSION_ESQUEMA, construir_propuesta
from interpretacion import interpretar_operaciones_con_relaciones

VERSION_COMPONENTE = '1.1.0'
VERSION_REGLAS = '1.0.1'
ARCHIVOS_REGLAS = ('interpretacion.py', 'monetario.py', 'revision.py')


def sha256_archivo(ruta):
    digest = hashlib.sha256()
    with Path(ruta).open('rb') as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b''):
            digest.update(bloque)
    return digest.hexdigest()


def inventario(carpeta):
    return {p.relative_to(carpeta).as_posix(): sha256_archivo(p)
            for p in sorted(Path(carpeta).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def huella(inventario_archivos):
    return hashlib.sha256(json.dumps(inventario_archivos, sort_keys=True).encode('utf-8')).hexdigest()


class ComponenteInterprete:
    def __init__(self, ruta_modelo=None):
        raiz = Path(__file__).resolve().parent
        ruta = Path(ruta_modelo or os.environ.get('INTERPRETE_MODELO', raiz / 'custom_ner_model'))
        ruta = (raiz / ruta).resolve() if not ruta.is_absolute() else ruta.resolve()
        if not ruta.is_dir():
            raise RuntimeError(f'No existe la carpeta del modelo: {ruta}')
        self._modelo = spacy.load(ruta)
        self._lock = threading.RLock()
        self._versiones = {'componente': VERSION_COMPONENTE, 'modelo': huella(inventario(ruta)),
                          'reglas': VERSION_REGLAS + '+' + huella({p: sha256_archivo(Path(__file__).with_name(p)) for p in ARCHIVOS_REGLAS}),
                          'esquema': VERSION_ESQUEMA}

    @property
    def versiones(self):
        return dict(self._versiones)

    def interpretar(self, texto):
        if not isinstance(texto, str):
            raise TypeError('texto debe ser una cadena')
        if len(texto) > 10000:
            raise ValueError('El componente admite como máximo 10000 caracteres')
        # spaCy comparte estructuras mutables; las llamadas a una instancia se
        # serializan. La concurrencia entre procesos corresponde al consumidor.
        with self._lock:
            resultado = interpretar_operaciones_con_relaciones(texto, nlp=self._modelo)
            return construir_propuesta(resultado, dict(self.versiones))

    def interpretar_json(self, texto):
        return self.interpretar(texto).model_dump_json()
