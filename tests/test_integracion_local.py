"""Contrato y carga local para el futuro consumidor del componente."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from componente import ComponenteInterprete
from contrato import Propuesta, construir_propuesta
from interpretacion import cargar_modelo, interpretar_operaciones_con_relaciones

ROOT = Path(__file__).resolve().parent.parent


class IntegracionLocalTests(unittest.TestCase):
    def test_evidencia_no_puede_sobrepasar_el_texto(self):
        resultado = interpretar_operaciones_con_relaciones('Pagué 25 dólares', nlp=cargar_modelo())
        resultado['operaciones'][0]['campos']['precio']['evidencia']['fin'] += 100
        versiones = {'componente': '1.1.0', 'reglas': '1.0.1', 'modelo': 'prueba', 'esquema': '1.0.0'}
        with self.assertRaisesRegex(ValueError, 'evidencia'):
            construir_propuesta(resultado, versiones)

    def test_schema_versionado(self):
        schema = json.loads((ROOT / 'schemas/propuesta-v1.schema.json').read_text(encoding='utf-8'))
        self.assertEqual(schema, Propuesta.model_json_schema())

    def test_modelo_ausente_falla_explicitamente(self):
        with self.assertRaisesRegex(RuntimeError, 'No existe la carpeta del modelo'):
            ComponenteInterprete('modelo_inexistente_para_prueba')

    def test_ruta_relativa_no_depende_del_directorio_de_ejecucion(self):
        anterior = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='interprete-cwd-') as temporal:
            try:
                os.chdir(temporal)
                with patch.dict(os.environ, {'INTERPRETE_MODELO': 'custom_ner_model'}):
                    motor = ComponenteInterprete()
                versiones = motor.versiones
                versiones['modelo'] = 'modificado'
                self.assertNotEqual(motor.versiones['modelo'], 'modificado')
                propuesta = motor.interpretar('Pagué $25 al proveedor')
                self.assertEqual(propuesta.operaciones[0].importe.valor, '25')
                self.assertFalse(propuesta.revision.registrable)
            finally:
                os.chdir(anterior)
