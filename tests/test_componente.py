"""Regresiones de los puntos 1 y 2; ejecutar python -m unittest -v."""
import json
from pathlib import Path
import unittest
from concurrent.futures import ThreadPoolExecutor
from componente import ComponenteInterprete
from contrato import Propuesta
from interpretacion import interpretar_operaciones_con_relaciones
from monetario import buscar_importes, normalizar_importe
from herramientas.evaluar_producto import CASOS, campo


class DineroTests(unittest.TestCase):
    def test_importes_exactos(self):
        for original, valor, moneda in [
            ('$25', '25', 'USD'), ('€18', '18', 'EUR'), ('US$ 89', '89', 'USD'),
            ('1.250,50 dólares', '1250.50', 'USD'), ('1,250.50 USD', '1250.50', 'USD'),
            ('12,75 dólares', '12.75', 'USD'), ('12.75 EUR', '12.75', 'EUR'),
            ('1 250 dólares', '1250', 'USD'), ('1 250,50 dólares', '1250.50', 'USD'),
            ('-7 dólares', '-7', 'USD'), ('−8 euros', '-8', 'EUR'),
            ('+7 dólares', '7', 'USD'), ('treinta y cuatro pesos', '34', 'PESO'),
            ('cinco dólares', '5', 'USD'), ('0 dólares', '0', 'USD'),
        ]:
            with self.subTest(original=original):
                resultado = normalizar_importe(original)
                self.assertEqual(resultado['original'], original)
                self.assertEqual(resultado['valor'], valor)
                self.assertEqual(resultado['moneda'], moneda)

    def test_abstencion_y_signos(self):
        for original, motivo in [
            ('1.250 dólares', 'SEPARADOR_AMBIGUO'),
            ('1,250 dólares', 'SEPARADOR_AMBIGUO'),
            ('12.34.56 dólares', 'FORMATO_MONETARIO_INVALIDO'),
            ('1..250 dólares', 'FORMATO_MONETARIO_INVALIDO'),
            ('ciento veinte dólares', 'NUMERO_NO_NORMALIZABLE'),
            ('-7 dólares', 'IMPORTE_NO_POSITIVO'),
            ('0 dólares', 'IMPORTE_NO_POSITIVO'),
            ('16 soles', 'MONEDA_NO_DETERMINADA'),
            ('abc12 dólares', 'FORMATO_MONETARIO_INVALIDO'),
            ('1 25 dólares', 'FORMATO_MONETARIO_INVALIDO'),
        ]:
            with self.subTest(original=original):
                resultado = normalizar_importe(original)
                self.assertEqual(resultado['estado'], 'AMBIGUO')
                self.assertIn(motivo, resultado['motivos'])

    def test_limites_no_fragmentan_importes(self):
        texto = 'Pagué 1.250,50 dólares y aboné -7 dólares.'
        self.assertEqual([m['texto'] for m in buscar_importes(texto)], ['1.250,50 dólares', '-7 dólares'])
        self.assertEqual(buscar_importes('12.34.56 dólares')[0]['texto'], '12.34.56 dólares')
        self.assertEqual(buscar_importes('1 250 dólares')[0]['texto'], '1 250 dólares')


class ComponenteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.motor = ComponenteInterprete()

    def test_evaluacion_producto_completa(self):
        for nombre, texto, esperado in CASOS:
            if esperado is None:
                continue
            with self.subTest(caso=nombre):
                resultado = interpretar_operaciones_con_relaciones(texto, nlp=self.motor._modelo)
                self.assertEqual(len(resultado['operaciones']), len(esperado))
                for op, campos in zip(resultado['operaciones'], esperado):
                    for clave, valor in campos.items():
                        self.assertEqual(campo(op, clave), valor)

    def test_casos_criticos_exigen_revision(self):
        for texto, codigo in [
            ('Pagué -7 dólares al proveedor', 'IMPORTE_NO_POSITIVO'),
            ('Vendí pan por 16 soles', 'MONEDA_NO_DETERMINADA'),
            ('Vendí pan y leche por 9 dólares en total', 'VARIOS_PRODUCTOS_SIN_DESGLOSE'),
            ('Pagué 1.250 dólares', 'SEPARADOR_AMBIGUO'),
            ('El cliente devolvió dos cuadernos y le devolví 14 dólares', 'ACCION_FUERA_DE_ALCANCE'),
        ]:
            with self.subTest(texto=texto):
                salida = self.motor.interpretar(texto)
                self.assertTrue(salida.revision.requiere_revision)
                self.assertFalse(salida.revision.registrable)
                self.assertIn(codigo, [b.codigo for b in salida.revision.bloqueos])

    def test_contrato_json_y_decimal(self):
        texto = 'Pagué 1.250,50 dólares al proveedor'
        salida = self.motor.interpretar(texto)
        datos = json.loads(salida.model_dump_json())
        self.assertEqual(datos['operaciones'][0]['importe']['valor'], '1250.50')
        self.assertEqual(datos['operaciones'][0]['importe']['original'], '1.250,50 dólares')
        self.assertEqual(Propuesta.model_validate_json(salida.model_dump_json()), salida)
        self.assertEqual(salida.versiones.esquema, '1.0.0')
        self.assertEqual(len(salida.versiones.modelo), 64)

    def test_varios_productos_conservan_evidencia(self):
        texto = 'Compré arroz por 12 dólares y vendí pan y leche por 9 dólares en total'
        salida = self.motor.interpretar(texto)
        self.assertEqual(len(salida.operaciones), 2)
        evidencia = salida.operaciones[1].productos_mencionados
        self.assertIsNotNone(evidencia)
        self.assertEqual(texto[evidencia.inicio:evidencia.fin], 'pan y leche')
        self.assertEqual(salida.operaciones[1].estado, 'AMBIGUO')

    def test_entradas_y_revision_obligatoria(self):
        with self.assertRaises(TypeError):
            self.motor.interpretar(123)
        with self.assertRaises(ValueError):
            self.motor.interpretar('a' * 10001)
        for texto in ['', 'Pagué $25 al proveedor']:
            salida = self.motor.interpretar(texto)
            self.assertTrue(salida.revision.requiere_revision)
            self.assertFalse(salida.revision.registrable)

    def test_instancia_repetible_con_hilos(self):
        textos = ['Pagué $25 al proveedor', 'Vendí pan y leche por 9 dólares', 'Pagué -7 dólares'] * 2
        referencia = [self.motor.interpretar_json(t) for t in textos]
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(self.motor.interpretar_json, textos)), referencia)


if __name__ == '__main__':
    unittest.main()
