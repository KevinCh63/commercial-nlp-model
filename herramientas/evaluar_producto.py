"""Prueba de producto sin entrenar, promover ni modificar el intérprete.

Las expectativas se fijan antes de ejecutar los casos. Las puntuaciones
comprueban los campos indicados, no toda la exactitud posible de una operación.
"""
from pathlib import Path
import sys
import json
import time
import statistics
import argparse
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from interpretacion import cargar_modelo, interpretar_operaciones_con_relaciones, serializar_resultado

CASOS = [
    ("venta_simple", "Vendí 2 cuadernos por 14 dólares en total", [{"accion": "vender", "cantidad": "2", "producto": "cuadernos", "precio": "14 dólares", "moneda": "USD"}]),
    ("compra_simple", "Compré 4 cajas de leche por 32 dólares en total", [{"accion": "comprar", "cantidad": "4 cajas", "producto": "leche", "precio": "32 dólares"}]),
    ("gasto", "Gasté 17 dólares en limpieza", [{"accion": "gastar", "concepto": "limpieza", "producto": None, "precio": "17 dólares"}]),
    ("contraparte", "Pagué 23 dólares al proveedor", [{"accion": "pagar", "contraparte": "proveedor", "precio": "23 dólares"}]),
    ("agente", "Daniela vendió 3 carpetas por 21 dólares en total", [{"accion": "vender", "agente": "Daniela", "cantidad": "3", "producto": "carpetas"}]),
    ("producto_compuesto", "Compré 2 botellas de aceite de oliva por 19 dólares en total", [{"cantidad": "2 botellas", "producto": "aceite de oliva", "precio": "19 dólares"}]),
    ("simbolo_dolar", "Vendí una mochila por $25", [{"accion": "vender", "producto": "mochila", "precio": "$25", "moneda": "USD"}]),
    ("simbolo_euro", "Pagué €18 por transporte", [{"accion": "pagar", "precio": "€18", "moneda": "EUR"}]),
    ("decimal_coma", "Pagué 12,75 dólares al proveedor", [{"precio": "12,75 dólares", "moneda": "USD"}]),
    ("miles_decimales", "Pagué 1.250,50 dólares al proveedor", [{"precio": "1.250,50 dólares", "moneda": "USD"}]),
    ("precio_unitario", "Vendí 4 botellas de jugo a 3 dólares por unidad", [{"cantidad": "4 botellas", "producto": "jugo", "precio": "3 dólares", "campos.precio.ambito": "UNITARIO"}]),
    ("dos_operaciones", "Compré arroz por 28 dólares y vendí café por 11 dólares", [{"accion": "comprar", "producto": "arroz", "precio": "28 dólares"}, {"accion": "vender", "producto": "café", "precio": "11 dólares"}]),
    ("importe_compartido", "Vendí pan y leche por 9 dólares en total", None),
    ("negacion", "No pagué los 35 dólares al proveedor", []),
    ("futuro", "Mañana venderé cinco cuadernos por 20 dólares", []),
    ("condicional", "Si vendo arroz, cobraré 20 dólares", []),
    ("incertidumbre", "Tal vez recibí 31 dólares del cliente", []),
    ("pregunta", "¿Vendiste dos cuadernos por 12 dólares?", []),
    ("negacion_parcial", "No compré café, pero vendí arroz por 13 dólares", [{"accion": "vender", "producto": "arroz", "precio": "13 dólares"}]),
    ("intencion_pasado", "Quería comprar una caja de jabón", []),
    ("abreviaturas", "vendi 2 pqts de arroz x 8 usd", [{"accion": "vender", "precio": "8 usd", "moneda": "USD"}]),
    ("error_ortografico", "Compre 3 cuadernos por 15 dolres", None),
    ("devolucion", "El cliente devolvió dos cuadernos y le devolví 14 dólares", None),
    ("saldo", "Compré papel por 85 dólares y aboné 25 dólares de esa compra", [{"accion": "comprar", "producto": "papel", "precio": "85 dólares"}, {"accion": "abonar", "precio": "25 dólares"}]),
    ("unidad_nueva", "Compré dos cartones de huevos por 18 dólares en total", [{"cantidad": "dos cartones", "producto": "huevos", "precio": "18 dólares"}]),
    ("concepto_nuevo", "Pagué 38 dólares por mantenimiento del congelador", [{"accion": "pagar", "concepto": "mantenimiento del congelador", "producto": None, "precio": "38 dólares"}]),
    ("moneda_no_admitida", "Vendí pan por 16 soles", None),
    ("importe_negativo", "Pagué -7 dólares al proveedor", None),
    ("entrada_vacia", "", []),
    ("tipo_incorrecto", 123, []),
]


def campo(objeto, ruta):
    for parte in ruta.split("."):
        if not isinstance(objeto, dict):
            return None
        objeto = objeto.get(parte)
    return objeto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--salida', default=str(ROOT / 'resultados_ner' / 'evaluacion_producto_actual.json'))
    argumentos = parser.parse_args()
    inicio = time.perf_counter()
    modelo = cargar_modelo()
    carga = time.perf_counter() - inicio
    registros, tiempos = [], []
    for nombre, texto, esperado in CASOS:
        inicio = time.perf_counter()
        try:
            salida = interpretar_operaciones_con_relaciones(texto, nlp=modelo)
            tiempo = (time.perf_counter() - inicio) * 1000
            tiempos.append(tiempo)
            ops = salida["operaciones"]
            diferencias = []
            if esperado is not None:
                if len(ops) != len(esperado):
                    diferencias.append({"campo": "cantidad_operaciones", "esperado": len(esperado), "obtenido": len(ops)})
                for i, (op, requisitos) in enumerate(zip(ops, esperado)):
                    for clave, valor in requisitos.items():
                        if campo(op, clave) != valor:
                            diferencias.append({"operacion": i, "campo": clave, "esperado": valor, "obtenido": campo(op, clave)})
            evidencia_invalida = []
            for i, op in enumerate(ops):
                for clave, meta in op.get("campos", {}).items():
                    evidencia = meta.get("evidencia")
                    if evidencia and isinstance(texto, str):
                        s, e = evidencia.get("inicio"), evidencia.get("fin")
                        if not isinstance(s, int) or not isinstance(e, int) or not 0 <= s < e <= len(texto) or texto[s:e] != evidencia.get("texto"):
                            evidencia_invalida.append({"operacion": i, "campo": clave, "evidencia": evidencia})
            registros.append({"caso": nombre, "texto": texto, "esperado": esperado,
                              "aprobado": not diferencias if esperado is not None else None,
                              "diferencias": diferencias, "evidencia_invalida": evidencia_invalida,
                              "latencia_ms": tiempo, "salida": salida})
        except Exception as exc:
            registros.append({"caso": nombre, "texto": texto, "error": f"{type(exc).__name__}: {exc}", "aprobado": False})
    # Repetición y llamadas simultáneas sobre una única instancia cargada.
    textos = [t for _, t, _ in CASOS[:12]]
    def ejecutar(texto):
        return serializar_resultado(interpretar_operaciones_con_relaciones(texto, nlp=modelo))
    referencia = [ejecutar(t) for t in textos]
    repetidas = [ejecutar(t) for t in textos]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrentes = list(pool.map(ejecutar, textos * 2))
    resultado = {
        "alcance": "Prueba exploratoria local; no mide exactitud poblacional ni capacidad de producción.",
        "carga_segundos": carga, "latencia_mediana_ms": statistics.median(tiempos),
        "latencia_p95_ms": sorted(tiempos)[int(0.95 * (len(tiempos) - 1))],
        "comprobables": sum(r.get("esperado") is not None for r in registros),
        "aprobados": sum(r.get("aprobado") is True for r in registros),
        "errores_ejecucion": sum("error" in r for r in registros),
        "evidencias_invalidas": sum(len(r.get("evidencia_invalida", [])) for r in registros),
        "repeticion_estable": referencia == repetidas,
        "concurrencia_estable": concurrentes == referencia * 2,
        "casos": registros,
    }
    destino = Path(argumentos.salida)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in resultado.items() if k != "casos"}, ensure_ascii=False, indent=2))
    for registro in registros:
        ops = registro.get("salida", {}).get("operaciones", [])
        print(registro["caso"], registro.get("aprobado"), registro.get("diferencias"),
              [(o.get("accion"), o.get("producto"), o.get("concepto"), o.get("precio"), o.get("estado")) for o in ops])


if __name__ == "__main__":
    main()
