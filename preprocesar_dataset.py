"""Extrae los puntos de la mano de todo el dataset y los guarda en datos/landmarks.npz.

Uso:
    python preprocesar_dataset.py                       # todo lo que haya en dataset/
    python preprocesar_dataset.py --origen a.7z carpeta_b --salida datos/x.npz

`--origen` acepta carpetas y archivos .7z/.zip/.tar. Los comprimidos se leen sin
descomprimirlos (no hace falta extraerlos). En una carpeta se procesan sus .7z/.zip/.tar
y sus carpetas train/test ya extraídas; si existen ambas versiones de un mismo
dataset (X.7z y carpeta X/), se usa solo la carpeta. Dentro de train/test:
  - imágenes (.jpg/.png): la letra sale de la carpeta que las contiene (train/A/...).
  - videos (.mp4/.avi/.mov): la letra sale del nombre S<id>-<Letra>-<Vista>-<Rep>
    o, si no se puede, de la carpeta que los contiene.
Archivos fuera de train/test (la vista de perfil viene así) se reparten por sujeto:
--sujetos-test (S20,S21 por defecto) van a test.
Los videos se muestrean a FPS_MUESTREO para que coincidan con la app en vivo.
"""
import argparse
import os
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import lsm_comun as lc

EXT_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp")
EXT_VIDEO = (".mp4", ".avi", ".mov", ".mkv")

_detector_imagen = None


def letra_de_ruta(ruta, es_video):
    if es_video:
        partes = os.path.splitext(os.path.basename(ruta))[0].split("-")
        if len(partes) >= 2:
            letra = lc.normalizar_letra(partes[1])
            if letra:
                return letra
    carpeta = os.path.dirname(ruta)
    while carpeta and os.path.basename(carpeta).lower() not in ("train", "test"):
        letra = lc.normalizar_letra(os.path.basename(carpeta))
        if letra:
            return letra
        carpeta = os.path.dirname(carpeta)
    return None


def buscar_archivos(origen):
    """Devuelve lista de (ruta, split, letra o None, es_video)."""
    archivos = []
    for sub, _, nombres in os.walk(origen):
        for nombre in nombres:
            ext = os.path.splitext(nombre)[1].lower()
            if ext in EXT_IMAGEN or ext in EXT_VIDEO:
                ruta = os.path.join(sub, nombre)
                es_video = ext in EXT_VIDEO
                split = split_de_ruta(ruta)
                if split:
                    archivos.append((ruta, split, letra_de_ruta(ruta, es_video), es_video))
    return archivos


SUJETOS_TEST = {"S20", "S21"}  # los mismos que usa la partición test del dinámico frontal


def split_de_ruta(ruta):
    """train/test según la carpeta. Si no hay (vista de perfil del dinámico), según el
    sujeto del nombre S<id>-...: SUJETOS_TEST van a test, el resto a train."""
    partes = [p.lower() for p in ruta.replace("\\", "/").split("/")]
    split = next((p for p in reversed(partes[:-1]) if p in ("train", "test")), None)
    if split:
        return split
    sujeto = partes[-1].split("-")[0].upper()
    if len(sujeto) > 1 and sujeto[0] == "S" and sujeto[1:].isdigit():
        return "test" if sujeto in SUJETOS_TEST else "train"
    return None


def leer_comprimido(ruta):
    """Recorre un .7z/.zip/.tar sin descomprimirlo a disco (en un HDD, escribir ~280 mil
    imágenes chicas tarda horas). bsdtar (incluido en Windows) lo convierte a un flujo
    tar que se lee en memoria. Devuelve (nombre, bytes) por archivo."""
    import subprocess
    import tarfile

    proc = subprocess.Popen(["tar", "-cf", "-", "--format", "pax", "@" + ruta], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for m in tf:
                if m.isfile():
                    yield m.name, tf.extractfile(m).read()
    finally:
        proc.kill()
        proc.wait()


EXT_COMPRIMIDO = (".7z", ".zip", ".tar")


def fuentes(origen):
    """Genera (nombre, split, letra, es_video, carga). carga = ruta en disco o bytes."""
    if os.path.isdir(origen):
        for ruta, split, letra, es_video in buscar_archivos(origen):
            yield ruta, split, letra, es_video, ruta
        for nombre in sorted(os.listdir(origen)):
            base, ext = os.path.splitext(nombre)
            if ext.lower() in EXT_COMPRIMIDO and not os.path.isdir(os.path.join(origen, base)):
                yield from fuentes(os.path.join(origen, nombre))
        return
    print(f"  leyendo {origen} sin descomprimir...", flush=True)
    for nombre, datos in leer_comprimido(origen):
        ext = os.path.splitext(nombre)[1].lower()
        split = split_de_ruta(nombre)
        if split and (ext in EXT_IMAGEN or ext in EXT_VIDEO):
            es_video = ext in EXT_VIDEO
            yield nombre, split, letra_de_ruta(nombre, es_video), es_video, datos


def _leer_imagen(carga):
    import cv2
    # np.fromfile + imdecode soporta rutas con acentos/Ñ en Windows (cv2.imread no).
    datos = np.frombuffer(carga, np.uint8) if isinstance(carga, bytes) else np.fromfile(carga, dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size else None


def procesar_imagen(carga):
    global _detector_imagen
    if _detector_imagen is None:
        _detector_imagen = lc.crear_detector("imagen")
    frame = _leer_imagen(carga)
    if frame is None:
        return None
    puntos, izquierda = lc.detectar(_detector_imagen, frame)
    if puntos is None:
        return np.full((1, lc.N_PUNTOS, 3), np.nan, dtype=np.float32)
    return lc.canonizar(puntos, izquierda)[None]


def procesar_video(ruta):
    import cv2
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    detector = lc.crear_detector("video")
    muestras, siguiente, i = [], 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if t + 1e-6 < siguiente:
                continue
            siguiente += 1.0 / lc.FPS_MUESTREO
            puntos, izquierda = lc.detectar(detector, frame, t * 1000)
            if puntos is None:
                muestras.append(np.full((lc.N_PUNTOS, 3), np.nan, dtype=np.float32))
            else:
                muestras.append(lc.canonizar(puntos, izquierda))
    finally:
        cap.release()
        detector.close()
    return np.stack(muestras) if muestras else None


def procesar(tarea):
    nombre, carga, es_video = tarea
    try:
        if es_video and isinstance(carga, bytes):  # OpenCV solo abre videos desde disco
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(nombre)[1], delete=False) as f:
                f.write(carga)
            try:
                return procesar_video(f.name)
            finally:
                os.remove(f.name)
        return procesar_video(carga) if es_video else procesar_imagen(carga)
    except Exception as e:  # un archivo dañado no debe tumbar todo el proceso
        print(f"  [error] {nombre}: {e}")
        return None


def procesar_lote(tareas):
    return [procesar(t) for t in tareas]


def por_lotes(iterable, tamano):
    lote = []
    for x in iterable:
        lote.append(x)
        if len(lote) == tamano:
            yield lote
            lote = []
    if lote:
        yield lote


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origen", nargs="+", default=["dataset"],
                        help="carpetas y/o archivos .7z/.zip/.tar (sin descomprimir)")
    parser.add_argument("--salida", default=os.path.join("datos", "landmarks.npz"))
    parser.add_argument("--procesos", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--sujetos-test", default=",".join(sorted(SUJETOS_TEST)),
                        help="sujetos para test cuando el archivo no está en una carpeta train/test")
    args = parser.parse_args()
    SUJETOS_TEST.clear()
    SUJETOS_TEST.update(s.strip().upper() for s in args.sujetos_test.split(",") if s.strip())

    for o in args.origen:
        if not os.path.exists(o):
            raise SystemExit(f"No existe '{o}'. Descarga los datasets primero: python descargar_dataset.py")
    lc.asegurar_detector()
    print(f"Leyendo {', '.join(args.origen)} con {args.procesos} procesos...")

    puntos, inicio, largo, etiquetas, splits, es_video = [], [], [], [], [], []
    detectadas, totales = defaultdict(int), Counter()
    sin_letra = []
    total_frames, n, t0 = 0, 0, time.time()

    def guardar(lote, resultados):
        nonlocal total_frames, n
        for (nombre, split, letra, video), resultado in zip(lote, resultados):
            n += 1
            totales[letra] += 1
            if n % 5000 == 0:
                print(f"  {n} archivos ({time.time() - t0:.0f} s)", flush=True)
            if resultado is None or np.isnan(resultado[:, 0, 0]).all():
                continue
            detectadas[letra] += 1
            puntos.append(resultado)
            inicio.append(total_frames)
            largo.append(len(resultado))
            etiquetas.append(letra)
            splits.append(split)
            es_video.append(video)
            total_frames += len(resultado)

    def con_letra():
        for nombre, split, letra, video, carga in (f for o in args.origen for f in fuentes(o)):
            if letra is None:
                sin_letra.append(nombre)
            else:
                yield (nombre, split, letra, video), carga

    # Lotes enviados a los procesos con un máximo en vuelo: así la lectura del .7z, que
    # va en este proceso, se traslapa con la detección y la memoria no crece sin límite.
    pendientes = deque()
    with ProcessPoolExecutor(max_workers=args.procesos) as pool:
        for lote in por_lotes(con_letra(), 64):
            metas = [m for m, _ in lote]
            tareas = [(m[0], carga, m[3]) for m, carga in lote]
            pendientes.append((metas, pool.submit(procesar_lote, tareas)))
            while len(pendientes) >= args.procesos * 3:
                metas, futuro = pendientes.popleft()
                guardar(metas, futuro.result())
        while pendientes:
            metas, futuro = pendientes.popleft()
            guardar(metas, futuro.result())
    print(f"  {n} archivos ({time.time() - t0:.0f} s)")

    if sin_letra:
        print(f"Aviso: {len(sin_letra)} archivos sin letra reconocible se omitieron. Ejemplos:")
        for nombre in sin_letra[:5]:
            print("   ", nombre)

    if not puntos:
        raise SystemExit("No se encontraron manos (¿hay carpetas train/test con imágenes o videos?).")
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    np.savez_compressed(
        args.salida,
        puntos=np.concatenate(puntos).astype(np.float32),
        inicio=np.array(inicio, dtype=np.int64),
        largo=np.array(largo, dtype=np.int64),
        etiqueta=np.array(etiquetas),
        split=np.array(splits),
        es_video=np.array(es_video, dtype=bool),
    )

    print(f"\nGuardado {args.salida}: {len(puntos)} muestras con mano detectada.")
    print("Detección por letra (archivos con mano / archivos totales):")
    for letra in lc.ABECEDARIO:
        if totales[letra]:
            print(f"  {letra}: {detectadas[letra]}/{totales[letra]} ({100 * detectadas[letra] / totales[letra]:.0f}%)")
    faltantes = [l for l in lc.ABECEDARIO if not detectadas[l]]
    if faltantes:
        print(f"Aviso: sin muestras para {faltantes}; el modelo no podrá reconocerlas.")


if __name__ == "__main__":
    main()
