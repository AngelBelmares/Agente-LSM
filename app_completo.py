"""Reconocimiento en vivo del abecedario LSM con la webcam.

Uso:
    python app_completo.py                 # webcam 0
    python app_completo.py --camara 1
    python app_completo.py --video ruta.mp4 [--sin-ventana]   # probar con un video

Teclas: q/Esc salir · Espacio agrega espacio · Retroceso borra · c limpia el texto.
Mantén una seña ~1 s para que se escriba en el texto.
"""
import argparse
import json
import os
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import lsm_comun as lc

VERDE, NARANJA, BLANCO, NEGRO = (0, 200, 0), (0, 165, 255), (255, 255, 255), (0, 0, 0)


def cargar_fuente(tamano):
    for ruta in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(ruta):
            return ImageFont.truetype(ruta, tamano)
    return ImageFont.load_default(tamano)


def escribir(frame, lineas):
    """cv2.putText no dibuja Ñ ni acentos; se usa Pillow. lineas: [(texto, (x, y), color BGR, fuente)]."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dibujo = ImageDraw.Draw(img)
    for texto, pos, color, fuente in lineas:
        dibujo.text(pos, texto, font=fuente, fill=color[::-1], stroke_width=2, stroke_fill=NEGRO)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def dibujar_mano(frame, puntos):
    from mediapipe.tasks.python.vision import HandLandmarksConnections

    alto = frame.shape[0]
    px = [(int(x * alto), int(y * alto)) for x, y, _ in puntos]  # x venía escalada por ancho/alto
    for c in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, px[c.start], px[c.end], BLANCO, 2)
    for p in px:
        cv2.circle(frame, p, 4, VERDE, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camara", type=int, default=0)
    parser.add_argument("--video", help="usar un archivo de video en vez de la webcam")
    parser.add_argument("--umbral", type=float, default=0.6, help="confianza mínima para aceptar una letra")
    parser.add_argument("--sostener", type=float, default=1.0, help="segundos que hay que sostener una seña para escribirla")
    parser.add_argument("--sin-espejo", action="store_true", help="no voltear la imagen en pantalla")
    parser.add_argument("--sin-ventana", action="store_true", help="no abrir ventana; imprime las letras (pruebas)")
    args = parser.parse_args()

    if not os.path.exists(lc.RUTA_MODELO) or not os.path.exists(lc.RUTA_CLASES):
        raise SystemExit(f"No existe {lc.RUTA_MODELO}. Corre primero preprocesar_dataset.py y entrenar_modelo.py.")
    import tensorflow as tf
    modelo = tf.keras.models.load_model(lc.RUTA_MODELO)
    with open(lc.RUTA_CLASES, encoding="utf-8") as f:
        clases = json.load(f)["clases"]
    detector = lc.crear_detector("video")

    fuente = cargar_fuente(int(os.environ.get("LSM_FUENTE", 28)))
    fuente_chica = cargar_fuente(20)
    cap = cv2.VideoCapture(args.video if args.video else args.camara)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara/video. Prueba con --camara 1.")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    espejo = not args.sin_espejo and not args.video

    buffer = deque(maxlen=lc.PASOS)
    historial = deque(maxlen=5)          # suavizado de probabilidades
    sin_mano_desde = None
    siguiente_muestra = 0.0
    letra_estable, estable_desde, ya_escrita = None, None, False
    texto = ""
    estado, color = "Coloca tu mano frente a la cámara", NARANJA
    top = []
    inicio, n_frame, ultimo_ts = time.monotonic(), 0, -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frame += 1
            t = n_frame / fps_video if args.video else time.monotonic() - inicio
            ts = max(int(t * 1000), ultimo_ts + 1)  # MediaPipe exige marcas crecientes
            ultimo_ts = ts

            puntos, izquierda = lc.detectar(detector, frame, ts)
            if puntos is None:
                if sin_mano_desde is None:
                    sin_mano_desde = t
                if t - sin_mano_desde > 0.5:
                    buffer.clear()
                    historial.clear()
                    letra_estable, ya_escrita = None, False
                    estado, color, top = "Coloca tu mano frente a la cámara", NARANJA, []
            else:
                sin_mano_desde = None

            if t >= siguiente_muestra:
                siguiente_muestra = t + 1.0 / lc.FPS_MUESTREO
                if puntos is not None or buffer:
                    buffer.append(lc.canonizar(puntos, izquierda) if puntos is not None
                                  else np.full((lc.N_PUNTOS, 3), np.nan, np.float32))
                if len(buffer) == lc.PASOS and puntos is not None:
                    x = lc.caracteristicas(np.stack(buffer)[None])
                    historial.append(modelo(x, training=False).numpy()[0])
                    prob = np.mean(historial, axis=0)
                    orden = np.argsort(prob)[::-1][:3]
                    top = [(clases[i], prob[i]) for i in orden]
                    letra, confianza = top[0]
                    if confianza >= args.umbral:
                        estado, color = f"Seña: {letra}  ({confianza * 100:.0f}%)", VERDE
                        if letra != letra_estable:
                            letra_estable, estable_desde, ya_escrita = letra, t, False
                        elif not ya_escrita and t - estable_desde >= args.sostener:
                            texto += letra
                            ya_escrita = True
                            if args.sin_ventana:
                                print(f"[{t:6.2f} s] {letra} ({confianza * 100:.0f}%)")
                    else:
                        estado, color = "Buscando seña...", NARANJA
                        letra_estable = None
                elif puntos is not None and len(buffer) < lc.PASOS:
                    estado, color = "Detectando mano...", NARANJA

            if args.sin_ventana:
                continue

            if puntos is not None:
                dibujar_mano(frame, puntos)  # se dibuja antes de voltear: queda alineado
            vista = cv2.flip(frame, 1) if espejo else frame
            lineas = [(estado, (20, 15), color, fuente),
                      (f"Texto: {texto[-30:]}", (20, vista.shape[0] - 45), BLANCO, fuente)]
            for k, (l, p) in enumerate(top):
                lineas.append((f"{l} {p * 100:4.0f}%", (vista.shape[1] - 130, 15 + 26 * k), BLANCO, fuente_chica))
            vista = escribir(vista, lineas)
            cv2.imshow("Agente LSM", vista)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord(" "):
                texto += " "
            elif tecla == 8:
                texto = texto[:-1]
            elif tecla == ord("c"):
                texto = ""
            if cv2.getWindowProperty("Agente LSM", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
    if texto:
        print(f"Texto final: {texto}")


if __name__ == "__main__":
    main()
