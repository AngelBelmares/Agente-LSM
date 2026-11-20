"""Utilidades compartidas: detector de manos (MediaPipe) y extracción de características.

El modelo no ve píxeles: ve los 21 puntos de la mano que detecta MediaPipe.
Así el fondo, la iluminación y la ropa dejan de importar, que era la razón
principal de que la CNN entrenada con fondo verde fallara con la webcam.
"""
import os
import unicodedata
import urllib.request

# El modelo se guarda y carga con TensorFlow; ignora un ~/.keras/keras.json que pida otro backend.
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_MODELOS = os.path.join(DIR_BASE, "modelos")
RUTA_DETECTOR = os.path.join(DIR_MODELOS, "hand_landmarker.task")
URL_DETECTOR = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task")
RUTA_MODELO = os.path.join(DIR_MODELOS, "modelo_lsm.keras")
RUTA_CLASES = os.path.join(DIR_MODELOS, "clases.json")

# Abecedario LSM: 21 letras estáticas + 6 dinámicas (requieren movimiento).
LETRAS_ESTATICAS = list("ABCDEFGHILMNOPRSTUVWY")
LETRAS_DINAMICAS = ["J", "K", "Ñ", "Q", "X", "Z"]
ABECEDARIO = sorted(LETRAS_ESTATICAS + LETRAS_DINAMICAS)

# Ventana temporal que ve el modelo: PASOS muestras tomadas a FPS_MUESTREO.
FPS_MUESTREO = 15
PASOS = 15  # 1 segundo
N_PUNTOS = 21
N_CARACT = N_PUNTOS * 3 + 3  # 63 de forma de la mano + 3 de trayectoria de la muñeca


def normalizar_letra(texto):
    """Devuelve la letra del abecedario que representa `texto`, o None."""
    if not texto:
        return None
    t = unicodedata.normalize("NFC", texto).strip().upper()
    alias = {"NN": "Ñ", "N~": "Ñ", "~N": "Ñ", "ENIE": "Ñ", "ENYE": "Ñ"}
    t = alias.get(t, t)
    return t if t in ABECEDARIO else None


def asegurar_detector():
    """Descarga el modelo oficial de MediaPipe Hand Landmarker si no existe."""
    if not os.path.exists(RUTA_DETECTOR):
        os.makedirs(DIR_MODELOS, exist_ok=True)
        print(f"Descargando detector de manos de MediaPipe en {RUTA_DETECTOR} ...")
        tmp = RUTA_DETECTOR + ".part"
        urllib.request.urlretrieve(URL_DETECTOR, tmp)
        os.replace(tmp, RUTA_DETECTOR)
    return RUTA_DETECTOR


def crear_detector(modo="imagen", confianza=0.5):
    """Crea un HandLandmarker. modo: 'imagen' (fotos sueltas) o 'video' (con seguimiento)."""
    from mediapipe.tasks.python import BaseOptions, vision

    modos = {"imagen": vision.RunningMode.IMAGE, "video": vision.RunningMode.VIDEO}
    opciones = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=asegurar_detector()),
        running_mode=modos[modo],
        num_hands=1,
        min_hand_detection_confidence=confianza,
        min_hand_presence_confidence=confianza,
        min_tracking_confidence=confianza,
    )
    return vision.HandLandmarker.create_from_options(opciones)


def detectar(detector, frame_bgr, timestamp_ms=None):
    """Detecta una mano en un frame BGR de OpenCV.

    Devuelve (puntos, es_izquierda) con puntos float32 (21, 3) en coordenadas donde
    x e y están en la misma escala (x se multiplica por ancho/alto), o (None, None).
    """
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    if timestamp_ms is None:
        resultado = detector.detect(imagen)
    else:
        resultado = detector.detect_for_video(imagen, int(timestamp_ms))
    if not resultado.hand_landmarks:
        return None, None
    alto, ancho = frame_bgr.shape[:2]
    puntos = np.array([[p.x * ancho / alto, p.y, p.z] for p in resultado.hand_landmarks[0]],
                      dtype=np.float32)
    es_izquierda = resultado.handedness[0][0].category_name == "Left"
    return puntos, es_izquierda


def canonizar(puntos, es_izquierda):
    """Refleja la mano izquierda para que ambas manos se vean como la derecha."""
    puntos = puntos.copy()
    if es_izquierda:
        puntos[:, 0] = -puntos[:, 0]
    return puntos


def caracteristicas(ventanas):
    """Convierte ventanas de manos (B, T, 21, 3) en características (B, T, N_CARACT).

    Los pasos sin mano detectada vienen como NaN. Por paso se calcula:
      - forma de la mano: puntos relativos a la muñeca, divididos entre su tamaño (63);
      - trayectoria: desplazamiento de la muñeca respecto al primer paso con mano,
        en tamaños de mano (2). Es lo que distingue I de J, N de Ñ, etc.;
      - bandera 1/0 de si en ese paso hubo mano (1).
    Los pasos sin mano copian el paso válido más cercano.
    """
    v = np.asarray(ventanas, dtype=np.float32)
    b, t = v.shape[:2]
    validos = ~np.isnan(v[:, :, 0, 0])
    salida = np.zeros((b, t, N_CARACT), dtype=np.float32)
    hay = validos.any(axis=1)
    if not hay.any():
        return salida

    pasos = np.arange(t)[None, :]
    primero = validos.argmax(axis=1)[:, None]
    idx = np.where(validos, pasos, -1)
    idx = np.maximum.accumulate(idx, axis=1)
    idx = np.where(idx < 0, primero, idx)
    p = np.take_along_axis(v, idx[:, :, None, None], axis=1)
    p = np.nan_to_num(p)

    escala = np.linalg.norm(p[:, :, 9, :2] - p[:, :, 0, :2], axis=-1)
    escala = np.maximum(escala, 1e-6)
    forma = (p - p[:, :, :1]) / escala[:, :, None, None]
    origen = np.take_along_axis(p[:, :, 0, :2], primero[:, :, None], axis=1)
    escala_ref = np.take_along_axis(escala, primero, axis=1)[:, :, None]
    trayectoria = (p[:, :, 0, :2] - origen) / escala_ref

    salida[:] = np.concatenate(
        [forma.reshape(b, t, -1), trayectoria, validos[:, :, None].astype(np.float32)], axis=-1)
    salida[~hay] = 0.0
    return salida
