# Agente LSM: reconocimiento del abecedario de Lengua de Señas Mexicana

Reconoce en tiempo real, con la webcam, las 27 letras del abecedario LSM: 21 estáticas y 6 con movimiento (J, K, Ñ, Q, X, Z). Las letras que sostienes se van escribiendo en pantalla.

## Cómo funciona

1. **MediaPipe Hand Landmarker** detecta 21 puntos de la mano en cada frame.
2. Los puntos se normalizan: quedan relativos a la muñeca y escalados al tamaño de la mano, y la mano izquierda se refleja como si fuera derecha. Así no importan el fondo, la luz, la distancia a la cámara ni con qué mano señas.
3. Un modelo pequeño (Conv1D + GRU, ~115 mil parámetros) clasifica una ventana de 1 s (15 muestras). Ve la forma de la mano y la trayectoria de la muñeca, y con eso distingue I de J, N de Ñ, etc.

### Cambios respecto a la versión anterior (CNN sobre píxeles)

| Problema | Antes | Ahora |
|---|---|---|
| Fondo | La CNN aprendía del fondo verde del dataset y fallaba con la webcam | Usa los puntos de la mano; el fondo no influye |
| Letras dinámicas | Los videos que estaban directo en `train/` se ignoraban, así que `clases.npy` tenía solo 21 letras. Además, un frame suelto no distingue I/J ni N/Ñ | Procesa todos los videos y modela el movimiento |
| Posición | La mano tenía que quedar dentro de un cuadro fijo | La mano se detecta en cualquier parte de la imagen |
| Texto | `cv2.putText` no dibuja Ñ ni acentos | Texto con Pillow |
| Inestabilidad | Cada frame daba una predicción distinta | Promedio de las últimas 5 predicciones y una seña sostenida 1 s para escribirla |
| Entrenamiento | Sin early stopping, validaba con el set de prueba | Validación aparte, muestreo balanceado por letra, aumento de datos, early stopping y reporte por letra |
| Backend de Keras | Se rompía si `~/.keras/keras.json` pedía torch | Forzado a TensorFlow |

## Instalación (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

El modelo de MediaPipe (`modelos/hand_landmarker.task`, ~8 MB) se descarga solo la primera vez.

## Uso rápido (modelo ya entrenado)

El repo incluye el modelo entrenado (`modelos/modelo_lsm.keras`, 1.4 MB). Basta con instalar y correr:

```powershell
python app_completo.py
```

## Resultados

Evaluado con las personas de la partición `test` del dataset, que el modelo no vio al entrenar:

- **93.8%** de exactitud en las 27 letras (27,624 ventanas).
- Dinámicas: J, K y Ñ 100%, Q 96.8%, X 94.9%, Z 81.1%.
- Confusiones principales, entre señas muy parecidas: V↔U (84%), M↔N (84%), R↔U (87%), D↔R (88%).

En una PC con Ryzen 5 5600X y solo CPU, preprocesar tomó ~30 min (279,716 imágenes + 1,242 videos) y entrenar ~12 min (60 épocas).

## Reentrenar

```powershell
python descargar_dataset.py --todo     # ~7.8 GB de Zenodo (.7z); no hay que descomprimirlos
python preprocesar_dataset.py          # lee los .7z y extrae los puntos -> datos/landmarks.npz
python entrenar_modelo.py              # entrena -> modelos/modelo_lsm.keras
python app_completo.py                 # webcam
```

- App: `--camara 1` usa otra cámara, `--umbral 0.6` fija la confianza mínima, `--sostener 1.0` los segundos para escribir una letra estática, `--sostener-dinamica 0.3` lo mismo para J K Ñ Q X Z y `--video archivo.mp4` prueba con un video.
- Teclas: `q`/`Esc` salir, `Espacio` espacio, `Retroceso` borrar, `c` limpiar.
- `--origen` acepta carpetas y `.7z`/`.zip`/`.tar`, uno o varios: `python preprocesar_dataset.py --origen a.7z carpeta_b`. Los comprimidos se leen con el `tar` de Windows, sin extraerlos. En carpetas se buscan todas las subcarpetas `train/` y `test/`. En imágenes, la letra sale de la carpeta. En videos sale del nombre (`S<id>-<Letra>-<Vista>-<Rep>`) o, si no se puede, de la carpeta.
- Al terminar, `entrenar_modelo.py` muestra la exactitud sobre las personas del set `test` (no vistas en entrenamiento) y la confusión principal de cada letra.

## Datasets (CC BY 4.0)

- Estático: R. Morfín, *Mexican Sign Language Alphabet (static signs only)*, Zenodo, 2023. [10.5281/zenodo.10067509](https://doi.org/10.5281/zenodo.10067509)
- Dinámico: J. A. Navarrete-López, I. H. Lopez-Nava, *Mexican Sign Language Alphabet (dynamic signs only)*, Zenodo, 2025. [10.5281/zenodo.14689869](https://doi.org/10.5281/zenodo.14689869)

## Limitaciones

- Reconoce una mano a la vez y letra por letra; no reconoce palabras ni frases de LSM.
- El dataset se grabó con buena luz y de frente o a 45°. Con poca luz o la mano muy de lado, MediaPipe puede no detectarla.
