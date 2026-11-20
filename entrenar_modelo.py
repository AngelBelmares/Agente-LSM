"""Entrena el clasificador de señas a partir de datos/landmarks.npz.

Uso:
    python entrenar_modelo.py [--datos datos/landmarks.npz] [--epocas 60]

Cada muestra es una ventana de PASOS pasos (1 s a 15 fps) de puntos de la mano:
  - imágenes estáticas: la misma pose repetida con pequeño temblor aleatorio;
  - videos dinámicos: ventanas deslizantes, priorizando los tramos con movimiento.
Guarda modelos/modelo_lsm.keras y modelos/clases.json.
"""
import argparse
import json
import os

import numpy as np

import lsm_comun as lc

SEMILLA = 42


class Muestras:
    """Índice de ventanas (secuencia, inicio) sobre el arreglo plano de puntos."""

    def __init__(self, datos, mascara):
        self.puntos = datos["puntos"]
        ids = np.flatnonzero(mascara)
        self.inicio = datos["inicio"][ids]
        self.largo = datos["largo"][ids]
        self.etiqueta = datos["etiqueta"][ids]
        self.es_video = datos["es_video"][ids]

    def ventana(self, i, desde, velocidad=1.0):
        """Devuelve (PASOS, 21, 3) para la secuencia i empezando en `desde`.

        Fuera de la secuencia se repite el primer/último paso (mano quieta antes y
        después de la seña). velocidad < 1 simula señar más lento."""
        ini, n = self.inicio[i], self.largo[i]
        idx = np.rint(desde + np.arange(lc.PASOS) * velocidad).astype(np.int64)
        return self.puntos[ini + np.clip(idx, 0, n - 1)]

    def ventana_aleatoria(self, i, rng):
        """Ventana de video con velocidad y posición aleatorias: los videos del dataset
        duran ~0.6 s y en la webcam la seña puede caer en cualquier parte del segundo."""
        n = self.largo[i]
        velocidad = rng.uniform(0.5, 1.25)
        largo_efectivo = (n - 1) / velocidad + 1
        holgura = largo_efectivo - lc.PASOS
        desde = rng.uniform(min(0.0, holgura), max(0.0, holgura))
        return self.ventana(i, desde * velocidad, velocidad)

    def indices_ventanas(self, paso_video=2, fraccion_movimiento=0.7):
        """Lista de (secuencia, inicio). En videos de letras dinámicas descarta las
        ventanas más quietas (el principio/fin del video suele ser una pose estática
        que se confundiría con otra letra)."""
        salida = []
        for i in range(len(self.largo)):
            n = self.largo[i]
            if not self.es_video[i]:
                salida.append((i, 0))
                continue
            if n <= lc.PASOS:  # video corto: la seña en distintas posiciones de la ventana
                salida.extend((i, int(s)) for s in range(n - lc.PASOS, 1, paso_video))
                continue
            inicios = np.arange(0, n - lc.PASOS + 1, paso_video)
            ventanas = np.stack([self.ventana(i, s) for s in inicios])
            validos = (~np.isnan(ventanas[:, :, 0, 0])).mean(axis=1)
            inicios, ventanas = inicios[validos >= 0.6], ventanas[validos >= 0.6]
            if len(inicios) == 0:
                continue
            if self.etiqueta[i] in lc.LETRAS_DINAMICAS and len(inicios) > 2:
                c = lc.caracteristicas(ventanas)
                movimiento = np.abs(np.diff(c[:, :, :-1], axis=1)).mean(axis=(1, 2))
                umbral = np.quantile(movimiento, 1 - fraccion_movimiento)
                inicios = inicios[movimiento >= umbral]
            salida.extend((i, int(s)) for s in inicios)
        return salida


def aumentar(ventanas, es_estatica, rng):
    """Rotación leve, temblor y pasos perdidos sobre (B, T, 21, 3)."""
    v = ventanas.copy()
    b = len(v)
    ang = rng.uniform(-np.pi / 12, np.pi / 12, size=b)  # ±15°
    cos, sin = np.cos(ang)[:, None, None], np.sin(ang)[:, None, None]
    muneca = v[:, :1, :1, :2]
    xy = v[..., :2] - muneca
    v[..., 0] = xy[..., 0] * cos - xy[..., 1] * sin
    v[..., 1] = xy[..., 0] * sin + xy[..., 1] * cos
    v[..., :2] += muneca
    escala = np.nanmedian(np.linalg.norm(v[:, :, 9, :2] - v[:, :, 0, :2], axis=-1), axis=1)
    escala = np.nan_to_num(escala, nan=0.1)[:, None, None, None]
    v += rng.normal(0, 0.03, size=v.shape).astype(np.float32) * escala
    # Desplazamiento lento de toda la mano en las estáticas (la gente no se queda quieta).
    deriva = rng.normal(0, 0.05, size=(b, 1, 1, 2)) * escala[..., :2]
    deriva = deriva * np.linspace(0, 1, lc.PASOS)[None, :, None, None]
    v[..., :2] += np.where(es_estatica[:, None, None, None], deriva, 0).astype(np.float32)
    perdidos = rng.random((b, lc.PASOS)) < 0.05
    perdidos[:, 0] = False
    v[perdidos] = np.nan
    return v


def crear_lote(muestras, ventanas_idx, clases, rng=None):
    if rng is None:
        vent = np.stack([muestras.ventana(i, s) for i, s in ventanas_idx])
    else:
        vent = np.stack([muestras.ventana_aleatoria(i, rng) if muestras.es_video[i] else muestras.ventana(i, s)
                         for i, s in ventanas_idx])
    seq = np.array([i for i, _ in ventanas_idx])
    if rng is not None:
        vent = aumentar(vent, ~muestras.es_video[seq], rng)
    x = lc.caracteristicas(vent)
    y = np.array([clases.index(muestras.etiqueta[i]) for i in seq])
    return x, y


def construir_modelo(n_clases):
    from tensorflow import keras
    from tensorflow.keras import layers

    entrada = keras.Input(shape=(lc.PASOS, lc.N_CARACT))
    x = layers.LayerNormalization()(entrada)
    x = layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    x = layers.GRU(96)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(n_clases, activation="softmax")(x)
    modelo = keras.Model(entrada, salida)
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def evaluar(modelo, muestras, clases, titulo):
    idx = muestras.indices_ventanas(paso_video=3)
    if not idx:
        return
    x, y = crear_lote(muestras, idx, clases)
    pred = modelo.predict(x, batch_size=512, verbose=0).argmax(axis=1)
    print(f"\n=== {titulo}: exactitud por ventana {100 * (pred == y).mean():.1f}% ({len(y)} ventanas)")
    # Por archivo: voto de mayoría entre sus ventanas (así se comporta la app con suavizado).
    seq = np.array([i for i, _ in idx])
    aciertos = {}
    for s in np.unique(seq):
        votos = np.bincount(pred[seq == s], minlength=len(clases))
        aciertos[s] = votos.argmax() == y[seq == s][0]
    print(f"    exactitud por archivo {100 * np.mean(list(aciertos.values())):.1f}% ({len(aciertos)} archivos)")
    print("    por letra (aciertos/total, confusión principal):")
    for c, letra in enumerate(clases):
        m = y == c
        if not m.any():
            continue
        errores = pred[m][pred[m] != c]
        confusion = f"  -> {clases[np.bincount(errores).argmax()]}" if len(errores) else ""
        print(f"      {letra}: {100 * (pred[m] == c).mean():5.1f}% de {m.sum()}{confusion}")


def cargar_datos(rutas):
    """Une varios .npz; los índices de inicio se desplazan para apuntar al arreglo unido."""
    partes = [dict(np.load(r)) for r in rutas]
    desplazamiento, inicios = 0, []
    for p in partes:
        inicios.append(p["inicio"] + desplazamiento)
        desplazamiento += len(p["puntos"])
    unido = {k: np.concatenate([p[k] for p in partes]) for k in partes[0] if k != "inicio"}
    unido["inicio"] = np.concatenate(inicios)
    return unido


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", nargs="+", default=[os.path.join("datos", "landmarks.npz")],
                        help="uno o varios .npz de preprocesar_dataset.py (p. ej. estático y dinámico)")
    parser.add_argument("--epocas", type=int, default=60)
    parser.add_argument("--lote", type=int, default=256)
    parser.add_argument("--por-clase", type=int, default=1500,
                        help="ventanas por letra en cada época (muestreo balanceado)")
    args = parser.parse_args()

    from tensorflow import keras
    keras.utils.set_random_seed(SEMILLA)
    rng = np.random.default_rng(SEMILLA)

    datos = cargar_datos(args.datos)
    clases = sorted(set(datos["etiqueta"].tolist()))
    print(f"Clases ({len(clases)}): {' '.join(clases)}")
    faltan = [l for l in lc.ABECEDARIO if l not in clases]
    if faltan:
        print(f"Aviso: faltan {faltan} en los datos; el modelo no las reconocerá.")

    es_train = datos["split"] == "train"
    # Validación: 10% de las secuencias de train (la partición test son otras personas
    # y se reserva para la evaluación final).
    val = es_train & (rng.random(len(es_train)) < 0.10)
    train = Muestras(datos, es_train & ~val)
    validacion = Muestras(datos, val)
    prueba = Muestras(datos, datos["split"] == "test")

    idx_train = train.indices_ventanas()
    por_clase = {}
    for k, (i, _) in enumerate(idx_train):
        por_clase.setdefault(train.etiqueta[i], []).append(k)
    print("Ventanas de entrenamiento por letra:",
          ", ".join(f"{l}={len(por_clase.get(l, []))}" for l in clases))

    class Lotes(keras.utils.PyDataset):
        """Cada época toma el mismo número de ventanas por letra, con aumento nuevo."""

        def __init__(self):
            super().__init__()
            self.on_epoch_end()

        def on_epoch_end(self):
            elegidos = np.concatenate([rng.choice(ks, size=args.por_clase, replace=True)
                                       for ks in por_clase.values()])
            rng.shuffle(elegidos)
            self.orden = elegidos

        def __len__(self):
            return int(np.ceil(len(self.orden) / args.lote))

        def __getitem__(self, n):
            ks = self.orden[n * args.lote:(n + 1) * args.lote]
            return crear_lote(train, [idx_train[k] for k in ks], clases, rng)

    datos_val = crear_lote(validacion, validacion.indices_ventanas(paso_video=3), clases) \
        if len(validacion.largo) else None

    os.makedirs(lc.DIR_MODELOS, exist_ok=True)
    modelo = construir_modelo(len(clases))
    modelo.summary()
    monitor = "val_loss" if datos_val else "loss"
    modelo.fit(
        Lotes(),
        epochs=args.epocas,
        validation_data=datos_val,
        verbose=2,  # una línea por época
        callbacks=[
            keras.callbacks.EarlyStopping(monitor=monitor, patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3),
            keras.callbacks.ModelCheckpoint(lc.RUTA_MODELO, monitor=monitor, save_best_only=True),
        ],
    )
    modelo.save(lc.RUTA_MODELO)
    with open(lc.RUTA_CLASES, "w", encoding="utf-8") as f:
        json.dump({"clases": clases, "pasos": lc.PASOS, "fps": lc.FPS_MUESTREO}, f, ensure_ascii=False)
    print(f"\nModelo guardado en {lc.RUTA_MODELO}")

    if len(prueba.largo):
        evaluar(modelo, prueba, clases, "Prueba (personas no vistas)")
    else:
        print("No hay partición test; se omite la evaluación final.")


if __name__ == "__main__":
    main()
