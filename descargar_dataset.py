"""Descarga los datasets LSM de Zenodo en dataset/ (no hace falta descomprimirlos).

Uso:
    python descargar_dataset.py                # estático + dinámico frontal (~6.5 GB)
    python descargar_dataset.py --todo         # también la vista de perfil del dinámico (+1.3 GB)

preprocesar_dataset.py lee los .7z directamente. Descomprimirlos no es necesario y,
en un disco duro mecánico, extraer las ~280 mil imágenes del estático tarda horas.

Datasets (licencia CC BY 4.0):
  - Estático (21 letras):  https://doi.org/10.5281/zenodo.10067509  (Morfín, 2023)
  - Dinámico (J K Ñ Q X Z): https://doi.org/10.5281/zenodo.14689869 (Navarrete-López y Lopez-Nava, 2025)
"""
import argparse
import hashlib
import os
import shutil
import urllib.request

ARCHIVOS = [
    # (nombre, url, md5, opcional)
    ("MSL-ABC.7z", "https://zenodo.org/records/10067509/files/MSL-ABC.7z?download=1",
     "d263efa295467f90a8a8ae93bb643f26", False),
    ("MSL-dynamic-signs-frontal-view.7z",
     "https://zenodo.org/records/14689869/files/MSL-dynamic-signs-frontal-view.7z?download=1",
     "58e7e416da6684ccbf413e89ef570411", False),
    ("MSL-dynamic-signs-profile.7z",
     "https://zenodo.org/records/14689869/files/MSL-dynamic-signs-profile.7z?download=1",
     "742f01c6427cf0136873bd075ee74118", True),
]


def md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 22), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(url, destino):
    tmp = destino + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        hecho = 0
        while bloque := r.read(1 << 20):
            f.write(bloque)
            hecho += len(bloque)
            if total:
                print(f"\r  {hecho / 2**30:.2f} / {total / 2**30:.2f} GB", end="", flush=True)
    print()
    os.replace(tmp, destino)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destino", default="dataset")
    parser.add_argument("--todo", action="store_true", help="incluir la vista de perfil del dataset dinámico")
    args = parser.parse_args()

    os.makedirs(args.destino, exist_ok=True)
    for nombre, url, suma, opcional in ARCHIVOS:
        if opcional and not args.todo:
            continue
        archivo = os.path.join(args.destino, nombre)
        if os.path.exists(archivo):
            print(f"{nombre}: ya existe, se verifica.")
        else:
            libre = shutil.disk_usage(args.destino).free / 2**30
            print(f"Descargando {nombre} (espacio libre: {libre:.0f} GB)...")
            descargar(url, archivo)
        if md5(archivo) != suma:
            raise SystemExit(f"MD5 incorrecto en {archivo}; bórralo y vuelve a correr el script.")
        print(f"  {nombre}: OK")
    print("Listo. Siguiente paso: python preprocesar_dataset.py")


if __name__ == "__main__":
    main()
