import os
import shutil
import cv2

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_completo")

def crear_estructura():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

def buscar_y_procesar():
    print("Buscando carpetas del dataset descomprimido...")
    total_imagenes = 0
    total_videos = 0

    for root, dirs, files in os.walk(BASE_DIR):
        # Ignorar la carpeta destino para no entrar en bucle
        if "dataset_completo" in root or ".venv" in root:
            continue

        # Detectar carpetas de tipo 'train' o 'test'
        folder_name = os.path.basename(root).lower()
        if folder_name in ["train", "test"]:
            split = folder_name
            for item in os.listdir(root):
                item_path = os.path.join(root, item)
                
                # Caso 1: Es subcarpeta con el nombre de la letra (ej. A, B, C...)
                if os.path.isdir(item_path):
                    letra = item.upper()
                    path_letra_destino = os.path.join(OUTPUT_DIR, split, letra)
                    os.makedirs(path_letra_destino, exist_ok=True)

                    for file in os.listdir(item_path):
                        file_path = os.path.join(item_path, file)
                        
                        # Procesar Imágenes Estáticas
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            # Generar nombre único para evitar colisiones entre grupos A, B y C
                            hash_prefix = str(abs(hash(root)))[:5]
                            dst = os.path.join(path_letra_destino, f"{hash_prefix}_{file}")
                            shutil.copy(file_path, dst)
                            total_imagenes += 1
                            
                        # Procesar Videos Dinámicos (J, K, Ñ, Q, X, Z)
                        elif file.lower().endswith(('.mp4', '.avi', '.mov')):
                            partes = file.split('-')
                            letra_video = partes[1].upper() if len(partes) >= 2 else letra
                            path_video_dest = os.path.join(OUTPUT_DIR, split, letra_video)
                            os.makedirs(path_video_dest, exist_ok=True)

                            cap = cv2.VideoCapture(file_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            
                            if total_frames > 0:
                                indices = [
                                    int(total_frames * 0.3),
                                    int(total_frames * 0.5),
                                    int(total_frames * 0.7),
                                    int(total_frames * 0.85),
                                    total_frames - 1
                                ]
                                frame_idx = 0
                                while cap.isOpened():
                                    ret, frame = cap.read()
                                    if not ret: break
                                    if frame_idx in indices:
                                        img_name = f"vid_{file}_frame_{frame_idx}.jpg"
                                        cv2.imwrite(os.path.join(path_video_dest, img_name), frame)
                                        total_imagenes += 1
                                    frame_idx += 1
                            cap.release()
                            total_videos += 1

    print(f"\n¡Preprocesamiento finalizado!")
    print(f"Imágenes organizadas: {total_imagenes}")
    print(f"Videos dinámicos procesados: {total_videos}")

if __name__ == "__main__":
    crear_estructura()
    buscar_y_procesar()