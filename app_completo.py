import cv2
import numpy as np
import tensorflow as tf

modelo = tf.keras.models.load_model('modelo_lsm.h5')
clases = np.load('clases.npy')

cap = cv2.VideoCapture(0)
contador_frames = 0
texto = "Coloca tu mano en el cuadro..."
color = (0, 165, 255)

while True:
    ret, frame = cap.read()
    if not ret: break
        
    contador_frames += 1
    cv2.rectangle(frame, (100, 100), (356, 356), (0, 255, 0), 2)
    
    if contador_frames % 3 == 0:
        roi = frame[100:356, 100:356]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, (128, 128))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, 128, 128, 3))
        
        prediccion = modelo(roi_reshaped, training=False).numpy()[0]
        indice_clase = np.argmax(prediccion)
        confianza = prediccion[indice_clase]
        letra = str(clases[indice_clase])

        if confianza > 0.45:
            texto = f"Sena: {letra} ({confianza*100:.1f}%)"
            color = (0, 255, 0)
        else:
            texto = "Buscando seña..."
            color = (0, 165, 255)

    cv2.putText(frame, texto, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Reconocimiento LSM Completo", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()