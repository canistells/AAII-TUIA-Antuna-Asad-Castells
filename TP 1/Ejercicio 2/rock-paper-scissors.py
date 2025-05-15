import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Cargar modelo y scaler
model = tf.keras.models.load_model('modelo/rps_model_gpu.h5')
scaler = joblib.load('escalador/scaler_gpu.pkl')

# Configuración del modelo
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
hands = vision.HandLandmarker.create_from_options(options)

# Nombres de clases
class_names = ['Piedra', 'Papel', 'Tijera']

# Capturar video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Leer frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convertir a formato de imagen de mediapipe
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Detectar manos
    result = hands.detect(image)

    if result.hand_landmarks:
        for i in range(len(result.hand_landmarks)):
            # Tomamos landmarks de la primera mano detectada
            landmarks = result.hand_landmarks[i]

            # Extraer x, y de cada landmark
            landmark_array = []
            xs, ys = [], []
            for landmark in landmarks:
                landmark_array.append(landmark.x)
                landmark_array.append(landmark.y)
                xs.append(landmark.x * frame.shape[1])  # frame.shape[1] = width
                ys.append(landmark.y * frame.shape[0])  # frame.shape[0] = height

            # Dibujar el bounding box alrededor de la mano
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

            # Normalizar landmarks
            landmark_array = np.array(landmark_array).reshape(1, -1)

            # Escalar segun el scaler
            landmark_array = scaler.transform(landmark_array)

            # Predecir
            prediction = model.predict(landmark_array)
            class_idx = np.argmax(prediction)
            prob = np.max(prediction)
            class_name = class_names[class_idx]

            # Poner el texto
            cv2.putText(frame, f'{class_name}, {prob:.2f}', (x_min - 20, y_min - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar video
    cv2.imshow('Rock-Paper-Scissors Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()