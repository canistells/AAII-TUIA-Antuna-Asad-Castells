# record-dataset.py

import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

"""# Inicializamos MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.HandLandmarker.create_from_options(
    mp_hands.HandLandmarkerOptions(
        num_hands=1,
        min_detection_confidence=0.5
    )
)"""

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
hands = vision.HandLandmarker.create_from_options(options)

# Crear listas para almacenar datos
data = []
labels = []

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def capture_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = hands.detect(mp_image)

    if result.hand_landmarks:
        # Tomamos la primera mano detectada
        landmarks = result.hand_landmarks[0]
        # STEP 5: Process the classification result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result)
        cv2.imshow('Ventana',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        # Convertimos a un array de (x,y) para cada punto
        return np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32).flatten()
    else:
        return None

# Mapeo de teclas a etiquetas
gesture_map = {
    ord('0'): 0,  # Piedra
    ord('1'): 1,  # Papel
    ord('2'): 2   # Tijeras
}

# Abrimos la cámara
cap = cv2.VideoCapture(0)

print("Presioná 0 para Piedra, 1 para Papel, 2 para Tijeras, y 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.flip(frame, 1)  # Espejar la imagen
    cv2.imshow('Grabando Dataset', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key in gesture_map:
        landmarks = capture_landmarks(frame)
        if landmarks is not None:
            data.append(landmarks)
            labels.append(gesture_map[key])
            print(f"Capturado gesto: {gesture_map[key]} (Total capturas: {len(labels)})")
        else:
            print("No se detectó ninguna mano.")

cap.release()
cv2.destroyAllWindows()

# Convertir listas a arrays
data = np.array(data)
labels = np.array(labels)

# Guardar dataset
os.makedirs('dataset', exist_ok=True)
np.save('dataset/rps_dataset.npy', data)
np.save('dataset/rps_labels.npy', labels)

print("Grabación finalizada. Dataset guardado en 'dataset/'.")