import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Configuración del modelo
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
hands = vision.HandLandmarker.create_from_options(options)

# Carpetas y nombres de archivo
os.makedirs('dataset', exist_ok=True)
data_path = 'dataset/rps_dataset.npy'
labels_path = 'dataset/rps_labels.npy'

# Consultar al usuario si desea crear uno nuevo o continuar
modo = input("¿Querés crear un nuevo dataset (n) o editar uno existente (e)? [n/e]: ").strip().lower()

# Cargar el dataset existente o iniciar uno nuevo
if modo == 'e' and os.path.exists(data_path) and os.path.exists(labels_path):
    data = list(np.load(data_path))
    labels = list(np.load(labels_path))
    print(f"Dataset cargado. Contiene {len(labels)} muestras.")
else:
    data = []
    labels = []
    print("Nuevo dataset iniciado.")

# Constantes de dibujo
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

# Mapa de gestos, esto se extrajo de la documentacion de mediapipe
def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Dibuja los landmarks de la mano en la imagen RGB.
    
    Args:
        rgb_image: Imagen RGB en la que se dibujarán los landmarks.
        detection_result: Resultado de la detección de manos.
        
    Returns:
        Imagen RGB con los landmarks dibujados.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        height, width, _ = annotated_image.shape
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return annotated_image

def capture_landmarks(image):
    """
    Captura los landmarks de la mano en la imagen proporcionada.

    Args:
        image: Imagen de entrada en la que se detectarán los landmarks.

    Returns:
        landmarks: Un array de numpy con las coordenadas de los landmarks de la mano.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = hands.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result)
        cv2.imshow('Ventana', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        return np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32).flatten()
    else:
        return None
    
    
# Mapa de gestos
gesture_map = {
    ord('0'): 0,  # Piedra
    ord('1'): 1,  # Papel
    ord('2'): 2   # Tijeras
}

cap = cv2.VideoCapture(0)

print("Presioná 0 para Piedra, 1 para Papel, 2 para Tijeras, y 'q' para salir.")

while True:
    # Leer el frame
    ret, frame = cap.read()
    if not ret:
        break
    
    rows, cols, _ = frame.shape
    num_rows, num_cols = 4, 4
    row_step = rows // num_rows
    col_step = cols // num_cols
    
    # Dibujar la cuadrícula
    for i in range(1, num_rows):
        y = i * row_step
        cv2.line(frame, (0, y), (cols, y), (0, 255, 0), 1)

    for j in range(1, num_cols):
        x = j * col_step
        cv2.line(frame, (x, 0), (x, rows), (0, 255, 0), 1)

    cv2.imshow('Grabando Dataset', frame)

    # Capturar la imagen y esperar por la tecla
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # Capturar landmarks si se presiona una tecla válida
    if key in gesture_map:
        landmarks = capture_landmarks(frame)
        if landmarks is not None:
            data.append(landmarks)
            labels.append(gesture_map[key])
            print(f"Capturado gesto: {gesture_map[key]} (Total capturas: {len(labels)})")
        else:
            print("No se detectó ninguna mano.")
            
# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# Guardar los datos
np.save(data_path, np.array(data))
np.save(labels_path, np.array(labels))
print("Grabación finalizada. Dataset guardado en 'dataset/'.")