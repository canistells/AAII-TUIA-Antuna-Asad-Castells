import numpy as np
import os
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Verificar si hay GPU disponible
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detectada:", gpus[0].name)
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No se detectó GPU. Usando CPU.")

# Rutas base
escalador_dir = 'escalador'
modelo_dir = 'modelo'

data_path = 'dataset/rps_dataset.npy'
labels_path = 'dataset/rps_labels.npy'

# Cargar dataset
X = np.load(data_path)
y = np.load(labels_path)

print(f"Shape de X: {X.shape}, Shape de y: {y.shape}")

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar el scaler
os.makedirs(escalador_dir, exist_ok=True)
scaler_path = os.path.join(escalador_dir, 'scaler_gpu.pkl')
joblib.dump(scaler, scaler_path)

# Separar en training y validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42, stratify=y
)

# Crear el modelo
model = Sequential([
    Dense(256, activation='relu', input_shape=(42,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
epochs = 2
batch_size = 16

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size
)

# Guardar el modelo
os.makedirs(modelo_dir, exist_ok=True)
model_path = os.path.join(modelo_dir, 'rps_model_gpu.h5')
model.save(model_path)
print(f"Modelo entrenado y guardado como {model_path}")

# Graficar precisión y pérdida
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()