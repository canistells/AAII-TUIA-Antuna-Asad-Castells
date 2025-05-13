import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

# Cargar dataset
dataset_path = 'dataset'
X = np.load(os.path.join(dataset_path, 'rps_dataset.npy'))
y = np.load(os.path.join(dataset_path, 'rps_labels.npy'))

print(f"Shape de X: {X.shape}, Shape de y: {y.shape}")

# Escalar los datos (opcional pero recomendable)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Guardar el scaler por si después querés normalizar inputs nuevos
import joblib
os.makedirs('escalador', exist_ok=True)
joblib.dump(scaler, 'escalador/scaler.pkl')

# Separar en training y validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Crear el modelo
model = Sequential([
    Dense(32, activation='relu', input_shape=(42,)),  # 21 landmarks * 2 coordenadas = 42 entradas
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # 3 clases: piedra, papel o tijera
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Definir parámetros de entrenamiento
epochs = 30
batch_size = 32 
# Entrenar
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size
)

# Guardar modelo
os.makedirs('modelo', exist_ok=True)
model.save('modelo/rps_model.h5')
print("Modelo entrenado y guardado como rps_model.h5")

# Grafica la precisión y pérdida de entrenamiento y validación
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