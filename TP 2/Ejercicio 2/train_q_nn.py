import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- Configuración de GPU ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table.pkl'
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

print(q_table)
# --- Preparar datos para entrenamiento ---
X = []
y = []

# Convertir la Q-table a un formato adecuado para entrenamiento
for state, q_values in q_table.items():
    X.append(state)     # Usamos solo las primeras 5 features
    y.append(q_values)  # Solo 2 acciones válidas: [do nothing, flap]

X = np.array(X)
y = np.array(y)

# Dividir en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


state_size = X.shape[1]   # 5
action_size = y.shape[1]  # 2

# --- Modelo de Red Neuronal ---
model = keras.Sequential([
    layers.Input(shape=(state_size,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(action_size)
])

# --- Compilación del modelo ---
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
model.summary()

# --- Callbacks ---
MODEL_SAVE_PATH = 'checkpoints/flappy_q_model.keras'
CHECKPOINT_PATH = 'checkpoints/flappy_q_cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

model_checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    save_weights_only=False,
    monitor='val_mse',
    mode='min',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

# --- Entrenamiento ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[model_checkpoint_callback, early_stopping, reduce_lr_callback],
    verbose=1
)

# --- Plot resultados ---
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.history['mse'], label='Entrenamiento MSE')
plt.plot(history.history['val_mse'], label='Validación MSE')
plt.legend()
plt.title('MSE durante entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('MSE')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento Loss')
plt.plot(history.history['val_loss'], label='Validación Loss')
plt.legend()
plt.title('Loss durante entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig('entrenamiento_resultados.png')
plt.show()

# --- Guardar modelo final ---
model.save('flappy_q_nn_model.keras')
print('Modelo guardado como flappy_q_nn_model.keras')