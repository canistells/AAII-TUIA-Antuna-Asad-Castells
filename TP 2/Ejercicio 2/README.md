# Ej 2 - TP2 - Q-Learning
El objetivo de este laboratorio es entrenar agentes para resolver videojuegos sencillos usando Q-Learning y la librería PLE.

## Preparación del entorno

1. Crear el entorno virtual:
```bash
python3 -m venv env
```

2. Activar el entorno:
```bash
source env/bin/activate
```

3. Instalar dependencias:
```bash
pip3 install -r requirements.txt
```

## Estructura del proyecto
- `test_agent.py`: Script principal para probar agentes en FlappyBird.
- `agentes/`: Carpeta con implementaciones de agentes.
    - `base.py`: Clase base para todos los agentes.
    - `random_agent.py`: Agente que toma acciones aleatorias.
    - `manual_agent.py`: Agente que permite jugar manualmente usando la barra espaciadora.

## Uso

Ejecuta un agente especificando la ruta completa de la clase:

```bash
python test_agent.py --agent agentes.random_agent.RandomAgent
```

Para jugar manualmente (salta con la barra espaciadora):

```bash
python test_agent.py --agent agentes.manual_agent.ManualAgent
```

Puedes crear tus propios agentes en la carpeta `agentes/` siguiendo la interfaz de la clase base.

## Notas
- El entorno está configurado para FlappyBird por defecto.
- Los agentes reciben la instancia del juego y la lista de acciones posibles al inicializarse.
- Para agregar un nuevo agente, crea un archivo en `agentes/` y define una clase que herede de `Agent`.

## Ejercicio A: Q-Learning en Flappy Bird

**A.1 Completar el agente Q-Learning:**
- Edita el archivo `agentes/dq_agent.py` y completa las funciones `discretize_state` y `act` siguiendo las indicaciones y comentarios en el código.
- El objetivo es que el agente aprenda a jugar Flappy Bird usando Q-Learning tabular.

**A.2 Entrenar el agente:**
- Ejecuta el script de entrenamiento:
  ```bash
  python train_q_agent.py
  ```
- Esto entrenará el agente y guardará la Q-table en un archivo.

**A.3 Probar el agente entrenado:**
- Una vez entrenado, puedes testear el desempeño de tu agente ejecutando:
  ```bash
  python test_agent.py --agent agentes.dq_agent.QAgent
  ```
- Asegúrate de que el archivo de la Q-table guardado esté disponible como `flappy_birds_q_table.pkl` para que el agente lo cargue al iniciar.

## Ejercicio B: Aproximación de la Q-table con una Red Neuronal

**B.1 Entrenar una red neuronal para aproximar la Q-table:**
- Utiliza el script `train_q_nn.py` para entrenar una red neuronal que aproxime la Q-table obtenida en el ejercicio anterior.
- El script carga la Q-table y entrena una red usando TensorFlow/Keras. Completa los placeholders de arquitectura y entrenamiento según sea necesario.
- El modelo se guardará como un TensorFlow SavedModel.

**B.2 Crear un agente basado en la red neuronal:**
- Completa la función `act` en `agentes/nn_agent.py` para que el agente use la red neuronal entrenada para tomar decisiones.
- El agente debe transformar el estado al formato de entrada de la red y elegir la acción con mayor valor Q predicho.

**B.3 Probar el agente neuronal:**
- Ejecuta el script de testeo usando el agente neuronal:
  ```bash
  python test_agent.py --agent agentes.nn_agent.NNAgent
  ```
- Asegúrate de que el modelo guardado esté disponible en la ruta esperada (`flappy_q_nn_model/` por defecto).
