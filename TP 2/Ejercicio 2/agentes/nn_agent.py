from agentes.base import Agent
import numpy as np
import tensorflow as tf

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model.keras'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)
        self.player_y_bins = np.linspace(0, 512, 20)  # Supone altura de pantalla 512px
        self.player_vel_bins = np.arange(0, 11, 10)  # Velocidad de 0 a 11
        self.pipe_dist_bins = np.linspace(0, 288, 10) # Supone ancho de pantalla 288px
        self.pipe_y_bins = np.linspace(0, 512, 20)    # Altura de los tubos
    
    def discretize_state(self, state):
        """
        Discretiza el estado continuo en un estado discreto (tupla).
        Ajusta los bins según las características del entorno Flappy Bird.
        El estado esperado es un dict con las siguientes claves:
        - 'player_y'
        - 'player_vel'
        - 'next_pipe_dist_to_player'
        - 'next_pipe_top_y'
        - 'next_pipe_bottom_y'
        - 'next_next_pipe_dist_to_player'
        - 'next_next_pipe_top_y'
        - 'next_next_pipe_bottom_y'
        """
        # Extraer variables relevantes
        player_y = state['player_y']
        player_vel = state['player_vel']
        next_pipe_dist = state['next_pipe_dist_to_player']
        next_pipe_top_y = state['next_pipe_top_y']
        next_pipe_bottom_y = state['next_pipe_bottom_y']
        # next_next_pipe_dist = state['next_next_pipe_dist_to_player']
        # next_next_pipe_top_y = state['next_next_pipe_top_y']
        # next_next_pipe_bottom_y = state['next_next_pipe_bottom_y']

        # Discretización usando los bins definidos
        player_y_bin = np.digitize(player_y, self.player_y_bins)
        player_vel_bin = np.digitize(player_vel, self.player_vel_bins)
        pipe_dist_bin = np.digitize(next_pipe_dist, self.pipe_dist_bins)
        pipe_top_bin = np.digitize(next_pipe_top_y, self.pipe_y_bins)
        pipe_bot_bin = np.digitize(next_pipe_bottom_y, self.pipe_y_bins)
        # next_pipe_dist_bin = np.digitize(next_next_pipe_dist, self.pipe_dist_bins)
        # next_pipe_top_bin = np.digitize(next_next_pipe_top_y, self.pipe_y_bins)
        # next_pipe_bot_bin = np.digitize(next_next_pipe_bottom_y, self.pipe_y_bins)

        return (
            player_y_bin,
            player_vel_bin,
            pipe_dist_bin,
            pipe_top_bin,
            pipe_bot_bin
            # next_pipe_dist_bin,
            # next_pipe_top_bin,
            # next_pipe_bot_bin
        )

    def act(self, state):
        discrete_state = self.discretize_state(state)
        array_state = np.array(discrete_state).reshape(1, -1)
        q_values = self.model.predict(array_state, verbose=0)
        # print(f"Q-values: {q_values}")
        return self.actions[np.argmax(q_values)]
