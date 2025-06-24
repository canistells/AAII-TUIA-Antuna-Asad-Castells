from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle
import random

class QAgent(Agent):
    """
    Agente de Q-Learning.
    Completar la discretización del estado y la función de acción.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
                 epsilon=0.0, epsilon_decay=0.995, min_epsilon=0.05, load_q_table_path="flappy_birds_q_table.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        # Parámetros de discretización según el entorno Flappy Bird
        # Estos valores pueden ajustarse según el rango de cada variable
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
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # max_q = np.max(q_values)
            # if max_q == 0:
            #     #print(f"Advertencia: Q-values para el estado {discrete_state} son todos cero. Eligiendo acción aleatoria.")
            #     # Si no hay Q-values, elegir una acción aleatoria
            #     return self.actions[0]
            q_values = self.q_table[discrete_state]
            return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
