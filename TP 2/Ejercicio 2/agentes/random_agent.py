from agentes.base import Agent
import numpy as np

class RandomAgent(Agent):
    """Agente que elige acciones aleatorias."""
    def __init__(self, actions, game=None):
        super().__init__(actions, game)
    def act(self, state):
        return np.random.choice(self.actions)
