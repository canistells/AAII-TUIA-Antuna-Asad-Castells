class Agent:
    """Clase base para agentes."""
    def __init__(self, actions, game=None):
        self.actions = actions
        self.game = game

    def reset(self):
        """Reinicia el estado interno del agente (si es necesario)."""
        pass

    def act(self, state):
        """Devuelve la acción a tomar dado el estado actual."""
        raise NotImplementedError
