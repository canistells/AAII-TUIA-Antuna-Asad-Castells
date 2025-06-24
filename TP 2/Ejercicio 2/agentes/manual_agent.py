from agentes.base import Agent
import pygame

class ManualAgent(Agent):
    """Agente que toma acciones manualmente: salta al presionar la barra espaciadora."""
    def __init__(self, actions, game=None):
        super().__init__(actions, game)
        self.jump_action = actions[0]
        self.noop_action = actions[1]
        self._space_was_pressed = False

    def act(self, state):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        jump = False
        if not self._space_was_pressed and keys[pygame.K_SPACE]:
            jump = True
        self._space_was_pressed = keys[pygame.K_SPACE]
        if jump:
            return self.jump_action
        return self.noop_action
