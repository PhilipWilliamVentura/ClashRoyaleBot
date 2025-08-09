import numpy as np
from actions import Actions
from card_detection import CardDetection
from troop_detection import TroopDetection

MAX_ALLIES = 10
MAX_ENEMIES = 10

class Env:
    def __init__(self):
        self.actions = Actions()
        self.troop_detection = TroopDetection()
        self.card_detection = CardDetection()
    
    def _get_state(self):
        elixir = self.actions.count_elixir()

        cards_in_hand = self.card_detection.run_card_detection()

        allies, enemies = self.troop_detection.run_troop_detection()

        # Normalize positions
        def normalize(units):
            return [(x / self.actions.WIDTH, y / self.actions.HEIGHT) for x, y in units]

        # Pad or truncate to fixed length
        def pad_units(units, max_units):
            units = normalize(units)
            if len(units) < max_units:
                units += [(0.0, 0.0)] * (max_units - len(units))
            return units[:max_units]

        ally_positions = pad_units(allies, MAX_ALLIES)
        enemy_positions = pad_units(enemies, MAX_ENEMIES)

        # Flatten positions
        final_ally = [coord for pos in ally_positions for coord in pos]
        final_enemy = [coord for pos in enemy_positions for coord in pos]
        
        state = np.array(cards_in_hand + [elixir / 10] + final_ally + final_enemy, dtype=np.float32)
        return state

    def _compute_reward(self, state):
        if state is None:
            return 0

        elixir = state[4] * 10

        # Reward based on elixir efficiency and enemy presence
        enemy_positions = state[1 + 2 * MAX_ALLIES:]
        enemy_presence = sum(enemy_positions)

        reward = -enemy_presence

        # Elixir efficiency
        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent = self.prev_elixir - elixir
            enemy_reduced = self.prev_enemy_presence - enemy_presence
            if elixir_spent > 0 and enemy_reduced > 0:
                reward += 2 * min(elixir_spent, enemy_reduced)  # tune this factor

        self.prev_elixir = elixir
        self.prev_enemy_presence = enemy_presence

        return reward
