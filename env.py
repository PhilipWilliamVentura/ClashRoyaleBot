import numpy as np
import threading
import time
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

        self.num_cards = 4
        self.grid_width = 18
        self.grid_height = 28

        self.game_end_flag = None
        self._game_end_thread = None
        self._game_end_thread_stop = threading.Event()
    
    def reboot(self):
        time.sleep(3)
        self.game_end_flag = None
        self._game_end_thread_stop.clear()
        self._game_end_thread = threading.Thread(target=self._game_end_detector, daemon=True)
        self._game_end_thread.start()
        state = self._get_state()
        return state
    
    def finish(self):
        self._game_end_thread_stop.set()
        if self._game_end_thread:
            self._game_end_thread.join()
    
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

    def get_all_actions(self):
        actions = [
            [card, x / (self.grid_width - 1), y / (self.grid_height - 1)]
            for card in range(self.num_cards)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
        ]
        actions.append([-1, 0, 0])  # No-op action
        return actions

    def _game_end_detector(self):
        while not self._game_end_thread_stop.is_set():
            result = self.actions.detect_winner()
            if result:
                self.game_end_flag = result
                break
        time.sleep(1)
