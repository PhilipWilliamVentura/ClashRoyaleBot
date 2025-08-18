import numpy as np
import threading
import time
from actions import Actions
from card_detection import CardDetection
from tower_detection import TowerDetection
from troop_detection import TroopDetection


MAX_ALLIES = 10
MAX_ENEMIES = 10

class Env:
    def __init__(self):
        self.actions = Actions()
        self.troop_detection = TroopDetection()
        self.tower_detection = TowerDetection()
        self.card_detection = CardDetection()
        self.state_size = 5 + 2 * (MAX_ALLIES + MAX_ENEMIES)  # cards in hand + elixir + ally positions + enemy positions

        self.num_cards = 4
        self.grid_width = 18
        self.grid_height = 14

        self.available_action = self.get_all_actions()
        self.action_size = len(self.available_action)

        self.game_end_flag = None
        self._game_end_thread = None
        self._game_end_thread_stop = threading.Event()

        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.prev_enemy_tower = None
    
    def reboot(self):
        time.sleep(3)
        self.game_end_flag = None
        self._game_end_thread_stop.clear()
        self._game_end_thread = threading.Thread(target=self._game_end_detector, daemon=True)
        self._game_end_thread.start()
        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.prev_ally_tower, self.prev_enemy_tower = self.tower_detection.run_tower_detection() 
        return self._get_state()
    
    def finish(self):
        self._game_end_thread_stop.set()
        if self._game_end_thread:
            self._game_end_thread.join()
    
    def step(self, valid_action):
        if self.game_end_flag:
            valid_action = len(self.available_action) - 1
            done = True
            reward = self._compute_reward(self._get_state())
            result = self.game_end_flag
            if result == "victory":
                reward += 100
                print("Victory detected - game end")
            elif result == "defeat":
                reward -= 100
                print("Defeat detected - game end")
            self.actions.play_again()
            return self._get_state(), reward, done
        
        action = self.available_action[valid_action]
        card_ind, x_frac, y_frac = action

        if card_ind >= 0 and card_ind < self.num_cards:
            x = int(x_frac * self.actions.WIDTH) + self.actions.TOP_LEFT_X
            y = int(y_frac * self.actions.HEIGHT / 2 + self.actions.TOP_LEFT_Y + self.actions.HEIGHT // 2) #Limit play to lower half
            print(f"Attempting to play card at {x}, {y}")
            self.actions.play_card(x, y, card_ind)
            time.sleep(1)

        done = False
        reward = self._compute_reward(self._get_state())
        next_state = self._get_state()
        return next_state, reward, done

    
    def _get_state(self):
        elixir = self.actions.count_elixir()

        cards_in_hand = self.card_detection.run_card_detection()

        allies, enemies = self.troop_detection.run_troop_detection()

        # Ensure cards in hand is length of 4
        while len(cards_in_hand) < 4:
            cards_in_hand.append(0.0)
        cards_in_hand = cards_in_hand[:4]

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
        enemy_positions = state[5 + 2 * MAX_ALLIES: len(state): 2]
        enemy_presence = sum(1 for pos in enemy_positions if pos > 0)

        reward = -enemy_presence

        # Elixir efficiency
        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent = self.prev_elixir - elixir
            enemy_reduced = self.prev_enemy_presence - enemy_presence
            if elixir_spent > 0 and enemy_reduced > 0:
                reward += 2 * min(elixir_spent, enemy_reduced)  # tune this factor (2 is arbitrary)

        self.prev_elixir = elixir
        self.prev_enemy_presence = enemy_presence

        # Tower detection reward
        curr_ally_tower, curr_enemy_tower = self.tower_detection.run_tower_detection()
        if self.prev_enemy_tower is not None:
            if curr_enemy_tower < self.prev_enemy_tower:
                reward = 20
        self.prev_enemy_tower = curr_enemy_tower

        if self.prev_ally_tower is not None:
            if curr_ally_tower < self.prev_ally_tower:
                reward -= 10
        self.prev_ally_tower = curr_ally_tower

        return reward

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
