import os
import torch
import json
import pickle
from actions import Actions
from env import Env
from dqn_agent import DoubleDQNAgent
from pynput import keyboard
from datetime import datetime

class KeyboardController:
    def __init__(self):
        self.should_exit = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'q':
                print("\nShutdown requested - cleaning up...")
                self.should_exit = True
        except AttributeError:
            pass  # Special key pressed

    def is_exit_requested(self):
        return self.should_exit
    
    def stop(self):
        self.listener.stop()


def load_agent(agent, path_prefix):
    """Load model weights, epsilon, and replay buffer if they exist."""
    model_path = f"{path_prefix}.pth"
    meta_path = f"{path_prefix}_meta.json"
    memory_path = f"{path_prefix}_memory.pkl"

    if os.path.exists(model_path):
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            agent.epsilon = json.load(f).get("epsilon", agent.epsilon)
    if os.path.exists(memory_path):
        with open(memory_path, "rb") as f:
            agent.memory = pickle.load(f)
        print(f"Loaded replay buffer with {len(agent.memory)} experiences")


def friendlymatch():
    env = Env()
    agent = DoubleDQNAgent(env.state_size, env.action_size)
    actions = Actions()

    # Load latest agent
    load_agent(agent, "models/model_latest")
    print(f"Resuming training. Epsilon: {agent.epsilon:.3f}, Memory size: {len(agent.memory)}")

    controller = KeyboardController()
    episodes = 1
    actions.start_friendly_match()

    for ep in range(episodes):
        state = env.reboot()
        total_reward = 0
        done = False
        print(f"Episode {ep+1} starting. Epsilon: {agent.epsilon:.3f}")

        while not done:
            if controller.is_exit_requested():
                controller.stop()
                env.finish()
                return

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        print("GG")



if __name__ == "__main__":
    friendlymatch()
