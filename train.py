import os
import torch
import glob
import json
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

def get_latest_model_path(models_dir="models"):
    model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
    if not model_files:
        return None
    model_files.sort()  # Lexicographical sort works for timestamps
    return model_files[-1]

def train():
    env = Env()
    agent = DoubleDQNAgent(env.state_size, env.action_size)

    os.makedirs("models", exist_ok=True)

    latest_model = get_latest_model_path("models")
    if latest_model:
        agent.policy_net.load_state_dict(torch.load(latest_model))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        meta_path = latest_model.replace("model_", "meta_").replace(".pth", ".json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                agent.epsilon = meta.get("epsilon", agent.epsilon)
            print(f"Epsilon loaded: {agent.epsilon}")

    controller = KeyboardController()
    episodes = 10000

    for ep in range(episodes):
        if controller.is_exit_requested():
            print("Training interrupted by user.")
            break

        state = env.reboot()
        total_reward = 0
        done = False
        print(f"Episode {ep+1} starting. Epsilon: {agent.epsilon:.3f}")

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()  # no batch_size arg here, handled internally
            state = next_state
            total_reward += reward

        print(f"Episode {ep+1} finished. Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        if ep % 10 == 0 or ep == episodes - 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join("models", f"model_{timestamp}.pth")
            torch.save(agent.policy_net.state_dict(), model_path)
            with open(os.path.join("models", f"meta_{timestamp}.json"), "w") as f:
                json.dump({"epsilon": agent.epsilon}, f)
            print(f"Saved model and epsilon at episode {ep+1}")

if __name__ == "__main__":
    train()
