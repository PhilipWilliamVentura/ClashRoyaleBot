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
    
    def stop(self):
        self.listener.stop()

def get_latest_model_path(models_dir="models"):
    """Get the most recent model file based on creation time"""
    model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
    if not model_files:
        return None
    
    # Sort by file modification time (most recent last)
    model_files.sort(key=os.path.getmtime)
    return model_files[-1]

def save_checkpoint(agent, episode):
    """Save model and metadata with episode number"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = os.path.join("models", f"model_ep{episode:06d}_{timestamp}.pth")
    torch.save(agent.policy_net.state_dict(), model_path)
    
    # Save metadata
    meta_path = os.path.join("models", f"meta_ep{episode:06d}_{timestamp}.json")
    metadata = {
        "episode": episode,
        "epsilon": agent.epsilon,
        "timestamp": timestamp
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    
    print(f"Saved checkpoint at episode {episode}, epsilon: {agent.epsilon:.4f}")

def load_latest_checkpoint(agent):
    latest_model = get_latest_model_path("models")
    if latest_model:
        print(f"Loading model: {latest_model}")
        state_dict = torch.load(latest_model, map_location="cpu")

        # Check first layer size
        fc1_weight = state_dict.get("fc1.weight")
        if fc1_weight is not None and fc1_weight.shape[1] != agent.policy_net.fc1.in_features:
            print(f"State size mismatch: checkpoint expects {fc1_weight.shape[1]} features, "
                  f"but agent has {agent.policy_net.fc1.in_features}. Skipping load.")
            return 0

        agent.policy_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Load metadata
        base_name = os.path.basename(latest_model)
        meta_name = base_name.replace("model_", "meta_").replace(".pth", ".json")
        meta_path = os.path.join("models", meta_name)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            agent.epsilon = meta.get("epsilon", agent.epsilon)
            start_episode = meta.get("episode", 0) + 1
            print(f"Loaded checkpoint - Episode: {meta.get('episode', 0)}, Epsilon: {agent.epsilon:.4f}")
            return start_episode
        else:
            print(f"Metadata file not found: {meta_path}")
            print("Starting with default epsilon")
    return 0


def train():
    env = Env()
    agent = DoubleDQNAgent(env.state_size, env.action_size)
    
    os.makedirs("models", exist_ok=True)
    
    # Load latest checkpoint and get starting episode
    start_episode = load_latest_checkpoint(agent)
    
    controller = KeyboardController()
    episodes = 10000
    
    for ep in range(start_episode, episodes):
        if controller.is_exit_requested():
            print("Training interrupted by user.")
            save_checkpoint(agent, ep)  # Save before exiting
            controller.stop()
            env.finish()
            break
        
        state = env.reboot()
        total_reward = 0
        done = False
        
        print(f"Episode {ep+1} starting. Epsilon: {agent.epsilon:.3f}")
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        
        print(f"Episode {ep+1} finished. Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # Save checkpoint every 10 episodes and at the end
        if ep % 10 == 0 or ep == episodes - 1:
            save_checkpoint(agent, ep)

if __name__ == "__main__":
    train()