import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class ClashRoyaleDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values
    
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999, target_update_freq=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # Online & target networks
        self.policy_net = ClashRoyaleDQN(state_size, action_size)
        self.target_net = ClashRoyaleDQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = []  # Simple replay buffer
        self.batch_size = 64

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 50000:
            self.memory.pop(0)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.policy_net(state_tensor)).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Q-values for current states
        q_values = self.policy_net(states).gather(1, actions)

        # Double DQN: action selection from policy_net, value from target_net
        next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions)

        # Target
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss
        loss = F.mse_loss(q_values, targets.detach())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target net periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

