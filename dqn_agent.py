import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

ALL_CARDS = ["Hog Rider", "Fireball", "Archers", "Mega Minion"]  # etc
card_to_idx = {name: i for i, name in enumerate(ALL_CARDS)}

GRID_W, GRID_H = 18, 8
NUM_POSITIONS = GRID_W * GRID_H
CARDS_IN_HAND = 4
ACTION_SIZE = CARDS_IN_HAND * NUM_POSITIONS

class ClashRoyaleDQN(nn.Module):
    def __init__(self, num_cards, card_embed_dim, other_features_dim):
        super().__init__()
        self.card_embedding = nn.Embedding(num_cards, card_embed_dim)
        input_size = card_embed_dim * CARDS_IN_HAND + other_features_dim
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, ACTION_SIZE)

    def forward(self, elixir, cards_in_hand, other_features):
        embedded_cards = self.card_embedding(cards_in_hand)
        embedded_cards = embedded_cards.view(embedded_cards.size(0), -1)
        x = torch.cat([elixir, embedded_cards, other_features], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim):
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 1e-3

        self.model = ClashRoyaleDQN(len(ALL_CARDS), 8, state_dim)
        self.target_model = ClashRoyaleDQN(len(ALL_CARDS), 8, state_dim)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        with torch.no_grad():
            q_values = self.model(*state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        elixirs, cards_in_hand, other_features, actions, rewards, next_elixirs, next_cards, next_other, dones = [], [], [], [], [], [], [], [], []

        for state, action, reward, next_state, done in batch:
            el, cards, other = state
            nel, ncards, nother = next_state
            elixirs.append(el)
            cards_in_hand.append(cards)
            other_features.append(other)
            actions.append(action)
            rewards.append(reward)
            next_elixirs.append(nel)
            next_cards.append(ncards)
            next_other.append(nother)
            dones.append(done)

        elixirs = torch.FloatTensor(elixirs).unsqueeze(1)
        cards_in_hand = torch.LongTensor(cards_in_hand)
        other_features = torch.FloatTensor(other_features)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_elixirs = torch.FloatTensor(next_elixirs).unsqueeze(1)
        next_cards = torch.LongTensor(next_cards)
        next_other = torch.FloatTensor(next_other)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.model(elixirs, cards_in_hand, other_features).gather(1, actions)
        next_q_values = self.target_model(next_elixirs, next_cards, next_other).max(1)[0].unsqueeze(1)
        targets = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def decode_action(action_index):
    card_slot = action_index // NUM_POSITIONS
    pos_idx = action_index % NUM_POSITIONS
    grid_x = pos_idx % GRID_W
    grid_y = pos_idx // GRID_W
    return card_slot, grid_x, grid_y
