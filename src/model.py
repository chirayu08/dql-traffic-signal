import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class DQN(nn.Module):
    def __init__(self, num_states, num_actions, lr):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(num_states, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, num_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.to(self.device)

    def forward(self, x):
        return self.net(x)

    def predict_one(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.forward(s).cpu().numpy()[0]

    def predict_batch(self, states):
        s = torch.FloatTensor(np.array(states)).to(self.device)
        with torch.no_grad():
            return self.forward(s).cpu().numpy()

    def train_batch(self, states, targets):
        s = torch.FloatTensor(np.array(states)).to(self.device)
        t = torch.FloatTensor(np.array(targets)).to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.forward(s), t)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))