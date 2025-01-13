from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import pickle
import torch.nn as nn
import os
from joblib import dump, load
from DQN_agent import ReplayBuffer, DQN_AGENT
from DQNNetwork import DQNNetwork

env = TimeLimit(env=HIVPatient(domain_randomization=False),
                max_episode_steps=200)

class DQN_model(torch.nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=4, depth=6):
        super(DQN_model, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


class ProjectAgent:
    def __init__(self):
        self.load()

    def act(self, observation):
        return self.greedy(observation)
    
    def greedy(self, state):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def load(self):
        self.model = DQN_model()
        model_saved = torch.load("trained_dqn_weights.pt")
        self.model.load_state_dict(model_saved['model_state_dict'])
        self.model.eval()
