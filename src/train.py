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
from DQN_Augustin import MLP

env = TimeLimit(env=HIVPatient(domain_randomization=False),
                max_episode_steps=200)  


class ProjectAgent:
    def __init__(self, name='MLP'):
        self.env = TimeLimit(
                            env=HIVPatient(domain_randomization=False),
                            max_episode_steps=200
                            )  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.original_env = self.env.env
        self.nb_actions = int(self.original_env.action_space.n)
        self.nb_neurons = 256
        self.model = DQNNetwork(state_dim=self.env.observation_space.shape[0], nb_neurons=self.nb_neurons, n_action=self.nb_actions).to(self.device)
    
    def act(self, observation, use_random=False):
        if use_random:
            return self.env.action_space.sample()
        elif self.name == 'RF_FQI':
            self.load()
            Qsa = []
            for a in range(self.nb_actions):
                sa = np.append(observation, a).reshape(1, -1)
                Qsa.append(self.Qfunction.predict(sa))
            return np.argmax(Qsa)

        elif self.name == 'DQN':
            device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        elif self.name == 'MLP':
            device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        else:
            raise ValueError("Unknown model")

    def save(self):
        if self.name == 'RF_FQI':
            filename = 'models/RF_FQI/Qfct'
            model = {'Qfunction': self.agent.rf_model}
            dump(model, filename, compress=9)
        elif self.name == 'DQN':
            filename = "models/DQN/config1.pt"
            torch.save(self.model.state_dict(), filename)

    def load(self):
        if self.name == 'RF_FQI':
            self.Qfunction = load("src/random_forest_model.pkl")
        elif self.name == 'DQN':
            device = torch.device('cpu')
            self.model.load_state_dict(torch.load('/home/onyxia/work/config4.pt', weights_only=True))
            self.model.eval()
        elif self.name == 'MLP':
            self.model = MLP(input_dim=self.env.observation_space.shape[0], hidden_dim=512, output_dim=self.env.action_space.n, depth=5, activation=torch.nn.SiLU(), normalization='None')

            self.model.load_state_dict(torch.load("/home/onyxia/work/mva-rl-assignment-pilsneyrouset/config4.pt"))

            self.model.eval()
