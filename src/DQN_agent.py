import numpy as np
import gymnasium as gym
import torch
from copy import deepcopy
import random 
from DQNNetwork import DQNNetwork
import torch.nn as nn
import torch.nn.utils as utils
from gymnasium.wrappers import TimeLimit
from torch.distributions.categorical import Categorical
from env_hiv import HIVPatient
from evaluate import evaluate_HIV   


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)
    

class DQN_AGENT():
    def __init__(self, config):
        self.env = TimeLimit(
                            env=HIVPatient(domain_randomization=False),
                            max_episode_steps=200
                            )
        self.device = "cpu"
        self.nb_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.model = torch.nn.Sequential(nn.Linear(self.state_dim, 256),
                          nn.ReLU(),
                          nn.Linear(256, 256),
                          nn.ReLU(), 
                          nn.Linear(256, self.nb_actions)).to(self.device)
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size, self.device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
    
    def train(self, max_episode=150):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        test_episode = 0
        best_score = 0
        state, _ = self.env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.greedy_action(state)
            # step
            next_state, reward, done, trunc, _ = self.env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                if episode > test_episode:
                    test_score = evaluate_HIV(agent=self, nb_episode=1)
                else:
                    test_score = 0
                if test_score > best_score:
                    best_score = test_score
                    self.best_model = deepcopy(self.model).to(self.device)
                print("Episode ", '{:3d}'.format(episode),
                      ", epsilon ", '{:6.2f}'.format(epsilon),
                      ", batch size ", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = self.env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        self.model.load_state_dict(self.best_model.state_dict())
        torch.save(self.model.state_dict(), "src/dqn_model_weights.pt")
        return episode_return

    def greedy_action(self, state):
        with torch.no_grad():
            Q = self.model(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
    
    def act(self, observation, use_random=False):
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
            

config = {'learning_rate': 0.001,
          'gamma': 0.98,
          'buffer_size': 100000,
          'epsilon_min': 0.02,
          'epsilon_max': 1.,
          'epsilon_decay_period': 15000,
          'epsilon_delay_decay': 100,
          'batch_size': 600,
          'gradient_steps': 2,
          'update_target_strategy': 'replace',
          'update_target_freq': 400,
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss()}

# agent = DQN_AGENT(config=config)
# episode_return = agent.train()