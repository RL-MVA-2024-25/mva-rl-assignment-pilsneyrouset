from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
#from fast_env import FastHIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
import torch
import torch.nn as nn
from copy import deepcopy
import locale


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s2, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s2, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    
    def __len__(self):
        return len(self.data)
        

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


class DQN:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"

        self.model = model
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size, device)
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        cumulative_reward_ep = 0
        epsilon = self.epsilon_max
        step = 0
        reward_best_model = 0
        state, _ = env.reset()
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            y, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, y, done)
            cumulative_reward_ep += reward
            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            
            # update target network 
            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # next transition
            step += 1

            # End of episod
            if done or trunc:
                episode += 1

                episode_return.append(cumulative_reward_ep)
                print("Episode ", '{:2d}'.format(episode),
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)),
                          ", ep return ", '{:,.0f}'.format(cumulative_reward_ep).replace(',', ' '),
                          sep='')
                
                # Save best model
                if cumulative_reward_ep > reward_best_model:
                    torch.save({
                    'model_state_dict': self.target_model.state_dict(),
                    'reward': cumulative_reward_ep,
                    }, f"model_saved_test.pt")
                    reward_best_model = cumulative_reward_ep
              
                state, _ = env.reset()
                cumulative_reward_ep = 0
            else:
                state = y

        return episode_return


class DQN_model(torch.nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=4, depth=6):
        super(DQN_model, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)
                                                ])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


model = DQN_model().to(device)

config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.99,
          'buffer_size': 2_000_000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 15_000,
          'epsilon_delay_decay': 4_000,
          'batch_size': 2000,
          'gradient_steps': 1,
          'update_target_freq': 500,
          'criterion': torch.nn.SmoothL1Loss()
          }

agent = DQN(config, model)
episode_return = agent.train(env, 200)