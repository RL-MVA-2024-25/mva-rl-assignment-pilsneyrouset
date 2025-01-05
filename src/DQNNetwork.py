import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, nb_neurons, n_action):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, nb_neurons)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(nb_neurons, nb_neurons)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(nb_neurons, nb_neurons*2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(nb_neurons*2, nb_neurons)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(nb_neurons, nb_neurons)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(nb_neurons, n_action)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.fc6(x)
        return x