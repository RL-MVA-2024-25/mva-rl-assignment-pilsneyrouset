import numpy as np
import gymnasium as gym
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit
import json
import pickle


class RandomForestFQI():
    def __init__(self, gamma=.7, horizon=200):
        self.env = TimeLimit(
                            env=HIVPatient(domain_randomization=False),
                            max_episode_steps=200
                            )
        self.states = gym.spaces.Discrete(4)
        self.actions = [np.array(pair) for pair in [[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]]]
        self.nb_actions = len(self.actions)
        self.gamma = gamma
        self.rf_model = None
        self.horizon = horizon

    def collect_samples(self, disable_tqdm=False, print_done_states=False):
        s, _ = self.env.reset()
        dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(self.horizon), disable=disable_tqdm):
            a = self.env.action_space.sample()
            s2, r, done, trunc, _ = self.env.step(a)
            dataset.append((s, a, r, s2, done, trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = self.env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        self.S = np.array(S)
        self.A = np.array(A).reshape((-1, 1))
        self.R = np.array(R)
        self.S2 = np.array(S2)
        self.D = np.array(D)

    def train(self, iterations=200, disable_tqdm=False):
        self.collect_samples()
        nb_samples = self.S.shape[0]
        Qfunctions = []
        SA = np.append(self.S, self.A, axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter == 0:
                value = self.R.copy()
            else:
                Q2 = np.zeros((nb_samples, self.nb_actions))
                for a2 in range(self.nb_actions):
                    A2 = a2 * np.ones((nb_samples, 1))
                    S2A2 = np.append(self.S2, A2, axis=1)
                    Q2[:, a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = self.R + self.gamma * (1 - self.D) * max_Q2
            Q = RandomForestRegressor()
            Q.fit(SA, value)
            Qfunctions.append(Q)
        self.rf_model = Qfunctions[-1]
        with open("random_forest_model.pkl", "wb") as fichier:
            pickle.dump(Qfunctions[-1], fichier)

    def act(self, state):
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(state, a).reshape(1, -1)
            Qsa.append(self.rf_model.predict(sa))
        return np.argmax(Qsa)
    
    def evaluate(self, nb_episodes=10):
        rewards = []
        for _ in range(nb_episodes):
            s, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            while not done and not truncated:
                a = self.greedy_action(s)
                s, r, done, truncated, _ = self.env.step(a)
                episode_reward += r
            rewards.append(episode_reward)
        return np.mean(rewards)
    

    

""" 
def collect_samples(env, horizon, disable_tqdm=False, print_done_states=False):
    " Collects a dataset of transitions from the environment. 
    Args:
    - env: the environment
    - horizon: the number of transitions to collect
    - disable_tqdm: if True, disables the progress bar
    - print_done_states: if True, prints "done!" when the environment is done
    Returns:
    - S: an array of states
    - A: an array of actions
    - R: an array of rewards
    - S2: an array of next states
    - D: an array of done flags
    "
    s, _ = env.reset()
    dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
        a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)
        dataset.append((s, a, r, s2, done, trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
            if done and print_done_states:
                print("done!")
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1, 1))
    R = np.array(R)
    S2 = np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D
"""

# agent = RandomForestFQI()
# agent.train()
 