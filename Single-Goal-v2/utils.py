import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from random import random
from krazy_gridworld import Color_Index


class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def set_one(self, length):
        for i in range(length):
            self.rewards[-i] = 1

    def sample(self, bs=64):

        if len(self.states) < bs:
            return None

        if (len(self.states) > self.max_size):
            self.states = self.states[-self.max_size:]
            self.actions = self.actions[-self.max_size:]
            self.rewards = self.rewards[-self.max_size:]
            self.next_states = self.next_states[-self.max_size:]
            self.dones = self.dones[-self.max_size:]

        indexes = np.arange(len(self.states))
        np.random.shuffle(indexes)
        indexes = indexes[:bs]

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in indexes:
            state, action, reward, next_state, done = self.states[i], self.actions[
                i], self.rewards[i], self.next_states[i], self.dones[i]
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

def get_state(env):
    return env.get_combined_obs().transpose((2, 0, 1))

def advice_satisfied(color, at_goal, is_lava):
    if at_goal and color == Color_Index.red:
        return True
    return False