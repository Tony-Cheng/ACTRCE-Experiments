import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from random import random
from krazy_gridworld import Color_Index


class Goal(object):
    red_goal = 0
    blue_goal = 1
    green_goal = 2
    red_lava = 3
    blue_lava = 4
    green_lava = 5
    avoid_lava = 6
    avoid_goal = 7


advices = {
    "Reach red goal",
    "Reach blue goal",
    "Reach green goal",
    "Reach red lava",
    "Reach blue lava",
    "Reach green lava",
    "Avoid any lava",
    "Avoid any goal"
}


class OptimisticTeacher(object):
    def __init__(self):
        super(OptimisticTeacher, self).__init__()

    def reset(self):
        self.plausible_advices = {"Avoid any lava", "Avoid any goal"}
        self.initial_advice = sample_advice()
        self.last_advice = None
        return self.initial_advice

    def compute_step(self, color, at_goal, is_lava):
        self.last_advice = None
        if at_goal and "Avoid any goal" in self.plausible_advices:
            self.plausible_advices.remove("Avoid any goal")
        if is_lava and "Avoid any lava" in self.plausible_advices:
            self.plausible_advices.remove("Avoid any lava")

        if at_goal and color == Color_Index.red:
            self.last_advice = "Reach red goal"
        if at_goal and color == Color_Index.blue:
            self.last_advice = "Reach blue goal"
        if at_goal and color == Color_Index.green:
            self.last_advice = "Reach green goal"
        if is_lava and color == Color_Index.red:
            self.last_advice = "Reach red lava"
        if is_lava and color == Color_Index.blue:
            self.last_advice = "Reach blue lava"
        if is_lava and color == Color_Index.green:
            self.last_advice = "Reach green lava"

        if self.initial_advice == self.last_advice:
            return True

        if self.initial_advice == "Avoid any goal" and "Avoid any goal" in self.plausible_advices:
            self.last_advice = "Avoid any goal"
            return False
        if self.initial_advice == "Avoid any lava" and "Avoid any lava" in self.plausible_advices:
            self.last_advice == "Avoid any lava"
            return False
        return False

    def get_advice(self):
        if self.last_advice is not None:
            if self.last_advice == self.initial_advice:
                return self.last_advice, True
            else:
                if self.last_advice not in self.plausible_advices:
                    self.plausible_advices.add(self.last_advice)
                advice_list = list(self.plausible_advices)
                return advice_list[int(random() * len(advice_list))], False
        return None


class ReplayBuffer:
    def __init__(self, max_size=5000):
        self.teacher = OptimisticTeacher()
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.all_states = []
        self.all_advices = []
        self.all_actions = []
        self.all_expected_rewards = []
        self.new_episode()

    def new_episode(self):
        self.cur_states = []
        self.cur_actions = []
        return self.teacher.reset()

    def add(self, state, action, color, at_goal, is_lava):
        self.cur_states.append(state)
        self.cur_actions.append(action)
        self.teacher.compute_step(color, at_goal, is_lava)

    def compute_reward(self, gamma=0.99):
        advice = self.teacher.get_advice()

        if advice is None:
            return
        else:
            advice, is_initial = advice

        cur_reward = 1.0

        for i in reversed(range(len(self.cur_states))):
            self.all_actions.append(self.cur_actions[i])
            self.all_states.append(self.cur_states[i])
            self.all_advices.append(advice.split(" "))
            self.all_expected_rewards.append(cur_reward)
            cur_reward *= gamma

        if (len(self.all_states) > self.max_size):
            self.all_states = self.all_states[-self.max_size:]
            self.all_advices = self.all_advices[-self.max_size:]
            self.all_actions = self.all_actions[-self.max_size:]
            self.all_expected_rewards = self.all_expected_rewards[-self.max_size:]

        return is_initial

    def __len__(self):
        return len(self.all_states)

    def sample(self, bs=64):

        if len(self.all_states) < bs:
            return None

        indexes = np.arange(len(self.all_states))
        np.random.shuffle(indexes)
        indexes = indexes[:64]
        states, advices, actions, expected_rewards = [], [], [], []

        for i in indexes:
            state, advice, action, reward = self.all_states[i], self.all_advices[
                i], self.all_actions[i], self.all_expected_rewards[i]
            states.append(np.array(state, copy=False))
            advices.append(np.array(advice, copy=False))
            actions.append(np.array(action, copy=False))
            expected_rewards.append(np.array(reward, copy=False))

        return (np.array(states), np.asarray(advices)), np.array(actions), np.array(expected_rewards)


def sample_advice():
    advice_list = list(advices)
    return advice_list[int(random() * len(advice_list))]


def get_state(env):
    return env.get_combined_obs().transpose((2, 0, 1))
