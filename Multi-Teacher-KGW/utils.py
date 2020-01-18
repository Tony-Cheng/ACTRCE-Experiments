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

    def update_advice(self, color, at_goal, is_lava):
        if at_goal and "Avoid any goal" in self.plausible_advices:
            self.plausible_advices.remove("Avoid any goal")
        if is_lava and "Avoid any lava" in self.plausible_advices:
            self.plausible_advices.remove("Avoid any lava")

    def get_advice(self, color, at_goal, is_lava):
        if at_goal and color == Color_Index.red:
            self.plausible_advices.add("Reach red goal")
        if at_goal and color == Color_Index.blue:
            self.plausible_advices.add("Reach blue goal")
        if at_goal and color == Color_Index.green:
            self.plausible_advices.add("Reach green goal")
        if is_lava and color == Color_Index.red:
            self.plausible_advices.add("Reach red lava")
        if is_lava and color == Color_Index.blue:
            self.plausible_advices.add("Reach blue lava")
        if is_lava and color == Color_Index.green:
            self.plausible_advices.add("Reach green lava")

        advice_list = list(self.plausible_advices)
        if len(advice_list) == 0:
            return None
        return advice_list[int(random() * len(advice_list))]


class DiscouragingTeacher(object):
    def __init__(self):
        super(DiscouragingTeacher, self).__init__()

    def reset(self, initial_advice):
        self.not_plausible_advices = set()
        self.initial_advice = initial_advice

    def update_advice(self, color, at_goal, is_lava):
        if at_goal and "Avoid any goal" not in self.not_plausible_advices:
            self.not_plausible_advices.add("Avoid any goal")
        if is_lava and "Avoid any lava" not in self.not_plausible_advices:
            self.not_plausible_advices.add("Avoid any lava")

    def get_advice(self, color, at_goal, is_lava):
        if not (at_goal and color == Color_Index.red):
            self.not_plausible_advices.add("Reach red goal")
        if not (at_goal and color == Color_Index.blue):
            self.not_plausible_advices.add("Reach blue goal")
        if not (at_goal and color == Color_Index.green):
            self.not_plausible_advices.add("Reach green goal")
        if not (is_lava and color == Color_Index.red):
            self.not_plausible_advices.add("Reach red lava")
        if not (is_lava and color == Color_Index.blue):
            self.not_plausible_advices.add("Reach blue lava")
        if not (is_lava and color == Color_Index.green):
            self.not_plausible_advices.add("Reach green lava")

        advice_list = list(self.not_plausible_advices)
        if self.initial_advice in advice_list:
            return self.initial_advice
        else:
            return None


class ReplayBuffer:
    def __init__(self, max_size=5000):
        self.teacher1 = OptimisticTeacher()
        self.teacher2 = DiscouragingTeacher()
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.all_states = []
        self.all_advices = []
        self.all_actions = []
        self.all_expected_rewards = []

    def new_episode(self, initial_advice):
        self.cur_states = []
        self.cur_actions = []
        self.teacher1.reset()
        self.teacher2.reset(initial_advice)

    def add(self, state, action, color, at_goal, is_lava):
        self.cur_states.append(state)
        self.cur_actions.append(action)
        self.teacher1.update_advice(color, at_goal, is_lava)
        self.teacher2.update_advice(color, at_goal, is_lava)

    def compute_reward(self, color, at_goal, is_lava, gamma=0.99):
        optimistic_advice = self.teacher1.get_advice(color, at_goal, is_lava)
        discouraging_advice = self.teacher2.get_advice(color, at_goal, is_lava)

        if optimistic_advice is not None:
            cur_reward = 1.0

            for i in reversed(range(len(self.cur_states))):
                self.all_actions.append(self.cur_actions[i])
                self.all_states.append(self.cur_states[i])
                self.all_advices.append(optimistic_advice.split(" "))
                self.all_expected_rewards.append(cur_reward)
                cur_reward *= gamma

        if discouraging_advice is not None:
            cur_reward = -1.0

            for i in reversed(range(len(self.cur_states))):
                self.all_actions.append(self.cur_actions[i])
                self.all_states.append(self.cur_states[i])
                self.all_advices.append(optimistic_advice.split(" "))
                self.all_expected_rewards.append(cur_reward)
                cur_reward *= gamma

        if (len(self.all_states) > self.max_size):
            self.all_states = self.all_states[-self.max_size:]
            self.all_advices = self.all_advices[-self.max_size:]
            self.all_actions = self.all_actions[-self.max_size:]
            self.all_expected_rewards = self.all_expected_rewards[-self.max_size:]

    def __len__(self):
        return len(self.all_states)

    def sample(self, bs=64):

        if len(self.all_states) < bs:
            return None

        indexes = np.arange(len(self.all_states))
        np.random.shuffle(indexes)
        indexes = indexes[:bs]
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


def advice_satisfied(initial_advice, color, at_goal, is_lava):
    if at_goal and color == Color_Index.red and initial_advice == "Reach red goal":
        return True, True
    if at_goal and color == Color_Index.blue and initial_advice == "Reach blue goal":
        return True, True
    if at_goal and color == Color_Index.green and initial_advice == "Reach green goal":
        return True, True
    if is_lava and color == Color_Index.red and initial_advice == "Reach red lava":
        return True, True
    if is_lava and color == Color_Index.blue and initial_advice == "Reach blue lava":
        return True, True
    if is_lava and color == Color_Index.green and initial_advice == "Reach green lava":
        return True, True
    if at_goal and initial_advice == "Avoid any goal":
        return True, False
    if is_lava and initial_advice == "Avoid any lava":
        return True, False
    if initial_advice == "Avoid any goal":
        return False, True
    if initial_advice == "Avoid any lava":
        return False, True
    return False, False
