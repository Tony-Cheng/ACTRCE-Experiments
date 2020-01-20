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


desired_advices = {
    "Reach red goal",
    "Reach blue goal",
    "Reach green goal"
}

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

    def get_advice(self, color, at_goal, is_lava):
        advice_list = []

        for advice in desired_advices:
            if advice_satisfied(advice, color, at_goal, is_lava):
                advice_list.append(advice)

        if len(advice_list) == 0:
            return None
        return advice_list[int(random() * len(advice_list))]


class KnowledgableTeacher(object):
    def __init__(self):
        super(KnowledgableTeacher, self).__init__()

    def get_advice(self, color, at_goal, is_lava, not_satisfied):
        advice_list = []

        for advice in advices:
            if advice_satisfied(advice, color, at_goal, is_lava) and advice not in not_satisfied:
                advice_list.append(advice)

        if len(advice_list) == 0:
            return None
        return advice_list[int(random() * len(advice_list))]


class DiscouragingTeacher(object):
    def __init__(self):
        super(DiscouragingTeacher, self).__init__()

    def get_advice(self, color, at_goal, is_lava):
        advice_list = []

        for advice in desired_advices:
            if not advice_satisfied(advice, color, at_goal, is_lava):
                advice_list.append(advice)

        if len(advice_list) == 0:
            return None
        return advice_list[int(random() * len(advice_list))]


class PessimisticTeacher(object):
    def __init__(self):
        super(PessimisticTeacher, self).__init__()

    def get_advice(self, color, at_goal, is_lava, not_satisfied):
        advice_list = []

        for advice in advices:
            if advice_satisfied(advice, color, at_goal, is_lava) and advice not in not_satisfied and \
                    advice not in desired_advices:
                advice_list.append(advice)

        if len(advice_list) == 0:
            return None
        return advice_list[int(random() * len(advice_list))]


class ReplayBuffer:
    def __init__(self, use_pessimistic=False, max_size=5000):
        self.use_pessimistic = use_pessimistic
        if use_pessimistic:
            self.teacher1 = PessimisticTeacher()
            self.teacher2 = PessimisticTeacher()
        else:
            self.teacher1 = OptimisticTeacher()
            self.teacher2 = DiscouragingTeacher()
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.all_states = []
        self.all_advices = []
        self.all_actions = []
        self.all_rewards = []
        self.all_next_states = []
        self.all_dones = []

    def new_episode(self):
        self.cur_states = []
        self.cur_actions = []
        self.cur_next_states = []
        self.cur_dones = []
        self.cur_colors = []
        self.cur_at_goals = []
        self.cur_is_lavas = []
        self.not_satisfied = set()

    def add(self, state, action, next_state, done, color, at_goal, is_lava):
        self.cur_states.append(state)
        self.cur_actions.append(action)
        self.cur_next_states.append(next_state)
        self.cur_dones.append(done)
        self.cur_colors.append(color)
        self.cur_at_goals.append(at_goal)
        self.cur_is_lavas.append(is_lava)

        if at_goal and "Avoid any goal" not in self.not_satisfied:
            self.not_satisfied.add("Avoid any goal")
        if is_lava and "Avoid any lava" not in self.not_satisfied:
            self.not_satisfied.add("Avoid any lava")

    def compute_reward(self, color, at_goal, is_lava, gamma=0.99):
        if self.use_pessimistic:
            advice1 = self.teacher1.get_advice(color, at_goal, is_lava, self.not_satisfied)
            advice2 = self.teacher2.get_advice(color, at_goal, is_lava, self.not_satisfied)
        else:
            advice1 = self.teacher1.get_advice(color, at_goal, is_lava)
            advice2 = self.teacher2.get_advice(color, at_goal, is_lava)

        if advice1 is not None:
            for i in range(len(self.cur_states)):
                self.all_states.append(self.cur_states[i])
                self.all_advices.append(advice1.split(' '))
                self.all_actions.append(self.cur_actions[i])
                if (advice1 == "Avoid any goal" or advice1 == "Avoid any lava") and \
                        advice_satisfied(advice1, color, at_goal, is_lava):
                    self.all_rewards.append(1/25)
                elif advice_satisfied(advice1, self.cur_colors[i], self.cur_at_goals[i], self.cur_is_lavas[i]):
                    self.all_rewards.append(1)
                else:
                    self.all_rewards.append(0)
                self.all_next_states.append(self.cur_next_states[i])
                self.all_dones.append(self.cur_dones[i])

        if advice2 is not None:
            for i in range(len(self.cur_states)):
                self.all_states.append(self.cur_states[i])
                self.all_advices.append(advice2.split(' '))
                self.all_actions.append(self.cur_actions[i])
                if (advice2 == "Avoid any goal" or advice2 == "Avoid any lava") and \
                        not advice_satisfied(advice2, color, at_goal, is_lava):
                    self.all_rewards.append(1/25)
                elif advice_satisfied(advice2, self.cur_colors[i], self.cur_at_goals[i], self.cur_is_lavas[i]):
                    self.all_rewards.append(1)
                else:
                    self.all_rewards.append(0)
                self.all_next_states.append(self.cur_next_states[i])
                self.all_dones.append(self.cur_dones[i])

        if (len(self.all_states) > self.max_size):
            self.all_states = self.all_states[-self.max_size:]
            self.all_advices = self.all_advices[-self.max_size:]
            self.all_actions = self.all_actions[-self.max_size:]
            self.all_rewards = self.all_rewards[-self.max_size:]
            self.all_next_states = self.all_next_states[-self.max_size:]
            self.all_dones = self.all_dones[-self.max_size:]

    def __len__(self):
        return len(self.all_states)

    def sample(self, bs=64):

        if len(self.all_states) < bs:
            return None

        indexes = np.arange(len(self.all_states))
        np.random.shuffle(indexes)
        indexes = indexes[:bs]
        states, advices, actions, rewards, next_states, dones = [], [], [], [], [], []

        for i in indexes:
            state, advice, action, reward, next_state, done = self.all_states[i], self.all_advices[
                i], self.all_actions[i], self.all_rewards[i], self.all_next_states[i], self.all_dones[i]
            states.append(np.array(state, copy=False))
            advices.append(np.array(advice, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            next_states.append(np.array(next_state, copy=False))
            dones.append(np.array(done, copy=False))

        return (np.array(states), np.asarray(advices)), np.array(actions), np.array(rewards), \
            np.array(next_states), np.array(dones)


def sample_advice(undesirable=False):
    if undesirable == True:
        advice_list = []
        for advice in advices:
            if advice not in desired_advices:
                advice_list.append(advice)
    else:        
        advice_list = list(desired_advices)
    advice = advice_list[int(random() * len(advice_list))]
    return advice


def get_state(env):
    return env.get_combined_obs().transpose((2, 0, 1))


def advice_satisfied(initial_advice, color, at_goal, is_lava, undesirable=False):
    if at_goal and color == Color_Index.red and initial_advice == "Reach red goal":
        return True
    if at_goal and color == Color_Index.blue and initial_advice == "Reach blue goal":
        return True
    if at_goal and color == Color_Index.green and initial_advice == "Reach green goal":
        return True
    if is_lava and color == Color_Index.red and initial_advice == "Reach red lava":
        return True
    if is_lava and color == Color_Index.blue and initial_advice == "Reach blue lava":
        return True
    if is_lava and color == Color_Index.green and initial_advice == "Reach green lava":
        return True
    if not at_goal and initial_advice == "Avoid any goal":
        if undesirable == True:
            return False
        return True
    if not is_lava and initial_advice == "Avoid any lava":
        if undesirable == True:
            return False
        return True
    if undesirable and at_goal and initial_advice == "Avoid any goal":
        return True
    if undesirable and is_lava and initial_advice == "Avoid any lava":
        return True
    return False
