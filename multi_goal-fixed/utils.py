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


atomics = {Goal.red_goal,
           Goal.blue_goal,
           Goal.green_goal,
           Goal.red_lava,
           Goal.blue_lava,
           Goal.green_lava,
           Goal.avoid_lava,
           Goal.avoid_goal}

desired_advices = {"Reach red goal": [{Goal.red_goal}],
                   "Reach blue goal": [{Goal.blue_goal}],
                   "Reach green goal": [{Goal.green_goal}],
                   "Reach red goal and Reach blue goal": [{Goal.red_goal, Goal.blue_goal}],
                   "Reach blue goal and Reach red goal": [{Goal.blue_goal, Goal.red_goal}],
                   "Reach red goal or Reach blue goal": [{Goal.red_goal}, {Goal.blue_goal}],
                   "Reach blue goal or Reach red goal": [{Goal.blue_goal}, {Goal.red_goal}],
                   "Reach red goal and Reach green goal": [{Goal.red_goal, Goal.green_goal}],
                   "Reach green goal and Reach red goal": [{Goal.green_goal, Goal.red_goal}],
                   "Reach red goal or Reach green goal": [{Goal.red_goal}, {Goal.green_goal}],
                   "Reach green goal or Reach red goal": [{Goal.green_goal}, {Goal.red_goal}],
                   "Reach blue goal and Reach green goal": [{Goal.blue_goal, Goal.green_goal}],
                   "Reach green goal and Reach blue goal": [{Goal.green_goal, Goal.blue_goal}],
                   "Reach blue goal or Reach green goal": [{Goal.blue_goal}, {Goal.green_goal}],
                   "Reach green goal or Reach blue goal": [{Goal.green_goal}, {Goal.blue_goal}],
                   "Reach red goal and Avoid any lava": [{Goal.red_goal, Goal.avoid_lava}],
                   "Avoid any lava and Reach red goal": [{Goal.avoid_lava, Goal.red_goal}],
                   "Reach blue goal and Avoid any lava": [{Goal.blue_goal, Goal.avoid_lava}],
                   "Avoid any lava and Reach blue goal": [{Goal.avoid_lava, Goal.blue_goal}],
                   "Reach green goal and Avoid any lava": [{Goal.green_goal, Goal.avoid_lava}],
                   "Avoid any lava and Reach green goal": [Goal.avoid_lava, Goal.green_goal]}

advices = {"Reach red goal": [{Goal.red_goal}],
           "Reach blue goal": [{Goal.blue_goal}],
           "Reach green goal": [{Goal.green_goal}],
           "Reach red goal and Reach blue goal": [{Goal.red_goal, Goal.blue_goal}],
           "Reach blue goal and Reach red goal": [{Goal.blue_goal, Goal.red_goal}],
           "Reach red goal or Reach blue goal": [{Goal.red_goal}, {Goal.blue_goal}],
           "Reach blue goal or Reach red goal": [{Goal.blue_goal}, {Goal.red_goal}],
           "Reach red goal and Reach green goal": [{Goal.red_goal, Goal.green_goal}],
           "Reach green goal and Reach red goal": [{Goal.green_goal, Goal.red_goal}],
           "Reach red goal or Reach green goal": [{Goal.red_goal}, {Goal.green_goal}],
           "Reach green goal or Reach red goal": [{Goal.green_goal}, {Goal.red_goal}],
           "Reach blue goal and Reach green goal": [{Goal.blue_goal, Goal.green_goal}],
           "Reach green goal and Reach blue goal": [{Goal.green_goal, Goal.blue_goal}],
           "Reach blue goal or Reach green goal": [{Goal.blue_goal}, {Goal.green_goal}],
           "Reach green goal or Reach blue goal": [{Goal.green_goal}, {Goal.blue_goal}],
           "Reach red goal and Avoid any lava": [{Goal.red_goal, Goal.avoid_lava}],
           "Avoid any lava and Reach red goal": [{Goal.avoid_lava, Goal.red_goal}],
           "Reach blue goal and Avoid any lava": [{Goal.blue_goal, Goal.avoid_lava}],
           "Avoid any lava and Reach blue goal": [{Goal.avoid_lava, Goal.blue_goal}],
           "Reach green goal and Avoid any lava": [{Goal.green_goal, Goal.avoid_lava}],
           "Avoid any lava and Reach green goal": [Goal.avoid_lava, Goal.green_goal],
           "Avoid any lava and Avoid any goal": [{Goal.avoid_goal, Goal.avoid_lava}],
           "Avoid any goal and Avoid any lava": [{Goal.avoid_goal, Goal.avoid_lava}]}


class OptimisticTeacher(object):
    def __init__(self):
        super(OptimisticTeacher, self).__init__()

    def get_advice(self, satisfied, not_satisfied):
        advice_list = []
        satisfied = True
        for advice in desired_advices:
            if check_satisfied(advice, satisfied, not_satisfied):
                advice_list.append(advice)

        if len(advice_list) == 0:
            return None
        return advice_list[int(random() * len(advice_list))]


class KnowledgableTeacher(object):
    def __init__(self):
        super(KnowledgableTeacher, self).__init__()

    def get_advice(self, satisfied, not_satisfied):
        advice_list = []

        for advice in advices:
            if check_satisfied(advice, satisfied, not_satisfied):
                advice_list.append(advice)

        if len(advice_list) == 0:
            return None
        return advice_list[int(random() * len(advice_list))]


class DiscouragingTeacher(object):
    def __init__(self):
        super(DiscouragingTeacher, self).__init__()

    def get_advice(self, satisfied, not_satisfied):
        advice_list = []

        for advice in desired_advices:
            if not check_satisfied(advice, satisfied, not_satisfied):
                advice_list.append(advice)

        if len(advice_list) == 0:
            return None
        return advice_list[int(random() * len(advice_list))]


class ReplayBuffer:
    def __init__(self, max_size=5000):
        self.teacher1 = KnowledgableTeacher()
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
        self.all_hx = []
        self.all_cx = []

    def new_episode(self):
        self.cur_states = []
        self.cur_actions = []
        self.cur_next_states = []
        self.cur_dones = []
        self.cur_colors = []
        self.cur_at_goals = []
        self.cur_is_lavas = []
        self.cur_satisfied = set()
        self.cur_not_satisfied = set()
        self.cur_hx = []
        self.cur_cx = []

    def add(self, state, action, next_state, done, hidden_states, color, at_goal, is_lava):
        self.cur_states.append(state)
        self.cur_actions.append(action)
        self.cur_next_states.append(next_state)
        self.cur_dones.append(done)
        self.cur_colors.append(color)
        self.cur_at_goals.append(at_goal)
        self.cur_is_lavas.append(is_lava)

        for atomic in atomics:
            if advice_satisfied(atomic, color, at_goal, is_lava):
                self.cur_satisfied.add(atomic)
            if advice_not_satisfied(atomic, color, at_goal, is_lava):
                self.cur_not_satisfied.add(atomic)

        (hx, cx) = hidden_states
        self.cur_hx.append(hx)
        self.cur_cx.append(cx)

    def compute_reward(self, color, at_goal, is_lava):
        advice1 = self.teacher1.get_advice(
            self.cur_satisfied, self.cur_not_satisfied)
        advice2 = self.teacher2.get_advice(
            self.cur_satisfied, self.cur_not_satisfied)

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
                self.all_hx.append(self.cur_hx[i])
                self.all_cx.append(self.cur_cx[i])

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
                self.all_hx.append(self.cur_hx[i])
                self.all_cx.append(self.cur_cx[i])

        if (len(self.all_states) > self.max_size):
            self.all_states = self.all_states[-self.max_size:]
            self.all_advices = self.all_advices[-self.max_size:]
            self.all_actions = self.all_actions[-self.max_size:]
            self.all_rewards = self.all_rewards[-self.max_size:]
            self.all_next_states = self.all_next_states[-self.max_size:]
            self.all_dones = self.all_dones[-self.max_size:]
            self.all_hx = self.all_hx[-self.max_size:]
            self.all_cx = self.all_cx[-self.max_size:]

    def __len__(self):
        return len(self.all_states)

    def sample(self, bs=64):

        if len(self.all_states) < bs:
            return None

        indexes = np.arange(len(self.all_states))
        np.random.shuffle(indexes)
        indexes = indexes[:bs]
        states, advices, actions, rewards, next_states, dones, hx, cx = [
        ], [], [], [], [], [], [], []

        for i in indexes:
            state, advice, action, reward, next_state, done, hx_, cx_ = self.all_states[i], self.all_advices[
                i], self.all_actions[i], self.all_rewards[i], self.all_next_states[i], self.all_dones[i], \
                self.all_hx[i], self.all_cx[i]
            states.append(np.array(state, copy=False))
            advices.append(np.array(advice, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            next_states.append(np.array(next_state, copy=False))
            dones.append(np.array(done, copy=False))
            hx.append(hx_)
            cx.append(cx_)

        return (np.array(states), np.asarray(advices)), np.array(actions), np.array(rewards), \
            np.array(next_states), np.array(
                dones), (hx, cx)


def sample_advice():
    advice_list = list(desired_advices)
    advice = advice_list[int(random() * len(advice_list))]
    return advice


def get_state(env):
    return env.get_combined_obs().transpose((2, 0, 1))


def advice_satisfied(advice, color, at_goal, is_lava):
    if at_goal and color == Color_Index.red and advice == Goal.red_goal:
        return True
    if at_goal and color == Color_Index.blue and advice == Goal.blue_goal:
        return True
    if at_goal and color == Color_Index.green and advice == Goal.green_goal:
        return True
    if is_lava and color == Color_Index.red and advice == Goal.red_lava:
        return True
    if is_lava and color == Color_Index.blue and advice == Goal.blue_goal:
        return True
    if is_lava and color == Color_Index.green and advice == Goal.green_goal:
        return True
    return False


def advice_not_satisfied(advice, color, at_goal, is_lava):
    if not at_goal and advice == Goal.avoid_goal:
        return True
    if not is_lava and advice == Goal.avoid_lava:
        return True
    return False


def check_satisfied(advice, satisfied, not_satisfied):
    for condition in advice:
        currently_satisfied = True
        for requirement in condition:
            if should_not_satisfy(requirement) and requirement in not_satisfied:
                currently_satisfied = False
            elif not should_not_satisfy(requirement) and requirement not in satisfied:
                currently_satisfied = False
        if currently_satisfied:
            return True
    return False


def should_not_satisfy(requirement):
    return requirement == Goal.avoid_lava or requirement == Goal.avoid_goal
