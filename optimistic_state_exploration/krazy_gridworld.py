import numpy as np
import sys
from collections import OrderedDict
import os
import random
import copy
from enum import Enum
import cv2
import time


class Color(Enum):
    black = (0, 0, 0)
    white = (255, 255, 255)
    silver = (192, 192, 192)
    gold = (255, 223, 0)
    green = (0, 255, 0)
    brown = (165, 42, 42)
    orange = (255, 140, 0)
    magenta = (255, 0, 255)
    purple = (75, 0, 130)
    blue = (0, 0, 255)
    red = (255, 0, 0)


class Color_Index(object):
    white = 0
    blue = 1
    red = 2
    green = 3


class Color_RGB(object):
    white = (255, 255, 255)
    blue = (0, 0, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    black = (0, 0, 0)


class TileTypes:
    def __init__(self):
        # self.hole = 0
        # self.normal = 1
        # self.goal = 2
        # self.agent = 3
        # self.transporter = 4
        # self.door = 5
        # self.key = 6
        # self.death = 7
        # self.ice = 8
        # self.energy = 9
        self.normal = 0
        self.goal = 1
        self.agent = 2
        self.death = 3

        self.all_tt = [self.normal, self.goal, self.agent, self.death]
        # self.colors = [Color.white, Color.blue, Color.red, Color.green]

    def switch_tile_type_values(self):
        shuffled_idxs = list(range(len(self.all_tt)))
        random.shuffle(shuffled_idxs)
        for i, tt_idx in enumerate(shuffled_idxs):
            self.all_tt[i] = tt_idx

    def reset_colors(self):
        random.shuffle(self.colors)


class Agent:
    def __init__(self, energy_replenish, num_steps_until_energy_needed):
        self.agent_position = [0, 0]
        self.agent_position_init = [0, 0]
        self.has_key = False
        self.dead = False
        # self.dynamics = [lambda x: x+1, lambda x: x-1, lambda x: x+1, lambda x: x-1]
        self.dynamics = [0, 1, 2, 3]
        self.energy_replenish = energy_replenish
        self.energy_init = num_steps_until_energy_needed
        self.num_steps_until_energy_needed = num_steps_until_energy_needed

    def change_dynamics(self):
        random.shuffle(self.dynamics)

    def give_energy(self):
        self.num_steps_until_energy_needed += self.energy_replenish

    def try_step(self, a):
        self.num_steps_until_energy_needed -= 1
        agent_pos = copy.deepcopy(self.agent_position)
        if a == self.dynamics[0]:
            agent_pos[0] += 1
        if a == self.dynamics[1]:
            agent_pos[0] -= 1
        if a == self.dynamics[2]:
            agent_pos[1] += 1
        if a == self.dynamics[3]:
            agent_pos[1] -= 1
        return agent_pos


class GameGrid:
    def __init__(self, task_rng, grid_squares_per_row,
                 tile_types, agent,
                 num_death, energy_sq_perc, ice_sq_perc,
                 num_goals, min_goal_distance, max_goal_distance,
                 num_transporters,
                 ):

        self.task_rng = task_rng
        self.tile_types = tile_types
        self.agent = agent

        self.grid_squares_per_row = grid_squares_per_row
        self.grid_np = None
        self.door_pos = None
        self.num_death = num_death
        self.energy_sq_perc = energy_sq_perc
        self.ice_sq_perc = ice_sq_perc
        self.num_goals = num_goals
        self.goal_squares = None
        self.num_transporters = num_transporters
        self.transporters = None
        self.min_goal_distance = min_goal_distance
        self.max_goal_distance = max_goal_distance

        self.get_new_game_grid()

    def get_new_game_grid(self):
        self.grid_np = np.ones(
            (self.grid_squares_per_row, self.grid_squares_per_row))
        self.grid_color = np.ones(
            (self.grid_squares_per_row, self.grid_squares_per_row))
        self.grid_color *= Color_Index.white
        self.grid_np *= self.tile_types.normal
        self.reset_death_squares()
        self.reset_goal_squares()
        self.game_grid_init = copy.deepcopy(self.grid_np)

    def get_one_non_agent_square(self):
        g = self.task_rng.randint(0, self.grid_squares_per_row - 1, (2,))
        if (g[0] != self.agent.agent_position[0] or g[1] != self.agent.agent_position[1]) and \
                self.grid_np[g[0], g[1]] == self.tile_types.normal:
            return g
        return self.get_one_non_agent_square()

    def reset_death_squares(self):
        ds = []
        visited_ds = []
        colors_needed = [Color_Index.blue, Color_Index.green, Color_Index.red]
        while len(ds) < self.num_death:
            d = self.get_one_non_agent_square()
            if self.grid_np[d[0], d[1]] == self.tile_types.normal and d.tolist() not in visited_ds:
                if self.door_pos is not None:
                    dist_1 = abs(d[0] - self.door_pos[1])
                    dist_2 = abs(d[1] - self.door_pos[2])
                    dist = dist_1 + dist_2
                    if dist > 2:
                        ds.append(d)
                        visited_ds.append(d.tolist())
                else:
                    ds.append(d)
                    visited_ds.append(d.tolist())
        for d in ds:
            self.grid_np[d[0], d[1]] = self.tile_types.death
            if len(colors_needed) > 0:
                color = colors_needed[0]
                colors_needed = colors_needed[1:]
                self.grid_color[d[0], d[1]] = color
            else:
                self.grid_color[d[0], d[1]] = random.randint(0, 3)

    def reset_energy_squares(self):
        ds = []
        num_d_squares = int(self.grid_squares_per_row *
                            self.grid_squares_per_row * self.energy_sq_perc)
        while len(ds) < num_d_squares:
            d = self.get_one_non_agent_square()
            if self.grid_np[d[0], d[1]] == self.tile_types.normal:
                ds.append(d)
        for d in ds:
            self.grid_np[d[0], d[1]] = self.tile_types.energy

    def reset_goal_squares(self):
        gs = []
        goal_squares_loc = []
        colors_needed = [Color_Index.blue, Color_Index.green, Color_Index.red]
        visited_gs = []
        while len(gs) < self.num_goals:
            g = self.get_one_non_agent_square()
            if self.grid_np[g[0], g[1]] == self.tile_types.normal:
                dist_1 = abs(g[1] - self.agent.agent_position[0])
                dist_2 = abs(g[0] - self.agent.agent_position[1])
                dist = dist_1 + dist_2
                if self.min_goal_distance < dist < self.max_goal_distance and g.tolist() not in visited_gs:
                    gs.append(g)
                    visited_gs.append(g.tolist())
        for g in gs:
            self.grid_np[g[0], g[1]] = self.tile_types.goal
            goal_squares_loc.append([g[0], g[1]])
            if len(colors_needed) > 0:
                color = colors_needed[0]
                colors_needed = colors_needed[1:]
                self.grid_color[g[0], g[1]] = color
            else:
                self.grid_color[g[0], g[1]] = color.randint(0, 3)
        self.goal_squares = goal_squares_loc

    def reset_transporter_squares(self):
        gs = []
        for _ in range(self.num_transporters):
            g_1 = self.get_one_non_agent_square()
            g_2 = self.get_one_non_agent_square()
            if self.grid_np[g_1[0], g_1[1]] == self.tile_types.normal:
                if self.grid_np[g_2[0], g_2[1]] == self.tile_types.normal:
                    gs.append([g_1, g_2])
        if len(gs) == self.num_transporters:
            for g in gs:
                for sub_g in g:
                    self.grid_np[sub_g[0], sub_g[1]
                                 ] = self.tile_types.transporter
            self.transporters = gs
        else:
            self.reset_transporter_squares()

    def reset_ice_squares(self):
        ds = []
        num_d_squares = int(self.grid_squares_per_row *
                            self.grid_squares_per_row * self.ice_sq_perc)
        while len(ds) < num_d_squares:
            d = self.get_one_non_agent_square()
            if self.grid_np[d[0], d[1]] == self.tile_types.normal:
                if self.door_pos is not None:
                    dist_1 = abs(d[0] - self.door_pos[1])
                    dist_2 = abs(d[1] - self.door_pos[2])
                    dist = dist_1 + dist_2
                    if dist > 2:
                        ds.append(d)
                else:
                    ds.append(d)
        for d in ds:
            self.grid_np[d[0], d[1]] = self.tile_types.ice

    def is_position_legal(self, pos):
        if (-1 < pos[0] < self.grid_squares_per_row) and (
                -1 < pos[1] < self.grid_squares_per_row):  # in bounds
            # if self.grid_np[pos[1], pos[0]] != self.tile_types.hole:  # not a hole
            #     if self.grid_np[pos[1], pos[0]] != self.tile_types.door:
            #         return True
            #     else:
            #         if self.agent.has_key:
            #             return True
            return True
        return False

    def get_grid_color(self):
        return self.grid_color


class KrazyGridWorld:

    def __init__(self, screen_height=0,
                 grid_squares_per_row=9,
                 one_hot_obs=True,
                 seed=42, task_seed=None, init_pos_seed=None,
                 num_death=3, ice_sq_perc=0.05,
                 num_goals=3, min_goal_distance=2, max_goal_distance=np.inf,
                 num_steps_before_energy_needed=11, energy_replenish=8, energy_sq_perc=0.05,
                 num_transporters=1,
                 sparse_rewards=True,
                 image_obs=False, use_local_obs=False):

        self.grid_squares_per_row = grid_squares_per_row

        if task_seed is None:
            task_seed = seed

        if init_pos_seed is None:
            init_pos_seed = seed

        self.init_pos_rng = np.random.RandomState(init_pos_seed)
        self.task_rng = np.random.RandomState(task_seed)
        random.seed(task_seed)

        self.one_hot_obs = one_hot_obs
        self.image_obs = image_obs
        self.use_local_obs = use_local_obs
        self.screen_dim = (screen_height, screen_height)  # width and height

        self.tile_types = TileTypes()
        self.agent = Agent(num_steps_until_energy_needed=num_steps_before_energy_needed,
                           energy_replenish=energy_replenish)
        self.game_grid = GameGrid(grid_squares_per_row=grid_squares_per_row,
                                  tile_types=self.tile_types,
                                  agent=self.agent,
                                  task_rng=self.task_rng,
                                  num_death=num_death,
                                  energy_sq_perc=energy_sq_perc,
                                  ice_sq_perc=ice_sq_perc,
                                  num_goals=num_goals, min_goal_distance=min_goal_distance,
                                  max_goal_distance=max_goal_distance,
                                  num_transporters=num_transporters)

        self.num_goals_obtained = 0
        self.sparse_reward = sparse_rewards

        self.reset_task()

        self.simple_image_viewer = None
        self.last_im_obs = None

    def reset(self, reset_agent_start_pos=False, reset_board=False,
              reset_colors=False, reset_dynamics=False):
        self.agent.dead = False
        self.agent.agent_position = copy.deepcopy(
            self.agent.agent_position_init)
        self.agent.num_steps_until_energy_needed = copy.deepcopy(
            self.agent.energy_init)
        self.num_goals_obtained = 0
        self.game_grid.grid_np = copy.deepcopy(self.game_grid.game_grid_init)
        # if reset_colors:
        #     self.tile_types.reset_colors()
        if reset_dynamics:
            self.agent.change_dynamics()
        if reset_board:
            self.reset_task()
        if reset_agent_start_pos:
            self.reset_agent_start_position()
        return self.get_obs()

    def reset_task(self):
        # reset the entire board and agent start position, generating a new MDP.
        self.game_grid.get_new_game_grid()
        # self.reset_agent_start_position()

    def reset_agent_start_position(self):
        # keep the previous board but update the agents starting position.
        # keeps the previous MDP but samples x_0.
        new_start = self.game_grid.get_one_non_agent_square()
        self.agent.agent_position = new_start
        self.agent.agent_position_init = new_start

    def get_obs(self):
        if self.image_obs:
            return self.get_img_obs()
        else:
            return None

    def get_color_obs(self):
        grid_color = copy.deepcopy(self.game_grid.grid_color)
        grid_color = grid_color.astype(np.uint8)
        if self.one_hot_obs:
            n_values = np.max(grid_color) + 1
            # n_values = 4
            grid_color = np.eye(n_values)[grid_color]
            return grid_color[:,:,1:]

    def get_combined_obs(self):
        obs_color = self.get_color_obs()
        obs_state = self.get_state_obs()
        new_grid = np.zeros((self.grid_squares_per_row, self.grid_squares_per_row, 7))
        new_grid[:,:,0] = obs_color[:,:,0]*obs_state[:,:,0]
        new_grid[:,:,1] = obs_color[:,:,0]*obs_state[:,:,2]
        new_grid[:,:,2] = obs_color[:,:,1]*obs_state[:,:,0]
        new_grid[:,:,3] = obs_color[:,:,1]*obs_state[:,:,2]
        new_grid[:,:,4] = obs_color[:,:,2]*obs_state[:,:,0]
        new_grid[:,:,5] = obs_color[:,:,2]*obs_state[:,:,2]
        new_grid[:,:,6] = obs_state[:,:,1]
        expand_multiplier = 4
        expanded_grid = np.zeros((expand_multiplier * self.grid_squares_per_row, \
            expand_multiplier * self.grid_squares_per_row, 7))
        for i in range(expand_multiplier * self.grid_squares_per_row):
            for j in range(expand_multiplier * self.grid_squares_per_row):
                for k in range(7):
                    expanded_grid[i, j,k] = new_grid[int(i/expand_multiplier), int(j/expand_multiplier), k]
        return expanded_grid

    def get_img_pyplot_obs(self):
        grid_color = copy.deepcopy(self.game_grid.grid_color)
        grid_color = grid_color.astype(np.uint8)
        if self.one_hot_obs:
            grid_img = np.zeros(
                (self.grid_squares_per_row, self.grid_squares_per_row, 3))
            for i in range(self.grid_squares_per_row):
                for j in range(self.grid_squares_per_row):
                    if grid_color[i, j] == Color_Index.white:
                        grid_img[i, j] = Color_RGB.white
                    elif grid_color[i, j] == Color_Index.red:
                        grid_img[i, j] = Color_RGB.red
                    elif grid_color[i, j] == Color_Index.green:
                        grid_img[i, j] = Color_RGB.green
                    elif grid_color[i, j] == Color_Index.blue:
                        grid_img[i, j] = Color_RGB.blue
            agent_pos = self.agent.agent_position
            grid_img[agent_pos[0], agent_pos[1]] = Color_RGB.black
            return grid_img
        return None

    def step(self, a, render=False):
        if self.agent.dead is False:
            proposed_step = self.agent.try_step(a)
            if self.game_grid.is_position_legal(proposed_step):
                self.agent.agent_position = proposed_step
            self.check_dead()
            self.check_at_goal()
            # self.check_at_energy()
            # self.check_at_transporter()

            #  this shit handles the ice squares
            # while True:
            #     if self.check_at_ice_square() is False:
            #         break
            #     else:
            #         #  don't take energy for going over ice.
            #         self.agent.num_steps_until_energy_needed += 1
            #         proposed_step_nu = self.agent.try_step(a)
            #         if self.game_grid.is_position_legal(proposed_step_nu):
            #             self.step(a)
            #         else:
            #             break

            # if self.agent.num_steps_until_energy_needed < 1:
            #     self.agent.dead = True

            if render:
                self.render()

            agent_pos = self.agent.agent_position
        return self.get_obs(), self.get_reward(), self.agent.dead, dict({'color': self.game_grid.grid_color[agent_pos[0], agent_pos[1]]})

    def check_dead(self):
        agent_pos = self.agent.agent_position
        game_grid = self.game_grid.grid_np
        if game_grid[agent_pos[0], agent_pos[1]] == self.tile_types.death:
            self.agent.dead = True

    def is_dead(self):
        return self.agent.dead

    def check_at_goal(self):
        if self.game_grid.grid_np[self.agent.agent_position[0], self.agent.agent_position[1]] == self.tile_types.goal:
            # self.game_grid.grid_np[self.agent.agent_position[0],
            #                        self.agent.agent_position[1]] = self.tile_types.normal
            self.num_goals_obtained += 1

    def at_goal(self):
        if self.game_grid.grid_np[self.agent.agent_position[0], self.agent.agent_position[1]] == self.tile_types.goal:
            return True
        return False

    def check_at_energy(self):
        if self.game_grid.grid_np[self.agent.agent_position[0], self.agent.agent_position[1]] == self.tile_types.energy:
            self.game_grid.grid_np[self.agent.agent_position[0],
                                   self.agent.agent_position[1]] = self.tile_types.normal
            self.agent.give_energy()

    def check_at_transporter(self):
        transport_sq = None
        if self.game_grid.grid_np[self.agent.agent_position[0], self.agent.agent_position[1]] == self.tile_types.transporter:
            for tr in self.game_grid.transporters:
                if self.agent.agent_position[0] == tr[0][0] and self.agent.agent_position[1] == tr[0][1]:
                    transport_sq = tr[1]
                elif self.agent.agent_position[0] == tr[1][0] and self.agent.agent_position[1] == tr[1][1]:
                    transport_sq = tr[0]
            if transport_sq is not None:
                self.agent.agent_position = [transport_sq[0], transport_sq[1]]

    def check_at_ice_square(self):
        if self.game_grid.grid_np[self.agent.agent_position[0], self.agent.agent_position[1]] == self.tile_types.ice:
            return True
        return False

    def render(self):
        if self.simple_image_viewer is None:
            from gym.envs.classic_control.rendering import SimpleImageViewer
            self.simple_image_viewer = SimpleImageViewer()
        im_obs = self.get_img_obs()
        self.simple_image_viewer.imshow(im_obs)
        time.sleep(0.075)

    def get_state_obs(self, flatten=False):
        grid_np = copy.deepcopy(self.game_grid.grid_np)
        agent_p = self.agent.agent_position
        # grid_np[agent_p[0], agent_p[1]] = self.tile_types.agent
        grid_np = grid_np.astype(np.uint8)
        # agent_p = np.array(self.agent.agent_position)
        if self.one_hot_obs:
            n_values = np.max(grid_np) + 1
            # n_values = 4
            grid_np = np.eye(n_values)[grid_np]
            # agent_p_temp = np.zeros((self.game_grid.grid_squares_per_row, self.game_grid.grid_squares_per_row, 1))
            # agent_p_temp[agent_p[0], agent_p[1], :] = 1
            grid_np[agent_p[0], agent_p[1], self.tile_types.agent] = 1

        if self.use_local_obs:
            neighbors = []
            x, y = self.agent.agent_position
            for _i, _j in [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]:
                i, j = (_i + x, _j + y)
                if 0 <= i < self.game_grid.grid_squares_per_row and 0 <= j < self.game_grid.grid_squares_per_row:
                    neighbors.append([j, i])
                else:
                    neighbors.append(None)

            grid_np = np.array(neighbors)

        if flatten:
            return grid_np.flatten()
        else:
            return grid_np[:,:,1:]

    def get_img_obs(self):
        grid_np = copy.deepcopy(self.game_grid.grid_np)
        grid_np[self.agent.agent_position[0],
                self.agent.agent_position[1]] = self.tile_types.agent
        fake_img = np.zeros(
            (self.game_grid.grid_squares_per_row, self.game_grid.grid_squares_per_row, 3))
        for i in range(len(self.tile_types.all_tt)):
            is_grid_sq_color_i = grid_np == self.tile_types.all_tt[i]
            one_idxs = is_grid_sq_color_i.astype(int)
            one_idxs = np.tile(np.expand_dims(one_idxs, -1), 3)
            one_idxs = one_idxs * np.array(self.tile_types.colors[i].value)
            fake_img += one_idxs

        if self.use_local_obs:
            neighbors = []
            x, y = self.agent.agent_position
            valid_idxs = np.zeros_like(fake_img)
            valid_idxs[x, y] = 1.0
            for _i, _j in [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]:
                i, j = (_i + x, _j + y)
                if 0 <= i < self.game_grid.grid_squares_per_row and 0 <= j < self.game_grid.grid_squares_per_row:
                    # neighbors.append([j, i])
                    valid_idxs[i, j] = 1.0
                else:
                    neighbors.append(None)
            fake_img *= valid_idxs

        res = cv2.resize(fake_img,
                         dsize=(256, 256),
                         interpolation=cv2.INTER_NEAREST)
        res = res.astype(np.uint8)
        return res

    def get_reward(self):
        if self.sparse_reward:
            return 0 + self.num_goals_obtained
        else:
            rew = 0
            for goal in self.game_grid.goal_squares:
                dist_1 = abs(goal[0] - self.agent.agent_position[0])
                dist_2 = abs(goal[1] - self.agent.agent_position[1])
                rew = rew + dist_1 + dist_2
            rew = -1.0*rew
            rew = rew + 3.0*self.num_goals_obtained
            return rew

    def close(self):
        self.simple_image_viewer.close()


def run_grid():
    kw = KrazyGridWorld(screen_height=256, grid_squares_per_row=10,
                        one_hot_obs=False, use_local_obs=False, image_obs=True,
                        seed=42, task_seed=68, init_pos_seed=70,
                        num_goals=3, max_goal_distance=np.inf, min_goal_distance=2,
                        death_square_percentage=0.08,
                        num_steps_before_energy_needed=50, energy_sq_perc=0.05, energy_replenish=8,
                        num_transporters=1,
                        ice_sq_perc=0.05,
                        )

    for i in range(1000):
        o, r, d, _ = kw.step(np.random.randint(0, 4, 1), render=True)
        obs = kw.get_state_obs()
        if d is True:
            kw.reset(reset_board=False, reset_colors=False,
                     reset_agent_start_pos=False, reset_dynamics=False)
    kw.close()


if __name__ == "__main__":
    run_grid()
