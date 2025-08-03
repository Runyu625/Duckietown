# coding=utf-8
import numpy as np
from gym import spaces

from ..simulator import Simulator
from .. import logger
from enum import Enum


# class Action(Enum):
#     NOOP = 0
#     N = 1
#     NE = 2
#     E= 3
#     SE = 4
#     S = 5
#     SW = 6
#     W = 7
#     NW = 8

class Action(Enum):
    NOOP = 0
    N = 1
    NE = 2
    NW = 3
    S = 4

# class Action(Enum):
#     NOOP = 0
#     N = 1
#     # S = 2

class DuckietownDiscreteEnv(Simulator):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def   __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, **kwargs):
        Simulator.__init__(self, **kwargs)
        logger.info("using DuckietownEnv")

        self.wheel_vel_min = -1
        self.wheel_vel_max = 1
        self.n_features = 4
        # self.n_actions = 2
        # self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        # self.action_space = spaces.Box(low=self.wheel_vel_min, high=self.wheel_vel_max, shape=(self.n_actions, self.n_agents),
        #            dtype=np.float32)
        # self.observation_space = spaces.Box(low=-1000, high=1000, shape=(self.n_agents, self.n_features),
        #                                     dtype=np.float32)
        # Action and observation space for mappo
        sa_action_space = [len(Action)]
        sa_action_space = spaces.Discrete(sa_action_space[0])
        self.action_size = len(Action)
        self.action_space = spaces.Tuple(tuple(self.n_agents * [sa_action_space]))
        act_spaces = []
        ma_spaces = []
        state_spaces = []
        for i in range(self.n_agents):

            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.n_features,),
                    dtype=np.float32,
                )
            ]
            state_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.n_features*self.n_agents,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))
        self.share_observation_space = spaces.Tuple(tuple(state_spaces))

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

    def action_to_vel(self, action):
        if action == 0:
            action = np.array([0, 0])
        elif action == 1:
            action = np.array([0.44, 0.0])
        elif action == 2:
            action = np.array([0.44, -2.0])
        elif action == 3:
            action = np.array([0.44, 2.0])
        elif action == 4:
            action = np.array([-0.44, 0])
        # if action == 0:
        #     action = np.array([0, 0])
        # elif action == 1:
        #     action = np.array([0.44, 0.0])
        # elif action == 2:
        #     action = np.array([0.44, -1.0])
        # elif action == 3:
        #     action = np.array([0.0, -1.0])
        # elif action == 4:
        #     action = np.array([-0.44, 1.0])
        # elif action == 5:
        #     action = np.array([-0.44, 0.0])
        # elif action == 6:
        #     action = np.array([-0.44, -1.0])
        # elif action == 7:
        #     action = np.array([0.0, -1.0])
        # elif action == 8:
        #     action = np.array([0.44, 1.0])

        return action


    def step(self, action):
        # vel, angle = action
        actions = []
        for v in range(self.n_agents):
            if self.mappo:
                act = np.argmax(action[0, v, :])
            else:
                act = action[v]
            actions.append(self.action_to_vel(act))

        action = np.array(actions).reshape(1, self.n_agents, -1)

        angle = action[0][:, 1]
        vel = action[0][:, 0]

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l



        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv


        # limiting output to limit, which is 1.0 for the duckiebot
        # u_r_limited = max(min(u_r, self.limit), -self.limit)
        # u_l_limited = max(min(u_l, self.limit), -self.limit)

        u_r_limited = u_r.clip(-self.limit, self.limit)
        u_l_limited = u_l.clip(-self.limit, self.limit)

        vels = np.array([u_l_limited, u_r_limited]).T

        # obs, reward, done, info = Simulator.step(self, [vels])
        # vels = [vels]*self.n_agents

        obs, reward, done, info = Simulator.step(self, vels)
        mine = {}
        mine["k"] = self.k
        mine["gain"] = self.gain
        mine["train"] = self.trim
        mine["radius"] = self.radius
        mine["omega_r"] = omega_r
        mine["omega_l"] = omega_l
        # info["DuckietownEnv"] = mine
        # return obs, reward, done, info
        return obs, reward, done, info

class DuckietownLF(DuckietownDiscreteEnv):
    """
    Environment for the Duckietown lane following task with
    and without obstacles (LF and LFV tasks)
    """

    def __init__(self, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)

    def step(self, action):
        obs, reward, done, info = DuckietownDiscreteEnv.step(self, action)
        return obs, reward, done, info


class DuckietownNav(DuckietownDiscreteEnv):
    """
    Environment for the Duckietown navigation task (NAV)
    """

    def __init__(self, **kwargs):
        self.goal_tile = None
        DuckietownDiscreteEnv.__init__(self, **kwargs)

    def reset(self, segment=False):
        DuckietownNav.reset(self)

        # Find the tile the agent starts on
        start_tile_pos = self.get_grid_coords(self.cur_pos)
        start_tile = self._get_tile(*start_tile_pos)

        # Select a random goal tile to navigate to
        assert len(self.drivable_tiles) > 1
        while True:
            tile_idx = self.np_random.randint(0, len(self.drivable_tiles))
            self.goal_tile = self.drivable_tiles[tile_idx]
            if self.goal_tile is not start_tile:
                break

    def step(self, action):
        obs, reward, done, info = DuckietownNav.step(self, action)

        info["goal_tile"] = self.goal_tile

        # TODO: add term to reward based on distance to goal?

        cur_tile_coords = self.get_grid_coords(self.cur_pos)
        cur_tile = self._get_tile(*cur_tile_coords)

        if cur_tile is self.goal_tile:
            done = True
            reward = 1000

        return obs, reward, done, info
