#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown-udem1-v0")
parser.add_argument("--map-name", default="Montreal_loop")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
parser.add_argument("--n_agents", default=1, type=int)
parser.add_argument("--simulated", default=1, type=bool)
args = parser.parse_args()

if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
        n_agents=args.n_agents,
        simulated=args.simulated,
        mappo=True
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

    # if key_handler[key.UP]:
    #     action += np.array([0.44, 0.0])
    # if key_handler[key.DOWN]:
    #     action -= np.array([0.44, 0])
    # if key_handler[key.LEFT]:
    #     action += np.array([0, 1])
    # if key_handler[key.RIGHT]:
    #     action -= np.array([0, 1])
    # if key_handler[key.SPACE]:
    #     action = np.array([0, 0])

    if key_handler[key.NUM_8] or key_handler[key.W]:
        action += np.array([0.44, 0.0])
    if key_handler[key.NUM_9]:
        action += np.array([0.44, -2.0])
    if key_handler[key.NUM_6] or key_handler[key.A]:
        action += np.array([0.0, 1.0])
    if key_handler[key.NUM_3]:
        action += np.array([-0.44, 1.0])
    if key_handler[key.NUM_2] or key_handler[key.S]:
        action += np.array([-0.44, 0.0])
    if key_handler[key.NUM_1]:
        action += np.array([-0.44, -1.0])
    if key_handler[key.NUM_4] or key_handler[key.D]:
        action += np.array([0.0, -1.0])
    if key_handler[key.NUM_7]:
        action += np.array([0.44, 2.0])

    # TODO: Check prob makes sense for physical robot but this was
    # leading to right also going backwards and left also going forward (both should go in the same longitudinal direction)
    # implementing this (prob better on the simulator so it affects all policies and not just manual control)
    # could make the car more realistic by limiting how much spinning without moving we could do,
    # but we should care to avoid the inconsistency they did
    # !!!!

    # v1 = action[0]
    # v2 = action[1]
    # # Limit radius of curvature
    # if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
    #     # adjust velocities evenly such that condition is fulfilled
    #     delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
    #     v1 += delta_v
    #     v2 -= delta_v

    # action[0] = v1
    # action[1] = v2

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5


    action = np.array(action)
    action = np.repeat([action], args.n_agents, axis=0).reshape(1, args.n_agents, -1)
    # obs, reward, done, info = env.step(action)
    obs, reward, done, info = env.step(action)
    # print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward[0]))

    if key_handler[key.RETURN]:

        im = Image.fromarray(obs)

        im.save("screen.png")

    if any(done):
        print("done!")
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
