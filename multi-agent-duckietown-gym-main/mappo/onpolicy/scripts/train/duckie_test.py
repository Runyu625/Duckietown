#!/usr/bin/env python
import sys
import os
# import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from mappo.onpolicy.config import get_config
# from robotic_warehouse.warehouse import Warehouse
# import lbforaging
# from mappo.onpolicy.envs.mpe.MPE_env import MPEEnv
# from mappo.onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
import time
import gym
# from gym_formation import gym_flock
# import gym_flock
from PIL import Image
import argparse
import sys
import gym
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.envs import DuckietownDiscreteEnv
import argparse

# Needed to run this on multi-agent-duckietown-gym/ to fix relative imports on VScode
# export PYTHONPATH=$PYTHONPATH:`pwd`

"""Train script for MPEs."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    # parser.add_argument('--num_agents', type=int,
    #                     default=4, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))



    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Duckietown-udem1-v0")
    parser.add_argument("--map-name", default="lab")
    parser.add_argument("--distortion", default=False, action="store_true")
    parser.add_argument("--camera_rand", default=False, action="store_true")
    parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
    parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
    parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
    parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
    parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--simulated", default=1, type=bool)
    # parser.add_argument("--n_agents", default=3, type=int)
    args = parser.parse_args()

    run_number = 33
    dir_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_args.model_dir = os.path.join(dir_name, 'results', 'duckietown', 'simple_spread', 'mappo','check', f'run{run_number}', 'models')
    # env init
    # envs = make_train_env(all_args)
    if all_args.env_name=='duckietown-discrete':
        all_args.discrete = True
        envs = DuckietownDiscreteEnv(
            seed=args.seed,
            map_name=args.map_name,
            draw_curve=args.draw_curve,
            draw_bbox=args.draw_bbox,
            domain_rand=args.domain_rand,
            frame_skip=args.frame_skip,
            distortion=args.distortion,
            camera_rand=args.camera_rand,
            dynamics_rand=args.dynamics_rand,
            n_agents=all_args.num_agents,
            mappo=True,
            simulated=args.simulated
        )
    else:
        all_args.discrete = False
        envs = DuckietownEnv(
            seed=args.seed,
            map_name=args.map_name,
            draw_curve=args.draw_curve,
            draw_bbox=args.draw_bbox,
            domain_rand=args.domain_rand,
            frame_skip=args.frame_skip,
            distortion=args.distortion,
            camera_rand=args.camera_rand,
            dynamics_rand=args.dynamics_rand,
            n_agents=all_args.num_agents,
            mappo=True,
            simulated=args.simulated
        )


    envs.reset()
    envs.render()

    config = {
        "all_args": all_args,
        "envs": envs,
        # "eval_envs": eval_envs,
        "num_agents": envs.n_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from mappo.onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from mappo.onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    
    runner = Runner(config)
    eval_steps = -1
    
    runner.eval_live(eval_steps)
    # pyglet.clock.schedule_interval(runner.run(), 1.0 / 30)

    # Enter main event loop
    # pyglet.app.run()
    # runner.run()
    
    # post process
    envs.close()
    # if all_args.use_eval and eval_envs is not envs:
    #     eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
