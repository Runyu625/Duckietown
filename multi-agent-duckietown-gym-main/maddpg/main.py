import pyglet
from pyglet.window import key
import numpy as np
import pandas as pd
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from gym_duckietown.envs import DuckietownEnv
import argparse
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import math


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

##### Train parameters #####
scenario = 'montreal_loop_epsilon_no_kinematics_no_invalid_simplenet'
evaluate = False
RENDER = True
PRINT_INTERVAL = 100
NUM_STEPS_LEARN = 400
N_GAMES = 10000
MAX_STEPS = 1000
EPSILON = 1
EPSILON_DECREASE = 0.9995
MIN_EPSILON = 0.02
############################
score_history = []
step_history = []
best_score = 0
total_steps = 0


class Maddpg_runner():
    def __init__(self, ):
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

        # Create model output directory
        self.chkpt_dir = os.path.join('output', scenario)
        if not os.path.isdir(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        self.env = DuckietownEnv(
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
            mappo=False,
            random_start_tile=False,
            simulated=args.simulated,
            kinematics=False
        )
        self.total_steps = total_steps
        self.n_agents = args.n_agents
        actor_dims = []
        for i in range(self.n_agents):
            actor_dims.append(self.env.observation_space[i].shape[0])
        critic_dims = sum(actor_dims)

        # action space is a list of arrays, assume each agent has same action space
        n_actions = self.env.action_space[0].shape[0]
        self.maddpg_agents = MADDPG(actor_dims, critic_dims, self.n_agents, n_actions,
                                    fc1=8, fc2=8, fc3=32,
                                    alpha=0.0001, beta=0.001, scenario=scenario,
                                    chkpt_dir=self.chkpt_dir)

        self.memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, self.n_agents, batch_size=512)
        self.render = RENDER
        self.epsilon = EPSILON
        self.epsilon_decrease = EPSILON_DECREASE
        self.min_epsilon = MIN_EPSILON


    def plot_rewards(self, df, path_save):
        """Plot learning curve (episode-reward)"""
        plt.rcParams.update({'font.size': 17})
        df['rolling_mean'] = df['Reward'].rolling(100).mean()
        plt.figure(figsize=(15, 8))
        plt.close()
        plt.figure()
        # plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
        plot = df[['Reward','rolling_mean']].plot(linewidth=1.5, figsize=(15, 8))
        plot.set_xlabel('Episode')
        plot.set_ylabel('Reward')
        fig = plot.get_figure()
        plt.legend().set_visible(True)
        fig.savefig(path_save)

    def run(self):
        best_score = 0
        total_steps = 0
        if evaluate:
            self.maddpg_agents.load_checkpoint()

        for i in tqdm(range(N_GAMES)):
            obs = self.env.reset()
            score = 0
            done = [False] * self.n_agents
            episode_step = 0
            while not any(done):
                if evaluate:
                    self.env.render()
                    # time.sleep(0.1) # to slow down the action for the video

                # Exploration
                if random.random() < self.epsilon:
                    if self.env.kinematics:
                        actions = [np.array([random.random(), 2*(random.random() - 0.5)])]
                    else:
                        actions = [np.array([random.random(), random.random()])]
                else:
                    actions = self.maddpg_agents.choose_action(obs)

                # Scale steering angle to [-pi,pi] range
                if self.env.kinematics:
                    actions[0][1] *= math.pi

                obs_, reward, done, info = self.env.step(actions)

                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)

                if episode_step >= MAX_STEPS:
                    done = [True] * self.n_agents

                self.memory.store_transition(obs, state, actions, reward, obs_, state_, done)
                obs = obs_
                score += np.sum(reward)
                self.total_steps += 1
                episode_step += 1

                # print('Step: ' + str(episode_step) + ' reward: ' + str(reward))

                if self.render and episode_step % 6 == 0:
                    self.env.render()

            if not evaluate:
                self.maddpg_agents.learn(self.memory)

            # print('** Episode: ' + str(i) + ' score: ' + str(score))
            score_history.append(score)
            step_history.append(episode_step)
            avg_score = np.mean(score_history[-100:])
            if avg_score > best_score:
                best_score = avg_score

            if i % PRINT_INTERVAL == 0 and i > 0:
                print('%%% episode', i, 'average score {:.1f}'.format(avg_score), ' epsilon', self.epsilon)

            self.epsilon *= self.epsilon_decrease
            self.epsilon = max(self.min_epsilon, self.epsilon)

        # Save models
        results_df = pd.DataFrame(list(zip(score_history, step_history)), columns=['Reward', 'Steps'])
        if not evaluate:
            self.maddpg_agents.save_checkpoint()
            results_df.to_csv(self.chkpt_dir + '/stats_train.csv')
            self.plot_rewards(results_df, self.chkpt_dir + '/plot_train.png')
        else:
            results_df.to_csv(self.chkpt_dir + '/stats_eval.csv')
            self.plot_rewards(results_df, self.chkpt_dir + '/plot_eval.png')


if __name__ == '__main__':

    runner = Maddpg_runner()

    # Enter main event loop
    pyglet.clock.schedule_interval(runner.run(), 1.0 / 5)

    #pyglet.app.run()

