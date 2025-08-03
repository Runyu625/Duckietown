from runner import Runner
# from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
import gym
from gym.envs.registration import register


register(
    id='rebalance-v0',
    entry_point='rebalance.rebalance_env:GridRebalanceEnv',
)


if __name__ == '__main__':
    for i in range(1):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        # env = StarCraft2Env(map_name=args.map,
        #                     step_mul=args.step_mul,
        #                     difficulty=args.difficulty,
        #                     game_version=args.game_version,
        #                     replay_dir=args.replay_dir)
        #
        # env_info = env.get_env_info()
        # args.n_actions = env_info["n_actions"]
        # args.n_agents = env_info["n_agents"]
        # args.state_shape = env_info["state_shape"]
        # args.obs_shape = env_info["obs_shape"]
        # args.episode_limit = env_info["episode_limit"]


        env = gym.make('rebalance-v0', n_agents=4, n_orders=4, max_steps=400,
                       grid_shape=(10,10), full_observable=False)
        args.n_actions = env.action_size
        args.n_agents = 4
        args.state_shape = env._full_state_size
        args.obs_shape = env.state_size
        args.episode_limit = 400

        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break

        alg = 'qmix'
        net = 'MLP'
        episodes = int(args.n_steps / 400)
        file_name = f"gridworld_CLDE_qmix_MLP_{episodes}_NoDelos"
        env.close(file_name)
