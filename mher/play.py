# DEPRECATED, use --play flag to mher.run instead
import pickle

import click
import numpy as np

import mher.config as config
from mher.rollouts.rollout import RolloutWorker
from mher.common import logger, set_global_seeds
from mher.common.vec_env import VecEnv


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)

def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)
    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
    }
    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts()

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


# playing with a model and an environment
def play(model, env, episodes=1):
    logger.log("Running trained model")
    obs = env.reset()
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = np.zeros((episodes, env.num_envs)) if isinstance(env, VecEnv) else np.zeros((episodes, 1))
    ep_num = 0
    while ep_num < episodes:
        actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew[ep_num] += rew
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            logger.log('episode_rew={}'.format(episode_rew[ep_num]))
            ep_num += 1
            obs = env.reset()
    average_reward = np.mean(episode_rew)
    logger.log('Total average test reward:{}'.format(average_reward))
    return average_reward
    

if __name__ == '__main__':
    main()
