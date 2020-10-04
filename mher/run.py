import copy
import multiprocessing
import os
import os.path as osp
import re
import sys

import gym
import numpy as np
import tensorflow as tf

from mher import config
from mher.rollouts.rollout import RolloutWorker
from mher.common import logger, set_global_seeds, tf_util
from mher.common.cmd_util import preprocess_kwargs
from mher.common.import_util import get_alg_module
from mher.common.init_utils import init_environment_import, init_mpi_import
from mher.common.logger import configure_logger
from mher.envs.make_env_utils import build_env
from mher.play import play
from mher.train import train

MPI = init_mpi_import()
_game_envs = init_environment_import()

def prepare(args):
    ## make save dir
    if args.save_path:
        os.makedirs(os.path.expanduser(args.save_path), exist_ok=True)
    # configure logger, disable logging in child MPI processes (with rank > 0)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        configure_logger(args.log_path)
    else:
        configure_logger(args.log_path, format_strs=[])
    # Seed everything.
    rank = MPI.COMM_WORLD.Get_rank()
    rank_seed = args.seed + 1000000 * rank if args.seed is not None else None
    set_global_seeds(rank_seed)
    return rank

def main(args): 
    # process argprase and parameters
    args, extra_args = preprocess_kwargs(args)
    rank = prepare(args)
    env, tmp_env = build_env(args, _game_envs)
    params = config.process_params(env, tmp_env, rank, args, extra_args)
    dims = config.configure_dims(tmp_env, params)

    # define objects
    sampler = config.configure_sampler(dims, params)
    buffer = config.configure_buffer(dims, params, sampler)
    policy = config.configure_ddpg(dims=dims, params=params, buffer=buffer)
    rollout_params, eval_params = config.configure_rollout(params)

    if args.load_path is not None:
        tf_util.load_variables(args.load_path)

    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(env, policy, dims, logger, **eval_params)

    n_epochs = config.configure_epoch(args.num_epoch, params)
    policy = train(
        policy=policy,
        rollout_worker=rollout_worker,
        save_path=args.save_path, 
        evaluator=evaluator, 
        n_epochs=n_epochs, 
        n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], 
        n_batches=params['n_batches'],
        policy_save_interval=params['policy_save_interval'], 
        random_init=params['random_init']
        )

    if args.play_episodes or args.play_no_training:
        play(policy, env, episodes=args.play_episodes)
    env.close()
    

if __name__ == '__main__':
    main(sys.argv)
