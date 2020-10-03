import multiprocessing
import os
import sys

import gym
import tensorflow as tf
from gym.wrappers import FilterObservation, FlattenObservation
from mher.common import logger, retro_wrappers, set_global_seeds
from mher.common.init_utils import init_mpi_import
from mher.common.monitor import Monitor
from mher.common.tf_util import get_session
from mher.common.vec_env import VecEnv, VecFrameStack, VecNormalize
from mher.common.vec_env.dummy_vec_env import DummyVecEnv
from mher.common.vec_env.subproc_vec_env import SubprocVecEnv
from mher.common.wrappers import ClipActionsWrapper
from mher.envs.env_utils import get_env_type

MPI = init_mpi_import()

def build_env(args, _game_envs):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args, _game_envs)
    config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    reward_scale = args.reward_scale if hasattr(args, 'reward_scale') else 1
    flatten_dict_observations = alg not in {'her'}
    env = make_vec_env(env_id, env_type, args.num_env or 1, seed, 
                        reward_scale=reward_scale, 
                        flatten_dict_observations=flatten_dict_observations)

    if env_type == 'mujoco':
        env = VecNormalize(env, use_tf=True)
    # build one simple env without vector wrapper
    tmp_env = make_env(env_id, env_type, seed=seed,
                        reward_scale=reward_scale,
                        flatten_dict_observations=flatten_dict_observations,
                        logger_dir=logger.get_dir())

    return env, tmp_env

def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer 
        )
    set_global_seeds(seed)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])


def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, 
            flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    if ':' in env_id:
        import importlib
        import re
        module_name = re.sub(':.*','',env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)

    env = gym.make(env_id, **env_kwargs)
    # if env_id.startswith('Sawyer'):
    #     from mher.algos.multi_world_wrapper import SawyerGoalWrapper
    #     env = SawyerGoalWrapper(env)
    # if (env_id.startswith('Sawyer') or env_id.startswith('Point2D')) and not hasattr(env, '_max_episode_steps'):
    #     env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)
    return env

def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    if reward_scale != 1.0:
        from mher.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env
