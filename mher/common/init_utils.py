import gym
from collections import defaultdict

def init_mpi_import():
    '''
    import mpi used for multi-process training 
    '''
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
    return MPI


def init_environment_import():
    '''
    import required environment code base
    '''
    # try:
    #     import pybullet_envs
    # except ImportError:
    #     pybullet_envs = None

    # try:
    #     import roboschool
    # except ImportError:
    #     roboschool = None

    # support mulitworld
    # try:
    #     import multiworld
    #     multiworld.register_all_envs()
    # except ImportError:
    #     multiworld = None

    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        # TODO: solve this with regexes
        try:
            env_type = env.entry_point.split(':')[0].split('.')[-1]
            _game_envs[env_type].add(env.id)
        except:
            pass
    return _game_envs