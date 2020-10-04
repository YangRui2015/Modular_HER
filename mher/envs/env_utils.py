'''
Util tools for environments
'''
import re

import gym


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b

def g_to_ag(o, env_id):
    if env_id == 'FetchReach':
        ag = o[:,0:3]
    elif env_id in ['FetchPush','FetchSlide', 'FetchPickAndPlace']:
        ag = o[:,3:6]
    else:
        raise NotImplementedError
    return ag

CACHED_ENVS = {}
def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]

def get_rewardfun(params, tmp_env):
    tmp_env.reset()
    def reward_fun(ag_2, g, info):  # vectorized
        return tmp_env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)
    return reward_fun

def get_env_type(args, _game_envs):
    env_id = args.env
    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        try:
            env_type = env.entry_point.split(':')[0].split('.')[-1]
            _game_envs[env_type].add(env.id)  # This is a set so add is idempotent
        except:
            pass

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types {}'.format(env_id, _game_envs.keys())

    return env_type, env_id

def obs_to_goal_fun(env):
    # only support Fetchenv and Handenv now
    from gym.envs.robotics import FetchEnv, hand_env
    from multiworld.envs.mujoco.sawyer_xyz import (sawyer_push_nips,
                                                   sawyer_reach)
    from multiworld.envs.pygame import point2d

    if isinstance(env.env, FetchEnv):
        obs_dim = env.observation_space['observation'].shape[0]
        goal_dim = env.observation_space['desired_goal'].shape[0]
        temp_dim = env.sim.data.get_site_xpos('robot0:grip').shape[0]
        def obs_to_goal(observation):
            observation = observation.reshape(-1, obs_dim)
            if env.has_object:
                goal = observation[:, temp_dim:temp_dim + goal_dim]
            else:
                goal = observation[:, :goal_dim]
            return goal.copy()
    elif isinstance(env.env, hand_env.HandEnv):
        goal_dim = env.observation_space['desired_goal'].shape[0]
        def obs_to_goal(observation):
            goal = observation[:, -goal_dim:]
            return goal.copy()
    elif isinstance(env.env.env, point2d.Point2DEnv):
        def obs_to_goal(observation):
            return observation.copy()
    elif isinstance(env.env.env, sawyer_push_nips.SawyerPushAndReachXYEnv):
        assert env.env.env.observation_space['observation'].shape == env.env.env.observation_space['achieved_goal'].shape, \
            "This environment's observation space doesn't equal goal space"
        def obs_to_goal(observation):
            return observation
    elif isinstance(env.env.env, sawyer_reach.SawyerReachXYZEnv):
        def obs_to_goal(observation):
            return observation
    else:
        import pdb; pdb.set_trace()
        raise NotImplementedError('Do not support such type {}'.format(env))
        
    return obs_to_goal
