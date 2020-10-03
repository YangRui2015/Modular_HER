import multiworld
import gym
import numpy as np
from gym.core import Wrapper
import copy

# for point env 
class PointGoalWrapper(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        import pdb; pdb.set_trace()
        obs_dict, reward, done, info = self.env.step(action)
        obs = {
            'observation':obs_dict['observation'],
            'desired_goal':obs_dict['desired_goal'],
            'achieved_goal':obs_dict['achieved_goal']
        }
        return obs, reward, done, info
    
    def render(self, mode='human'):
        return self.env.render()
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        obs = {
            'state_achieved_goal': achieved_goal,
            'state_desired_goal':desired_goal
        }
        action = np.array([])
        return self.env.compute_reward(action, obs)

    def sample_goal(self):
        goal_dict = self.env.sample_goal()
        return goal_dict['desired_goal']

# for sawyer env
class SawyerGoalWrapper(Wrapper):
    reward_type_dict = {
        'dense':'hand_distance',
        'sparse':'hand_success'
    }
    observation_keys = ['observation', 'desired_goal', 'achieved_goal']
        
    def __init__(self, env, reward_type='sparse'):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.action_space = env.action_space
        # observation
        for key in list(env.observation_space.spaces.keys()):
            if key not in self.observation_keys:
                del env.observation_space.spaces[key]

        self.observation_space = env.observation_space
        self.reward_type = reward_type
        self.env.reward_type = self.reward_type_dict[self.reward_type]
        # self.env.indicator_threshold = 0.03
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = {
            'observation':obs_dict['observation'],
            'desired_goal':obs_dict['desired_goal'],
            'achieved_goal':obs_dict['achieved_goal']
        }
        if 'hand_success' in info.keys():
            info['is_success'] = info['hand_success']
        if 'success' in info.keys():
            info['is_success'] = info['success']
        import pdb; pdb.set_trace()
        return obs, reward, done, info
    
    def render(self, mode='human'):
        return self.env.render()
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        obs = {
            'state_achieved_goal': achieved_goal,
            'state_desired_goal':desired_goal
        }
        action = np.array([])
        return self.env.compute_rewards(action, obs)

    def sample_goal(self):
        goal_dict = self.env.sample_goal()
        return goal_dict['desired_goal']
