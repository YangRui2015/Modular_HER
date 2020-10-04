import gym
import numpy as np


class Sampler:
    def __init__(self, T, reward_fun, batch_size):
        self.T = T
        self.reward_fun = reward_fun
        self.batch_size = batch_size
    
    def _sample_transitions(self, episode_batch):
        num_episodes = episode_batch['u'].shape[0]
        episode_idxs = np.random.randint(0, num_episodes, self.batch_size)
        t_samples = np.random.randint(self.T, size=self.batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}
        info = {
            'num_episodes': num_episodes,
            'episode_idxs':episode_idxs,
            't_samples':t_samples
        }
        return transitions, info

    def recompute_reward(self, transitions):
        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value
        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        return self.reward_fun(**reward_params)
    
    def reshape_transitions(self, transitions):
        transitions = {k: transitions[k].reshape(self.batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}
        assert(transitions['u'].shape[0] == self.batch_size)
        return transitions

    def sample(self, episode_batch):
        pass


class RandomSampler(Sampler):
    def sample(self, episode_batch):
        transitions, _ = self._sample_transitions(episode_batch)
        transitions['r'] = self.recompute_reward(transitions)
        transitions = self.reshape_transitions(transitions)
        return transitions

class HER_Sampler(Sampler):
    valid_strategy = ['future', 'last', 'random', 'episode']
    def __init__(self, T, reward_fun, batch_size, strategy, replay_k):
        super(HER_Sampler, self).__init__(T, reward_fun, batch_size)
        self.strategy = strategy
        self.replay_k = replay_k
        if strategy in self.valid_strategy:
            self.relabel_p = 1 - (1. / (1 + replay_k))
        else:
            self.relabel_p = 0
    
    def sample(self, episode_batch):
        transitions, info = self._sample_transitions(episode_batch)
        episode_idxs, t_samples, num_episodes = info['episode_idxs'], info['t_samples'], info['num_episodes']
        if self.strategy in self.valid_strategy:
            her_indexes = (np.random.uniform(size=self.batch_size) < self.relabel_p)
            
        if self.strategy == 'future':
            future_offset = (np.random.uniform(size=self.batch_size) * (self.T - t_samples)).astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            transitions['g'][her_indexes] = future_ag
        elif self.strategy == 'last':
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], -1]
            transitions['g'][her_indexes] = future_ag
        elif self.strategy == 'episode':
            random_t_samples = np.random.randint(self.T, size=self.batch_size)[her_indexes]
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], random_t_samples]
            transitions['g'][her_indexes] = future_ag
        elif self.strategy == 'random':
            random_episode_idxs = np.random.randint(0, num_episodes, self.batch_size)[her_indexes]
            random_t_samples = np.random.randint(self.T, size=self.batch_size)[her_indexes]
            future_ag = episode_batch['ag'][random_episode_idxs, random_t_samples]
            transitions['g'][her_indexes] = future_ag
        else: # no relabeling
            pass

        transitions['r'] = self.recompute_reward(transitions)
        transitions = self.reshape_transitions(transitions)
        return transitions




def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, obs_to_goal_fun=None, no_her=False):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    if no_her:
        print( '*' * 10 + 'Do not use HER in this method' + '*' * 10)

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout

        # Select which episodes and time steps to use. 
        # np.random.randint doesn't contain the last one, so comes from 0 to roolout_batch_size-1
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # # precomputing rewrad before relabeling
        # info = {}
        # for key, value in transitions.items():
        #     if key.startswith('info_'):
        #         info[key.replace('info_', '')] = value
        # reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        # reward_params['info'] = info
        # temp_reward = reward_fun(**reward_params)
        # # print(temp_reward.mean())
        
        if not no_her:
            her_indexes = (np.random.uniform(size=batch_size) < future_p)
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]

            # Replace goal with achieved goal but only for the previously-selected
            # HER transitions (as defined by her_indexes). For the other transitions,
            # keep the original goal.
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    def _sample_nstep_her_transitions(episode_batch, batch_size_in_transitions, steps, gamma):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        if np.random.random() < 0.1:
            print('nstep sampler')
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout

        assert steps < T, 'Steps should be much less than T.'

        # Select which episodes and time steps to use. 
        # np.random.randint doesn't contain the last one, so comes from 0 to roolout_batch_size-1
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)

        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        n_step_ags = np.zeros((batch_size, steps, episode_batch['ag'].shape[-1]))
        n_step_reward_mask = np.ones((batch_size, steps)) * np.array([pow(gamma,i) for i in range(steps)])

        for i in range(steps):
            i_t_samples = t_samples + i
            n_step_reward_mask[:,i][np.where(i_t_samples > T - 1)] = 0
            i_t_samples[i_t_samples > T-1] = T-1   # last state to compute reward
            n_step_ags[:,i,:] = episode_batch['ag_2'][episode_idxs, i_t_samples]

        i_t_samples = t_samples + steps # last state to observe
        i_t_samples[i_t_samples > T] = T
        n_step_os = episode_batch['o'][episode_idxs, i_t_samples]
        n_step_gamma = np.ones((batch_size,1)) * pow(gamma, steps)

        # use inverse order to find the first zero in each row of reward mask
        for i in range(steps-1, 0, -1):
            n_step_gamma[np.where(n_step_reward_mask[:,i] == 0)] = pow(gamma, i)

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            her_indexes = (np.random.uniform(size=batch_size) < future_p)
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]

            # Replace goal with achieved goal but only for the previously-selected
            # HER transitions (as defined by her_indexes). For the other transitions,keep the original goal.
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            # goal noisy can be used to make better fit of Q(s,g,a)
            transitions['g'][her_indexes] = future_ag  #+ 0.005 * np.clip(np.random.randn(*future_ag.shape), -1,1) 
        
        n_step_gs = transitions['g'].repeat(steps, axis=0)

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        ags = n_step_ags.reshape((batch_size * steps, -1))
        gs = n_step_gs

        reward_params = {
            'ag_2':ags,
            'g':gs
        }
        reward_params['info'] = info
        n_step_reward = reward_fun(**reward_params)

        transitions['r'] = (n_step_reward.reshape((batch_size, steps)) * n_step_reward_mask).sum(axis=1).copy()
        transitions['o_2'] = n_step_os.copy()
        transitions['gamma'] = n_step_gamma.copy()
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions
    
    def _sample_nstep_correct_her_transitions(episode_batch, batch_size_in_transitions, steps, gamma, Q_pi_fun, Q_fun):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        if np.random.random() < 0.1:
            print('correct n step')
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout

        assert steps < T, 'Steps should be much less than T.'

        # Select which episodes and time steps to use. 
        # np.random.randint doesn't contain the last one, so comes from 0 to roolout_batch_size-1
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)

        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        n_step_ags = np.zeros((batch_size, steps, episode_batch['ag'].shape[-1]))
        n_step_reward_mask = np.ones((batch_size, steps)) * np.array([pow(gamma,i) for i in range(steps)])
        n_step_o2s= np.zeros((batch_size, steps, episode_batch['o'].shape[-1]))
        n_step_us = np.zeros((batch_size, steps, episode_batch['u'].shape[-1]))
        n_step_gamma_matrix = np.ones((batch_size, steps))  # for lambda * Q

        for i in range(steps):
            i_t_samples = t_samples + i
            n_step_reward_mask[:,i][np.where(i_t_samples > T - 1)] = 0
            n_step_gamma_matrix[:,i] = pow(gamma, i+1)
            if i >= 1:  # more than length, use the last one
                n_step_gamma_matrix[:,i][np.where(i_t_samples > T -1)] = n_step_gamma_matrix[:, i-1][np.where(i_t_samples > T-1)]
            i_t_samples[i_t_samples > T-1] = T-1   # last state to compute reward
            n_step_ags[:,i,:] = episode_batch['ag_2'][episode_idxs, i_t_samples]
            n_step_o2s[:,i, :] = episode_batch['o_2'][episode_idxs, i_t_samples]
            n_step_us[:,i,:] = episode_batch['u'][episode_idxs, i_t_samples]

        i_t_samples = t_samples + steps    # last state to observe
        i_t_samples[i_t_samples > T] = T

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            her_indexes = (np.random.uniform(size=batch_size) < future_p)
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]

            # Replace goal with achieved goal but only for the previously-selected
            # HER transitions (as defined by her_indexes). For the other transitions,keep the original goal.
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            # goal noisy can be used to make better fit of Q(s,g,a)
            transitions['g'][her_indexes] = future_ag  #+ 0.005 * np.clip(np.random.randn(*future_ag.shape), -1,1) 
        n_step_gs = transitions['g'].repeat(steps, axis=0)

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        ags = n_step_ags.reshape((batch_size * steps, -1))
        gs = n_step_gs

        reward_params = {'ag_2':ags,'g':gs}
        reward_params['info'] = info
        n_step_reward = reward_fun(**reward_params)

        transitions['r'] = (n_step_reward.reshape((batch_size, steps)) * n_step_reward_mask).sum(axis=1).copy()
        transitions['o_2'] = n_step_o2s[:, -1, :].reshape((batch_size, episode_batch['o'].shape[-1])).copy()
        transitions['gamma'] = n_step_gamma_matrix[:, -1].copy()

        correction = 0
        for i in range(steps - 1):
            obs = n_step_o2s[:, i, :].reshape((batch_size, episode_batch['o'].shape[-1]))
            acts = n_step_us[:, i+1,:].reshape((batch_size, episode_batch['u'].shape[-1]))
            correction += pow(gamma, i+1) * (Q_pi_fun(o=obs, g=transitions['g'].reshape(-1)) - Q_fun(o=obs, g=transitions['g'].reshape(-1),u=acts)).reshape(-1) 
        transitions['r']  += correction

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions

    
    def _sample_nstep_lambda_her_transitions(episode_batch, batch_size_in_transitions, steps, gamma, Q_fun, lamb=0.7):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout

        assert steps < T, 'Steps should be much less than T.'
        assert steps <= 3, 'Steps should be less than 3'  # only support

        # Select which episodes and time steps to use. 
        # np.random.randint doesn't contain the last one, so comes from 0 to roolout_batch_size-1
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)

        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        n_step_ags = np.zeros((batch_size, steps, episode_batch['ag'].shape[-1]))
        n_step_reward_mask = np.ones((batch_size, steps)) * np.array([pow(gamma,i) for i in range(steps)])
        n_step_o2s= np.zeros((batch_size, steps, episode_batch['o'].shape[-1]))
        n_step_gamma_matrix = np.ones((batch_size, steps))  # for lambda * Q

        for i in range(steps):
            i_t_samples = t_samples + i
            n_step_reward_mask[:,i][np.where(i_t_samples > T - 1)] = 0
            n_step_gamma_matrix[:,i] = pow(gamma, i+1)
            if i >= 1:  # more than length, use the last one
                n_step_gamma_matrix[:,i][np.where(i_t_samples > T -1)] = n_step_gamma_matrix[:, i-1][np.where(i_t_samples > T-1)]
            i_t_samples[i_t_samples > T-1] = T-1   # last state to compute reward
            n_step_ags[:,i,:] = episode_batch['ag_2'][episode_idxs, i_t_samples]
            n_step_o2s[:,i, :] = episode_batch['o_2'][episode_idxs, i_t_samples]

        i_t_samples = t_samples + steps    # last state to observe
        i_t_samples[i_t_samples > T] = T

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            her_indexes = (np.random.uniform(size=batch_size) < future_p)
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]

            # Replace goal with achieved goal but only for the previously-selected
            # HER transitions (as defined by her_indexes). For the other transitions,keep the original goal.
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            # goal noisy can be used to make better fit of Q(s,g,a)
            transitions['g'][her_indexes] = future_ag  #+ 0.005 * np.clip(np.random.randn(*future_ag.shape), -1,1) 
        n_step_gs = transitions['g'].repeat(steps, axis=0)

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        ags = n_step_ags.reshape((batch_size * steps, -1))
        gs = n_step_gs

        reward_params = {'ag_2':ags,'g':gs}
        reward_params['info'] = info
        n_step_reward = reward_fun(**reward_params)

        transitions['r'] = (n_step_reward.reshape((batch_size, steps)) * n_step_reward_mask).sum(axis=1).copy()
        transitions['o_2'] = n_step_o2s[:, -1, :].reshape((batch_size, episode_batch['o'].shape[-1])).copy()
        transitions['gamma'] = n_step_gamma_matrix[:, -1].copy()
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        if steps == 1:
            transitions['r'] += transitions['gamma'].reshape(-1) * Q_fun(
                o=transitions['o_2'], 
                g=transitions['g']).reshape(-1)
        elif steps == 2:
            lambda_return_1 = n_step_reward.reshape((batch_size, steps))[:,0] + gamma * Q_fun(
                o=n_step_o2s[:,0,:].reshape((batch_size, episode_batch['o'].shape[-1])), 
                g=transitions['g']).reshape(-1)
            lambda_return_2 = transitions['r'] + transitions['gamma'].reshape(-1) * Q_fun(
                o=transitions['o_2'], 
                g=transitions['g']).reshape(-1)
            lambda_target = (lamb * lambda_return_1 + (lamb ** 2) * lambda_return_2) / (lamb + lamb ** 2)
            transitions['r'] = lambda_target.copy()
        elif steps == 3:
            lambda_return_1 = n_step_reward.reshape((batch_size, steps))[:,0] + gamma * Q_fun(
                o=n_step_o2s[:,0,:].reshape((batch_size, episode_batch['o'].shape[-1])), 
                g=transitions['g']).reshape(-1)
            lambda_return_2 = (n_step_reward.reshape((batch_size, steps))[:,:2] * n_step_reward_mask[:,:2]).sum(axis=1) + n_step_gamma_matrix[:, -2] * Q_fun(
                o=n_step_o2s[:,1,:].reshape((batch_size, episode_batch['o'].shape[-1])), 
                g=transitions['g']).reshape(-1)
            lambda_return_3 = transitions['r'] + transitions['gamma'].reshape(-1) * Q_fun(
                o=transitions['o_2'], 
                g=transitions['g']).reshape(-1)
            lambda_target = (lamb * lambda_return_1 + (lamb ** 2) * lambda_return_2 + (lamb ** 3) * lambda_return_3) / (lamb + lamb ** 2 + lamb ** 3)
            transitions['r'] = lambda_target.copy()
        else:
            raise NotImplementedError

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions

    def _sample_nstep_dynamic_her_transitions(episode_batch, batch_size_in_transitions, steps, gamma, Q_fun, dynamic_model, action_fun, alpha=0.5):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout

        
        # Select which episodes and time steps to use. 
        # np.random.randint doesn't contain the last one, so comes from 0 to roolout_batch_size-1
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # preupdate dynamic model
        loss = dynamic_model.update(transitions['o'], transitions['u'], transitions['o_2'], times=2)
        # print(loss)

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            her_indexes = (np.random.uniform(size=batch_size) < future_p)
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]

            # Replace goal with achieved goal but only for the previously-selected
            # HER transitions (as defined by her_indexes). For the other transitions,
            # keep the original goal.
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        ## model-based on-policy
        reward_list = [transitions['r']]
        last_state = transitions['o_2']
        if steps > 1:
            for _ in range(1, steps):
                state_array = last_state
                action_array = action_fun(o=state_array, g=transitions['g'])
                next_state_array = dynamic_model.predict_next_state(state_array, action_array)
                # test loss
                predicted_obs = dynamic_model.predict_next_state(state_array, transitions['u'])
                loss = np.abs((transitions['o_2'] - predicted_obs)).mean()
                if np.random.random() < 0.1:
                    print(loss)
                    # print(transitions['o_2'][0])
                    # print(predicted_obs[0])

                reward_params = {
                    'g':transitions['g'],
                    'ag_2':obs_to_goal_fun(next_state_array),
                    'info':{}
                }
                next_reward = reward_fun(**reward_params)
                reward_list.append(next_reward.copy())
                last_state = next_state_array

        last_Q = Q_fun(o=last_state, g=transitions['g'])
        target = 0
        for i in range(steps):
            target += pow(gamma, i) * reward_list[i]
        target += pow(gamma, steps) * last_Q.reshape(-1)
        transitions['r'] = target.copy()
        # allievate the model bias
        if steps > 1:
            target_step1 = reward_list[0] + gamma * Q_fun(o=transitions['o_2'], g=transitions['g']).reshape(-1)
            transitions['r'] = (alpha * transitions['r'] + target_step1) / (1 + alpha)
           
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions

    return _sample_her_transitions, _sample_nstep_her_transitions, _sample_nstep_lambda_her_transitions , _sample_nstep_dynamic_her_transitions
#_sample_nstep_lambda_her_transitions
