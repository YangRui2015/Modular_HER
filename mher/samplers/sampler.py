import numpy as np


class Sampler:
    def __init__(self, T, reward_fun, batch_size):
        self.T = T
        self.reward_fun = reward_fun
        self.batch_size = batch_size
    
    def _get_transitions(self, episode_batch, episode_idxs, t_samples):
        return {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}
    
    def _sample_transitions(self, episode_batch):
        num_episodes = episode_batch['u'].shape[0]
        episode_idxs = np.random.randint(0, num_episodes, self.batch_size)
        t_samples = np.random.randint(self.T, size=self.batch_size)
        transitions = self._get_transitions(episode_batch, episode_idxs, t_samples)
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

class RelabelSampler(Sampler):
    def __init__(self, T, reward_fun, batch_size, relabel_p):
        '''relabel_p defines the probability for relabeling'''
        super(RelabelSampler, self).__init__(T, reward_fun, batch_size)
        self.relabel_p = relabel_p

    def _relabel_idxs(self):
        return (np.random.uniform(size=self.batch_size) < self.relabel_p)

    def relabel_transition(self, transitions, relabel_indexes, relabel_ag):
        assert relabel_indexes.sum() == len(relabel_ag)
        transitions['g'][relabel_indexes] = relabel_ag
        transitions['r'] = self.recompute_reward(transitions)
        return transitions
