import numpy as np 
from mher.samplers.sampler import Sampler 
from mher.samplers.her_sampler import HER_Sampler
from mher.common.segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedSampler(Sampler):
    def __init__(self, T, reward_fun, batch_size, size_in_transitions, alpha, beta, eps, *args):
        '''beta: float  To what degree to use importance weights
        (0 - no corrections, 1 - full correction)'''
        super(PrioritizedSampler, self).__init__(T, reward_fun, batch_size, *args)
        assert alpha >= 0 and beta >= 0
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        capacity = 1
        while capacity < size_in_transitions:
            capacity *= 2
        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)
        self.capacity = size_in_transitions
        self._max_priority = 1.0
        self.n_transitions_stored = 0
    
    def update_new_priorities(self, episode_idxs):
        N = len(episode_idxs) * self.T
        priority_array = np.zeros(N) + self._max_priority 
        episode_idxs_repeat = (episode_idxs * self.T).repeat(self.T) + np.arange(self.T)
        self.update_priorities(episode_idxs_repeat, priority_array)
        self.n_transitions_stored += len(episode_idxs) * self.T
        self.n_transitions_stored = min(self.n_transitions_stored, self.capacity)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions"""
        assert len(idxes) == len(priorities) and np.all(priorities >= 0)
        priorities += self.eps  # avoid zero
        new_priority = np.power(priorities.flatten(), self.alpha)
        self.sum_tree.set_items(idxes, new_priority)
        self.min_tree.set_items(idxes, new_priority)
        self._max_priority = max(np.max(priorities), self._max_priority)

    def _sample_idxes(self):
        culm_sums = np.random.random(size=self.batch_size) * self.sum_tree.sum()
        idxes = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            idxes[i] = self.sum_tree.find_prefixsum_idx(culm_sums[i])
        episode_idxs = idxes // self.T
        t_samples = idxes % self.T
        return episode_idxs.astype(np.int), t_samples.astype(np.int), idxes.astype(np.int)

    def priority_sample(self, episode_batch):
        episode_idxs, t_samples, idxes = self._sample_idxes()
        p_min = self.min_tree.min() / self.sum_tree.sum()
        transitions = self._get_transitions(episode_batch, episode_idxs, t_samples)
        p_samples = self.sum_tree.get_items(idxes) / self.sum_tree.sum()
        weights = np.power(p_samples / p_min, - self.beta)  
        transitions['w'] = weights
        info = {
            'episode_idxs': episode_idxs,
            't_samples': t_samples,
            'idxes': idxes,
            'num_episodes': episode_batch['u'].shape[0]
        }
        return transitions, info
        
    def sample(self, episode_batch):
        transitions, info = self.priority_sample(episode_batch)
        transitions['r'] = self.recompute_reward(transitions)
        transitions = self.reshape_transitions(transitions)
        return (transitions, info['idxes'])


class PrioritizedHERSampler(PrioritizedSampler, HER_Sampler):
    '''not good with relabeling after prioritized sampling'''
    def __init__(self, T, reward_fun, batch_size, size_in_transitions, alpha, beta, eps, relabel_p, strategy):
        super().__init__(T, reward_fun, batch_size, size_in_transitions, alpha, beta, eps, relabel_p, strategy)
    
    def sample(self, episode_batch):
        transitions, info = self.priority_sample(episode_batch)
        relabel_ag, relabel_indexes = self._get_relabel_ag(episode_batch, info['episode_idxs'], info['t_samples'], info['num_episodes'])
        transitions = self.relabel_transition(transitions, relabel_indexes, relabel_ag)
        transitions = self.reshape_transitions(transitions)
        return (transitions, info['idxes'])





