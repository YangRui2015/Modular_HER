import numpy as np 
from mher.samplers.sampler import Sampler 
from mher.common.segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedSampler(Sampler):
    def __init__(self, T, reward_fun, batch_size, size_in_transitions, alpha, beta):
        '''beta: float  To what degree to use importance weights
        (0 - no corrections, 1 - full correction)'''
        super(PrioritizedSampler, self).__init__(T, reward_fun, batch_size)
        assert alpha >= 0 and beta >= 0
        self.alpha = alpha
        self.beta = beta

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
        self.update_priorities(episode_idxs, priority_array)
        self.n_transitions_stored += len(episode_idxs)
        self.n_transitions_stored = min(self.n_transitions_stored, self.capacity)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions"""
        assert len(idxes) == len(priorities)
        assert np.all(priorities > 0) and np.all((idxes >= 0) & (idxes < self.n_transitions_stored))
        new_priority = np.power(priorities.flatten(), self.alpha)
        self.sum_tree.set_items(idxes, new_priority)
        self.min_tree.set_items(idxes, new_priority)
        self._max_priority = np.max(priorities)

    def _sample_idxes(self):
        culm_sums = np.random.random(size=self.batch_size) * self.sum_tree.sum(0, self.n_transitions_stored - 1)
        idxes = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            idx = self.sum_tree.find_prefixsum_idx(culm_sums[i])
            assert idx < self.n_transitions_stored
            idxes[i] = idxes
        episode_idxs = idxes // self.T
        t_samples = idxes % self.T
        return episode_idxs, t_samples
        
    def sample(self, episode_batch):
        episode_idxs, t_samples = self._sample_idxes()
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.n_transitions_stored) ** (- self.beta)
        transitions = self._get_transitions(episode_batch, episode_idxs, t_samples)

        idxs = episode_idxs * self.T + t_samples
        p_samples = self.sum_tree.get_items(idxs) / self.sum_tree.sum()
        weights = (p_samples * self.n_transitions_stored) ** (-self.beta) / max_weight
        return (transitions, weights, idxs)



