import numpy as np
from mher.samplers.sampler import RelabelSampler


class HER_Sampler(RelabelSampler):
    valid_strategy = ['future', 'last', 'random', 'episode']
    def __init__(self, T, reward_fun, batch_size, relabel_p, strategy):
        super(HER_Sampler, self).__init__(T, reward_fun, batch_size, relabel_p)
        self.strategy = strategy
    
    def _get_relabel_ag(self, episode_batch, episode_idxs, t_samples, num_episodes):
        relabel_indexes = self._relabel_idxs()
        if self.strategy == 'future' or self.strategy not in self.valid_strategy:
            future_offset = (np.random.uniform(size=self.batch_size) * (self.T - t_samples)).astype(int)
            future_t = (t_samples + 1 + future_offset)[relabel_indexes]
            future_ag = episode_batch['ag'][episode_idxs[relabel_indexes], future_t]
        elif self.strategy == 'last':
            future_ag = episode_batch['ag'][episode_idxs[relabel_indexes], -1]
        elif self.strategy == 'episode':
            random_t_samples = np.random.randint(self.T, size=self.batch_size)[relabel_indexes]
            future_ag = episode_batch['ag'][episode_idxs[relabel_indexes], random_t_samples]
        else: # self.strategy == 'random'
            random_episode_idxs = np.random.randint(0, num_episodes, self.batch_size)[relabel_indexes]
            random_t_samples = np.random.randint(self.T, size=self.batch_size)[relabel_indexes]
            future_ag = episode_batch['ag'][random_episode_idxs, random_t_samples]
        return future_ag, relabel_indexes

    def sample(self, episode_batch):
        transitions, info = self._sample_transitions(episode_batch)
        relabel_ag, relabel_indexes = self._get_relabel_ag(episode_batch, info['episode_idxs'], info['t_samples'], info['num_episodes'])
        transitions = self.relabel_transition(transitions, relabel_indexes, relabel_ag)
        transitions = self.reshape_transitions(transitions)
        return transitions

