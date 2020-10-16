import numpy as np
from mher.samplers.sampler import RelabelSampler


class HER_Sampler(RelabelSampler):
    valid_strategy = ['future', 'last', 'random', 'episode', 'cut']
    def __init__(self, T, reward_fun, batch_size, relabel_p, strategy, *args):
        super(HER_Sampler, self).__init__(T, reward_fun, batch_size, relabel_p)
        self.strategy = strategy
        self.cur_L = 1
        self.inc_L = T / 500
    
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
        elif self.strategy == 'cut':
            print(int(self.cur_L))
            future_offset = (np.random.uniform(size=self.batch_size) * np.minimum(int(self.cur_L), (self.T - t_samples))).astype(int)
            future_t = (t_samples + 1 + future_offset)[relabel_indexes]
            future_ag = episode_batch['ag'][episode_idxs[relabel_indexes], future_t]
            self.cur_L  += self.inc_L
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

class ClipHER_Sampler(HER_Sampler):
    def __init__(self, T, reward_fun, batch_size, relabel_p, num_epoch=200, *args):
        super(ClipHER_Sampler, self).__init__(T, reward_fun, batch_size, relabel_p, 'future', *args)
        self.cur_L = 1
        self.inc_L = T / num_epoch
    
    def _get_relabel_ag(self, episode_batch, episode_idxs, t_samples, num_episodes):
        relabel_indexes = self._relabel_idxs()
        future_offset = (np.random.uniform(size=self.batch_size) * np.minimum(int(self.cur_L), (self.T - t_samples))).astype(int)
        future_t = (t_samples + 1 + future_offset)[relabel_indexes]
        future_ag = episode_batch['ag'][episode_idxs[relabel_indexes], future_t]
        return future_ag, relabel_indexes
        
    def sample(self, episode_batch):
        transitions, info = self._sample_transitions(episode_batch)
        relabel_ag, relabel_indexes = self._get_relabel_ag(episode_batch, info['episode_idxs'], info['t_samples'], info['num_episodes'])
        transitions = self.relabel_transition(transitions, relabel_indexes, relabel_ag)
        transitions = self.reshape_transitions(transitions)
        self.cur_L += self.inc_L
        return transitions

