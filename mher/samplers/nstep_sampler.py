import copy
import numpy as np

from mher.samplers.sampler import RelabelSampler
from mher.samplers.her_sampler import HER_Sampler


class Nstep_Sampler(RelabelSampler):
    def __init__(self, T, reward_fun, batch_size, replay_p, nstep, gamma, *args):
        super(Nstep_Sampler, self).__init__(T, reward_fun, batch_size, replay_p, *args)
        self.nstep = nstep
        self.gamma = gamma
    
    def _sample_nstep_transitions(self, episode_batch):
        transitions, info = self._sample_transitions(episode_batch)
        episode_idxs, t_samples = info['episode_idxs'], info['t_samples']
        transitions['r'] = self.recompute_reward(transitions)
        transition_lis = [transitions]
        nstep_masks = [np.ones(self.batch_size)]
        for i in range(1, self.nstep):
            t_samples_i = t_samples + i
            out_range_idxs = np.where(t_samples_i > self.T-1)
            t_samples_i[out_range_idxs] = self.T - 1
            transitions = self._get_transitions(episode_batch, episode_idxs, t_samples_i)
            transition_lis.append(transitions)
            mask = np.ones(self.batch_size) * pow(self.gamma, i)
            mask[out_range_idxs] = 0
            nstep_masks.append(mask)
        return transition_lis, nstep_masks, info
    
    def _recompute_nstep_reward(self, transition_lis):
        for i in range(len(transition_lis)):
            transition_lis[i]['r'] = self.recompute_reward(transition_lis[i])
        return transition_lis

    # process to get final transitions
    def _get_out_transitions(self, transition_lis, nstep_masks):
        out_transitions = copy.deepcopy(transition_lis[0])
        final_gamma = np.ones(self.batch_size) * pow(self.gamma, self.nstep)  # gamma
        for i in range(1, self.nstep):
            out_transitions['r'] += nstep_masks[i] * transition_lis[i]['r']
            final_gamma[np.where((nstep_masks[i] == 0) & (final_gamma == pow(self.gamma, self.nstep)))] = pow(self.gamma, i)
        out_transitions['o_2'] = transition_lis[-1]['o_2'].copy()
        out_transitions['gamma'] = final_gamma.copy()
        return out_transitions

    def sample(self, episode_batch):
        transition_lis, nstep_masks, _ = self._sample_nstep_transitions(episode_batch)
        transition_lis = self._recompute_nstep_reward(transition_lis)
        out_transitions = self._get_out_transitions(transition_lis, nstep_masks)
        self.reshape_transitions(out_transitions)
        return out_transitions
        
        
class Nstep_HER_Sampler(Nstep_Sampler, HER_Sampler):
    def __init__(self, T, reward_fun, batch_size, relabel_p, nstep, gamma, strategy):
        super().__init__(T, reward_fun, batch_size, relabel_p, nstep, gamma, strategy)

    def relabel_nstep_transitions(self, episode_batch, transition_lis, info):
        relabel_ag, relabel_indexes = self._get_relabel_ag(episode_batch, info['episode_idxs'], info['t_samples'], info['num_episodes'])
        for i in range(len(transition_lis)):
            transitions = transition_lis[i]
            transitions = self.relabel_transition(transitions, relabel_indexes, relabel_ag)
            transition_lis[i] = transitions
        return transition_lis
    
    def sample(self, episode_batch):
        transition_lis, nstep_masks, info = self._sample_nstep_transitions(episode_batch)
        transition_lis = self.relabel_nstep_transitions(episode_batch, transition_lis, info)
        out_transitions = self._get_out_transitions(transition_lis, nstep_masks)
        out_transitions = self.reshape_transitions(out_transitions)
        return out_transitions
