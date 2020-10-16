import threading

import numpy as np
from mher.buffers.replay_buffer import ReplayBuffer
from mher.common.segment_tree import MinSegmentTree, SumSegmentTree


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_shapes, size_in_transitions, T, sampler):
        """Create Prioritized Replay buffer"""
        super(PrioritizedReplayBuffer, self).__init__(buffer_shapes, size_in_transitions, T, sampler)

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)"""
        episode_idxs = super().store_episode(episode_batch)
        # save priority
        if not hasattr(episode_idxs, '__len__'):
            episode_idxs = np.array([episode_idxs]) 
        self.sampler.update_new_priorities(episode_idxs)

    def update_priorities(self, idxs, priorities):
        self.sampler.update_priorities(idxs, priorities)

