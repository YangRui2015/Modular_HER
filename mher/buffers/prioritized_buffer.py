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
        # save for max priority
        with self.lock:
            self.sampler.update_new_priorities(episode_idxs)

    def update_priority(self, idxs, priorities):
        self.sampler.update_priority(idxs, priorities)

    def sample(self):
        """Returns a dict {key: array(batch_size x shapes[key])}"""
        buffers = {}
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size] 

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions, weights, idxs = self.sampler.sample(buffers)
        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key
        return (transitions, weights, idxs)

