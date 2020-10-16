import threading

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sampler): 
        """Creates a replay buffer.
        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sampler (class): sampler class used to sample from buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T   # size in episodes
        self.T = T
        self.sampler = sampler
        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape]) for key, shape in buffer_shapes.items()}
        # memory management
        self.point = 0
        self.current_size = 0
        self.n_transitions_stored = 0
        self.lock = threading.Lock()
        
    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]
        # make o_2 and ag_2
        if 'o_2' not in buffers and 'ag_2' not in buffers:
            buffers['o_2'] = buffers['o'][:, 1:, :]
            buffers['ag_2'] = buffers['ag'][:, 1:, :]
        transitions = self.sampler.sample(buffers)
        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(rollout_batch_size x (T or T+1) x dim_key)"""
        buffer_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(buffer_sizes) == buffer_sizes[0])
        buffer_size = buffer_sizes[0]
        with self.lock:
            idxs = self._get_storage_idx(buffer_size)  #use ordered idx get lower performance
            # load inputs into buffers
            for key in episode_batch.keys():
                if key in self.buffers:
                    self.buffers[key][idxs] = episode_batch[key]
            self.n_transitions_stored += buffer_size * self.T
        return idxs

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    # if full, insert randomly
    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow) 
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx
    
    # if full, insert in order
    def _get_ordered_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"

        if self.point+inc <= self.size - 1:
            idx = np.arange(self.point, self.point + inc)
        else:
            overflow = inc - (self.size - self.point)
            idx_a = np.arange(self.point, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])

        self.point = (self.point + inc) % self.size

        # update replay size, don't add when it already surpass self.size
        if self.current_size < self.size:
            self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx
