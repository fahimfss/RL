import collections
import numpy as np
import torch

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BETA_LAST = 60000

"""
This class was created to make the prioritized experience replay buffer more efficient (using SumSegmentTree)
Unfortunately, this class was not able to perform as good as the PrioritizedExperienceBuffer class in 
experience_replay.py. Possibly because of incorrect implementation.

Also, runtime for both the classes are quite similar. That's why the PrioritizedExperienceBuffer
class was used in the code instead of this class.
"""


class PrioritizedReplay:
    def __init__(self, replay_size, batch_size):
        self.buffer = []
        self.index = 0
        self.replay_size = replay_size
        self._priority_values = np.zeros((replay_size,), dtype=np.float32)
        self._max_priority = 1.0
        self.beta = 0.4
        self.beta_inc = (1 - self.beta) / BETA_LAST
        self._sum_priority_tree = SumSegmentTree(replay_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, is_done):
        exp = Experience(state, action, reward, next_state, is_done)
        self._add(exp)

    def _add(self, experience):
        if len(self) < self.replay_size:
            self._sum_priority_tree.update(len(self), self._max_priority)
            self._priority_values[len(self)] = self._max_priority
            self.buffer.append(experience)

        else:
            self.buffer[self.index] = experience
            self._sum_priority_tree.update(self.index, self._max_priority)
            self._priority_values[self.index] = self._max_priority
            self.index += 1
            if self.index == self.replay_size:
                self.index = 0

    def sample(self):
        assert len(self) >= self.batch_size

        priorities_total_sum = self._sum_priority_tree.total_sum()
        pb = priorities_total_sum / self.batch_size

        self.beta = min(self.beta + self.beta_inc, 1)

        samples = []
        indices = []
        probs = []

        range_left = 0

        for i in range(self.batch_size):
            range_right = (i + 1) * pb
            rand_prio = np.random.uniform(range_left, range_right)
            sample_index = self._sum_priority_tree.prefix_sum(rand_prio)

            samples.append(self.buffer[sample_index])
            indices.append(sample_index)

            probs.append(self._priority_values[sample_index] + 1e-10)
            range_left = range_right

        total = len(self.buffer)
        weights = (total * np.sqrt(np.array(probs, dtype=np.float32))) ** (-self.beta)
        weights /= np.max(weights)

        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(
            device)

        weights = torch.tensor(weights).unsqueeze(1).to(device)
        samples = states, actions, rewards, next_states, dones
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        priorities = np.sqrt(priorities)
        for index, priority in zip(indices, priorities):
            self._priority_values[index] = priority
            self._sum_priority_tree.update(index, priority)
            self._max_priority = max(self._max_priority, priority)


class SumSegmentTree:
    def __init__(self, size):
        self._tree = np.zeros((size * 4,), dtype=np.float32)
        self.size = size

    def _update(self, node: int, start: int, end: int, pos: int, val):
        if start == end:
            self._tree[node] = val
            return
        mid = (start + end) // 2
        ch1 = (node * 2) + 1
        ch2 = ch1 + 1
        if pos <= mid:
            self._update(ch1, start, mid, pos, val)
        else:
            self._update(ch2, mid + 1, end, pos, val)

        self._tree[node] = self._tree[ch1] + self._tree[ch2]

    def update(self, pos: int, val):
        assert 0 <= pos <= self.size
        self._update(0, 0, self.size, pos, val)

    def _prefix_sum(self, node: int, start: int, end: int, sum_val):
        if start == end:
            return start

        ch1 = (node * 2) + 1
        ch2 = ch1 + 1
        mid = (start + end) // 2

        ch1_val = self._tree[ch1]

        if sum_val <= ch1_val:
            return self._prefix_sum(ch1, start, mid, sum_val)
        else:
            return self._prefix_sum(ch2, mid + 1, end, sum_val - ch1_val)

    # returns then smallest index where the sum of the elements
    # from start (index 0) to index is >= sum
    def prefix_sum(self, sum_val):
        return self._prefix_sum(0, 0, self.size, sum_val)

    def total_sum(self):
        return self._tree[0]