import collections
import numpy as np

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class NStepExperienceBuffer:
    def __init__(self, gamma, n_steps, replay_size):
        self.buffer = []
        self.index = 0
        self.gamma = gamma
        self.n_steps = n_steps
        self.replay_size = replay_size
        self.n_gamma = self.gamma ** (self.n_steps - 1)
        self._reset()

    def _reset(self):
        self.n_states = collections.deque(maxlen=self.n_steps)
        self.n_rewards = collections.deque(maxlen=self.n_steps)
        self.n_actions = collections.deque(maxlen=self.n_steps)
        self.reward_sum = 0.0

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        state, action, reward, is_done, new_state = experience

        if self.n_steps == 1:
            exp = Experience(state, action, reward, is_done, new_state)
            self._append(exp)

        else:
            if len(self.n_states) == 0:
                self.n_states.append(state)

            b_state = None
            if len(self.n_states) == self.n_steps:
                b_state = self.n_states.popleft()

            self.n_states.append(new_state)
            self.n_rewards.append(reward)
            self.n_actions.append(action)

            self.reward_sum /= self.gamma
            self.reward_sum += reward * self.n_gamma

            if b_state is not None:
                b_action = self.n_actions.popleft()
                exp = Experience(b_state, b_action, self.reward_sum, is_done, new_state)
                self._append(exp)
                b_reward = self.n_rewards.popleft()
                self.reward_sum -= b_reward

            if is_done:
                reward_sum = 0
                rewards = np.array(self.n_rewards.copy())

                g = 1
                for r in rewards:
                    reward_sum += g * r
                    g *= self.gamma

                while len(self.n_states) > 1:
                    b_state = self.n_states.popleft()
                    b_reward = self.n_rewards.popleft()
                    b_action = self.n_actions.popleft()

                    exp = Experience(b_state, b_action, reward_sum, is_done, new_state)
                    self._append(exp)

                    reward_sum -= b_reward
                    reward_sum /= self.gamma

                self._reset()

    def _append(self, experience):
        if len(self.buffer) < self.replay_size:
            self.buffer.append(experience)

        else:
            self.buffer[self.index % self.replay_size] = experience
            self.index += 1

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return states, actions, rewards, dones, next_states


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

    def _range_sum(self, node: int, start: int, end: int, range_left: int, range_right: int):
        if start > end or start > range_right or end < range_left:
            return 0
        if start >= range_left and end <= range_right:
            return self._tree[node]

        mid = (start + end) // 2
        ch1 = (node * 2) + 1
        ch2 = ch1 + 1

        return self._range_sum(ch1, start, mid, range_left, range_right) + \
               self._range_sum(ch2, mid+1, end, range_left, range_right)

    def range_sum(self, range_left: int, range_right: int):
        assert 0 <= range_left <= self.size
        assert 0 <= range_right <= self.size
        assert range_left <= range_right
        return self._range_sum(0, 0, self.size, range_left, range_right)

    # returns then smallest next_index where the sum of the elements
    # from given start_index to next_index >= sum_value
    # returns -1 if no such index exists
    def next_sum_index(self, start_index: int, sum_val):
        assert 0 <= start_index <= self.size
        l, r = start_index, self.size
        next_ind = -1
        m = int((l + r) / 2)
        while l <= r:
            rng_sum = self.range_sum(start_index, m)
            if rng_sum >= sum_val:
                r = m - 1
                next_ind = m
            else:
                l = m + 1
            m = int((l + r) / 2)

        if next_ind != -1 and self.range_sum(start_index, next_ind) >= sum_val:
            return next_ind
        return -1

    def _prefix_sum(self, node: int, start: int, end: int, sum):
        if start == end:
            if self._tree[node] >= sum:
                return start
            return -1
        ch1 = (node * 2) + 1
        ch2 = ch1 + 1
        mid = (start + end) // 2

        ch1_val = self._tree[ch1]
        if sum <= ch1_val:
            return self._prefix_sum(ch1, start, mid, sum)
        else:
            return self._prefix_sum(ch2, mid + 1, end, sum - ch1_val)

    # returns then smallest index where the sum of the elements
    # from start (index 0) to index is >= sum
    # returns -1 if no such index exists
    def prefix_sum(self, sum):
        return self._prefix_sum(0, 0, self.size, sum)

    def total_sum(self):
        return self._tree[0]


# class MinSegmentTree:
#     def __init__(self, size, default_val):
#         self._tree = np.full(shape=(size*4,), fill_value=default_val, dtype=np.float32)
#         self.size = size
#
#     def _update(self, node: int, start: int, end: int, pos: int, val):
#         if start == end:
#             self._tree[node] = val
#             return
#         mid = (start + end) >> 1
#         ch1 = (node << 1) + 1
#         ch2 = ch1 + 1
#         if pos <= mid:
#             self._update(ch1, start, mid, pos, val)
#         else:
#             self._update(ch2, mid + 1, end, pos, val)
#
#         self._tree[node] = min(self._tree[ch1], self._tree[ch2])
#
#     def update(self, pos: int, val):
#         assert 0 <= pos <= self.size
#         self._update(0, 0, self.size, pos, val)
#
#     def min(self):
#         return self._tree[0]


class PrioritizedNStepReplayBuffer:
    def __init__(self, gamma, n_steps, replay_size, alpha):
        self.buffer = []
        self.index = 0
        self.gamma = gamma
        self.n_steps = n_steps
        self.replay_size = replay_size
        self._priority_values = np.zeros((replay_size,), dtype=np.float32)
        self._alpha = alpha
        self._max_priority = 1.0 ** alpha
        self._n_gamma = gamma ** (n_steps - 1)
        self._sum_priority_tree = SumSegmentTree(replay_size)
        # self._min_priority_tree = MinSegmentTree(replay_size, 1.0e9)
        self._reset()

    def _reset(self):
        self._n_states = collections.deque(maxlen=self.n_steps)
        self._n_rewards = collections.deque(maxlen=self.n_steps)
        self._n_actions = collections.deque(maxlen=self.n_steps)
        self._reward_sum = 0.0

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        state, action, reward, is_done, new_state = experience

        if self.n_steps == 1:
            exp = Experience(state, action, reward, is_done, new_state)
            self._append(exp)
        else:
            if len(self._n_states) == 0:
                self._n_states.append(state)

            b_state = None
            if len(self._n_states) == self.n_steps:
                b_state = self._n_states.popleft()

            self._n_states.append(new_state)
            self._n_rewards.append(reward)
            self._n_actions.append(action)

            self._reward_sum /= self.gamma
            self._reward_sum += reward * self._n_gamma

            if b_state is not None:
                b_action = self._n_actions.popleft()
                exp = Experience(b_state, b_action, self._reward_sum, is_done, new_state)
                self._append(exp)
                b_reward = self._n_rewards.popleft()
                self._reward_sum -= b_reward

            if is_done:
                reward_sum = 0
                rewards = np.array(self._n_rewards.copy())

                g = 1
                for r in rewards:
                    reward_sum += g * r
                    g *= self.gamma

                while len(self._n_states) > 1:
                    b_state = self._n_states.popleft()
                    b_reward = self._n_rewards.popleft()
                    b_action = self._n_actions.popleft()

                    exp = Experience(b_state, b_action, reward_sum, is_done, new_state)
                    self._append(exp)

                    reward_sum -= b_reward
                    reward_sum /= self.gamma

                self._reset()

    def _append(self, experience):
        if len(self) < self.replay_size:
            self._sum_priority_tree.update(len(self), self._max_priority)
            # self._min_priority_tree.update(len(self), self._max_priority)
            self._priority_values[len(self)] = self._max_priority
            self.buffer.append(experience)

        else:
            self.buffer[self.index] = experience
            self._sum_priority_tree.update(self.index, self._max_priority)
            # self._min_priority_tree.update(self.index, self._max_priority)
            self._priority_values[self.index] = self._max_priority
            self.index += 1
            if self.index == self.replay_size:
                self.index = 0

    def sample(self, batch_size: int, beta):
        assert len(self) >= batch_size

        priorities_total_sum = self._sum_priority_tree.total_sum()
        pb = priorities_total_sum / batch_size

        weights = []
        samples = []
        indices = []

        # left_index = 0
        #
        # p_min = self._min_priority_tree.min() / priorities_total_sum
        # max_weight = (p_min * len(self)) ** (-beta)

        max_weight = None

        range_left = 0
        for i in range(batch_size):
            range_right = (i + 1) * pb
            rand_prio = np.random.uniform(range_left, range_right)
            sample_index = self._sum_priority_tree.prefix_sum(rand_prio)
            if sample_index < 0 or sample_index >= len(self):
                sample_index = np.random.randint(0, len(self) - 1)

            samples.append(self.buffer[sample_index])
            indices.append(sample_index)

            sample_priority = self._priority_values[sample_index] / priorities_total_sum
            weight = (sample_priority * len(self)) ** (-beta)
            weights.append(weight)

            if max_weight is not None:
                max_weight = max(weight, max_weight)
            else:
                max_weight = weight

            range_left = range_right

        weights = np.array(weights, dtype=np.float32)
        weights /= max_weight

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            p_alpha = priority ** self._alpha
            self._priority_values[index] = p_alpha
            self._sum_priority_tree.update(index, p_alpha)
            # self._min_priority_tree.update(index, p_alpha)
            self._max_priority = max(self._max_priority, p_alpha)


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append([state, action, reward, next_state, done])
        else:
            self.buffer[self.pos] = [state, action, reward, next_state, done]

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)