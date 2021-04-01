import collections
import numpy as np
import torch

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# BETA_LAST is used for incrementing the beta variable of the PrioritizedExperienceBuffer
# beta variable will reach the value 1 after this (BETA_LAST) many calls to the sample method
BETA_LAST = 60000


class PrioritizedExperienceBuffer(object):
    """Stores experiences with priorities and provides priority based experience sampling mechanism
    This class is a modified version of the PrioReplayBufferList class found here:
    https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/bench/prio_buffer_bench.py
    """

    def __init__(self, replay_size, batch_size, prob_alpha=0.6):
        """Initialize an PrioritizedExperienceBuffer object.

        Params
        ======
            replay_size (int): max size of the replay buffer
            batch_size (int): sample size
            prob_alpha (float): alpha value for the probability calculation
        """
        self.prob_alpha = prob_alpha
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.buffer = []                 # buffer array to store experiences
        self.pos = 0                     # current position (index) in buffer to store experience
        self.beta = 0.4                  # beta value for the weight calculation
        self.beta_inc = (1 - self.beta) / BETA_LAST                   # beta increment value
        self.priorities = np.zeros((replay_size,), dtype=np.float32)  # priority of each experience
        self.max_priority = 1.0          # max priority of experiences

    def add(self, state, action, reward, next_state, done):
        # Add experience to the replay buffer

        assert state.ndim == next_state.ndim

        # Append experience to the buffer if buffer size less than replay size; otherwise insert in the oldest position
        if len(self.buffer) < self.replay_size:
            self.buffer.append(Experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = Experience(state, action, reward, next_state, done)

        # Assign max priority to the latest inserted experience
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.replay_size

    def sample(self):
        """Returns a batch of experiences, indices of the experiences and the weights of the experiences
        Based on Prioritized Experience Replay paper: https://arxiv.org/abs/1511.05952

        This code follows the weight creation mechanism of prioritized experience
        replay found here (PrioReplayBuffer>sample()):
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/05_dqn_prio_replay.py
        """

        if len(self.buffer) == self.replay_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # calculate probability of each experience
        probabilities = priorities ** self.prob_alpha
        probabilities /= probabilities.sum()

        # Select experience indices based on experience probability
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probabilities)
        # Create the samples array containing the experiences
        samples = [self.buffer[idx] for idx in indices]

        # Create the weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # Increment beta variable
        self.beta = min(self.beta + self.beta_inc, 1)

        # Convert the experiences to separate python tensors
        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(
            device)
        weights = torch.tensor(weights).unsqueeze(1).to(device)

        samples = states, actions, rewards, next_states, dones
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """Updates the priorities of the sampled experiences

        Params
        ======
            batch_indices (array_like): indices of the experiences
            batch_priorities (array_like): new priorities of the experiences
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            self.max_priority = max(self.max_priority, priority)
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

