import numpy as np
import random

from model import DuelingQNetwork
from experience_replay import PrioritizedExperienceBuffer

import torch
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # mini-batch size
START_TRAIN = 512       # Start training steps after replay buffer reaches this size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

# initializing device variable for enabling cuda device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=-1):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed; unused if seed value is negative
        """

        self.state_size = state_size
        self.action_size = action_size

        if seed >= 0:
            random.seed(seed)

        # Initialize the local network and the target network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedExperienceBuffer(BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= START_TRAIN:
                samples, indices, weights = self.memory.sample()
                self.learn(samples, indices, weights)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, samples, indices, weights):
        """Update value parameters using given batch of experience tuples using double q learning.

        This code follows the calculation of next_state_action_values (Double Q Learning) found here (calc_loss method):
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/03_dqn_double.py
        Paper: Deep Reinforcement Learning with Double Q-learning (https://arxiv.org/abs/1509.06461)

        This code follows the loss update methods of prioritized experience replay found here:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/05_dqn_prio_replay.py
        Paper: Prioritized Experience Replay (https://arxiv.org/abs/1511.05952)

        Params
        ======
            samples (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            indices (array_like): indices of the experience samples in the replay buffer
            weights (array_like): weights of the experiences
        """

        states, actions, rewards, next_states, dones = samples

        # zero grad the optimizer
        self.optimizer.zero_grad()

        # get the current state action values using the local q_network
        state_action_values = self.qnetwork_local(states).gather(1, actions)

        # get the target values using the target q_network and reward values
        with torch.no_grad():
            # select the next actions using the local q_network (Double q Learning)
            next_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)

            # select the next action values using the target q_network
            next_state_action_values = self.qnetwork_target(next_states).gather(1, next_actions)

            # calculate target
            target = rewards + ((1 - dones) * next_state_action_values * GAMMA)

        # calculate loss for backpropagation
        losses = weights * (state_action_values - target) ** 2
        losses_v = losses.mean()

        # propagate the loss
        losses_v.backward()

        # update weights
        self.optimizer.step()

        # calculate priorities for replay buffer update
        prios = (losses + 1e-10).data.cpu().numpy()
        self.memory.update_priorities(indices, prios)

        # update the target network
        soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

