import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model Using Dueling Architecture
    Paper: Dueling Network Architectures for Deep Reinforcement Learning (https://arxiv.org/abs/1511.06581)
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed; unused if seed value is negative
        """
        super(DuelingQNetwork, self).__init__()
        if seed >= 0:
            self.seed = torch.manual_seed(seed)

        # Create the common fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )

        # Fully connected advantage part (output size: action_size)
        self.adv_part = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_size)
        )

        # Fully connected value part (output size: 1)
        self.value_part = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        """Build a network that maps state -> values & advantages"""
        # Get the output from the common fully connected layers
        x = self.fc(state)

        # Pass the output of the common fully connected layers to the advantage and value parts
        val = self.value_part(x)
        adv = self.adv_part(x)

        # Return value according the the dueling paper
        return val + (adv - adv.mean(dim=1, keepdim=True))

