import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DQN_Dueling import meteors_wrappers
from DQN_Dueling.model import DuelingDQN

import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

import gc
import os

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

DEFAULT_ENV_NAME = "METEORS"
RUN_NAME = "DQN_Tweaked_Dueling_N1_GCF4_CUSTOM_EDECAY_FULL"
MEAN_REWARD_BOUND = 20

GAMMA = 0.995
BATCH_SIZE = 32

REPLAY_SIZE = 80000
REPLAY_START_SIZE = 20000

GRAD_CALC_FRAMES = 4

LEARNING_RATE = 1e-4
LEARNING_RATE_SYNC_FRAMES = 20000
LEARNING_RATE_DECREASE = LEARNING_RATE / 50

TARGET_SYNC_FRAMES = 20000

# EPSILON_DECAY_LAST_FRAME = 400000
# EPSILON_START = 1.0
# EPSILON_FINAL = 0.01

# Epsilon data: Tuple of EPSILON_DECAY_LAST_FRAME, EPSILON_START, EPSILON_FINAL
EPSILON_DATA = [(200000, 1.0, 0.20), (600000, 0.20, 0.05), (1200000, 0.05, 0.01)]

NOISY_NET = False

# N_STEP = 2


class ExperienceBuffer:
    def __init__(self):
        self.buffer = []
        self.index = 0

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        if len(self.buffer) < REPLAY_SIZE:
            self.buffer.append(experience)

        else:
            self.buffer[self.index % REPLAY_SIZE] = experience
            self.index += 1

    def sample_indices(self, batch_size):
        assert len(self) >= BATCH_SIZE
        batch = random.sample(self.buffer, batch_size)
        return zip(* batch)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_episode(self, net, epsilon, device):
        if not NOISY_NET and np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        self.exp_buffer.append([self.state, action, reward, new_state, is_done])
        self.state = new_state

        done_reward = None
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward


def calc_loss(exp_buffer, net, tgt_net, device="cpu"):
    states, actions, rewards, next_states, dones = exp_buffer.sample_indices(BATCH_SIZE)

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(np.array(rewards, dtype=np.float32)).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * GAMMA + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_epsilon(current_frame):
    start_frame = 0.0
    end_epsilon = 0.0
    for e in EPSILON_DATA:
        end_frame, start_epsilon, end_epsilon = e
        if current_frame <= end_frame:
            m = (start_epsilon - end_epsilon) / (start_frame - end_frame)
            c = start_epsilon - (m * start_frame)
            return m * current_frame + c
        start_frame = end_frame + 1
    return end_epsilon


if __name__ == "__main__":
    device = torch.device("cuda")
    env = meteors_wrappers.make_env()

    net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(log_dir=CUR_DIR + "/log/tensorboard/" + RUN_NAME)
    print(net)

    buffer = ExperienceBuffer()
    agent = Agent(env, buffer)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    frame_idx = 0

    st_time = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = calc_epsilon(frame_idx)
        reward = agent.play_episode(net, epsilon, device=device)

        if reward is not None:
            total_rewards.append(reward)
            speed = frame_idx/(time.time() - st_time)

            m_reward = np.mean(total_rewards[-100:])

            if NOISY_NET:
                print("%d: done %d games, reward %.3f, avg. speed %.2f f/s, learning rate %.8f" % (
                    frame_idx, len(total_rewards), m_reward, speed, LEARNING_RATE))
            else:
                print("%d: done %d games, reward %.3f, eps %.2f, avg. speed %.2f f/s, learning rate %.8f" % (
                    frame_idx, len(total_rewards), m_reward, epsilon, speed, LEARNING_RATE))

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.flush()

            if best_m_reward is None or best_m_reward < m_reward:
                # torch.save(net.state_dict(), CUR_DIR + "/log/saves/best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (best_m_reward, m_reward))
                best_m_reward = m_reward

            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if frame_idx > 2500000:
            print("Frames limit reached!")
            break

        if frame_idx % TARGET_SYNC_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
            gc.collect()

        if frame_idx % LEARNING_RATE_SYNC_FRAMES == 0:
            LEARNING_RATE -= LEARNING_RATE_DECREASE
            LEARNING_RATE = max(LEARNING_RATE, 1e-6)

            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % GRAD_CALC_FRAMES == 0:
            optimizer.zero_grad()
            loss_t = calc_loss(buffer, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()

    writer.close()