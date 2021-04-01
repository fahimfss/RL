import sys
import os
import torch
import numpy as np
from collections import deque
from agent import Agent
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# CUR_DIR represents the current directory. Useful for accessing log and checkpoints folder
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

# RUN_NAME represents a specific run. Checkpoint files and Tensorboard logs are saved using the RUN_NAME.
# Helpful for comparing different runs.
RUN_NAME = "Test"

# Initializing the Unity Banana environment
env = UnityEnvironment(file_name=CUR_DIR+"/Banana_Linux/Banana.x86_64", worker_id=1)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Initializing the state_size and action_size for the environment
t_env_info = env.reset(train_mode=True)[brain_name]
t_state = t_env_info.vector_observations[0]
state_size = len(t_state)
action_size = brain.vector_action_space_size
print('State shape: ', state_size)
print('Number of actions: ', action_size)

agent = Agent(state_size=state_size, action_size=action_size)


def dqn(n_episodes=2000, max_t=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    Solves the Banana Environment
    Stores Tensorboard logs in "/log/tensorbaord/(RUN_NAME)" directory
    Stores Agent's qnetwork_local > state_dict values in "/checkpoints/(RUN_NAME).pth" directory

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    writer = SummaryWriter(log_dir=CUR_DIR + "/log/tensorboard/" + RUN_NAME)  # initialize writer object for tensorboard

    # Reset the eps when mean score reaches 10.5
    reset_eps = True

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            # Select action using agent
            action = agent.act(state, eps)

            # Get next_state, reward and done values from env
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            # Agent learn step
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break

        # Write to tensorboard logs
        if len(scores_window) > 0:
            writer.add_scalar("score_mean_100", np.mean(scores_window), i_episode)
        writer.add_scalar("score", score, i_episode)
        writer.flush()

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        # Tf the mean score is 14, the environment is solved.
        if np.mean(scores_window) >= 14:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/' + RUN_NAME + '.pth')
            break

        # Make the agent explore more once it reaches the mean score 10.5
        if np.mean(scores_window) >= 10.5 and reset_eps:
            eps = 0.25
            eps_decay = 0.99
            reset_eps = False

    writer.close()
    env.close()


dqn()
