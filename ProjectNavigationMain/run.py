import sys
import os
import torch
from agent import Agent
from unityagents import UnityEnvironment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# CUR_DIR represents the current directory. Useful for accessing log and checkpoints folder
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

# RUN_NAME represents a specific run. Checkpoint files and Tensorboard logs are saved using the RUN_NAME.
# Helpful for comparing different runs.
RUN_NAME = "Test3"

# Initializing the Unity Banana environment
env = UnityEnvironment(file_name=CUR_DIR+"/Banana_Linux/Banana.x86_64", worker_id=1)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Initializing the state_size and action_size for the environment
env_info = env.reset(train_mode=False)[brain_name]
t_state = env_info.vector_observations[0]
state_size = len(t_state)
action_size = brain.vector_action_space_size
print('State shape: ', state_size)
print('Number of actions: ', action_size)

agent = Agent(state_size=state_size, action_size=action_size)
agent.qnetwork_local.load_state_dict(torch.load('checkpoints/' + RUN_NAME + '.pth'))

state = env_info.vector_observations[0]  # get the current state
score = 0  # initialize the score
while True:
    action = agent.act(state)                # select an action using the trained agent
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]             # get the reward
    done = env_info.local_done[0]            # see if episode has finished
    score += reward                          # update the score
    state = next_state                       # roll over the state to next time step
    if done:                                 # exit loop if episode finished
        break

print("Score: {}".format(score))

