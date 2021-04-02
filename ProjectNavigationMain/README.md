# Udacity Deep Reinforcement Learning Project: Navigation
## Project Overview 
 In this project, I experimented with the Unity ML-Agents Banana Collector environment. I used Deep Q-Learning in my project, along with Dueling 
 Network Architectures [(link)](https://arxiv.org/abs/1511.06581), Double Q-learning [(link)](https://arxiv.org/abs/1509.06461) and Prioritized Experience Replay 
 [(link)](https://arxiv.org/abs/1511.05952).

#### Project Files
- **navigation_sovler.py:**  This file contains the dqn() method which is used to train the RL agent  
- **agent.py:**  This file contains the Agent class, which is responsible for interacting with the environment, 
store experiences in memory and train the Deep Neural Network for state-action value prediction.
- **model.py:** This file contains the Dueling Deep Neural Network architecture used by the agent.
- **experience_replay.py:** This file contains the PrioritizedExperienceBuffer class, which stores experiences with 
priorities. This class can provide sample experiences based on their priorities, to train the agent
- **run.py:** This file can run a trained Agent on the Banana Collector environment 
- **log/tensorboard:** This folder contains the tensorboard graphs of different training runs
- **checkpoints:** This folder contains saved models of different runs
<br/>

Every RL project should have well-defined state, action and reward spaces. For this project the state, action and reward spaces are described below:  
- **State-space:** The Banana Collector environment is a 3D world created using Unity. The environment consists of a moveable player, a fenced field and a lot of bananas. State-space is an array representation
of the environment consisting of 37 floating-point values.  
- **Action-space:** The player, which is represented by the RL agent, can perform 4 actions: 0: move forward, 1: move backward, 2: turn left, 3: turn right.  
- **Reward-space:** The agent gets +1 points for each yellow banana collected. A banana is collected when the agent moves over it.  
- **Agent's goal:** The agent's goal is to maximize the number of yellow bananas collected in each episode. In this project, the environment is considered solved, when the agent is capable of collecting 14 bananas on average for the last 100 episodes.
<br/>

## Getting Started
- The following python libraries are required to run the project: pytorch, numpy, tensorboardx and unityagents
- The Banana Collector environment folder is not included in this github project, but can be found [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip).
<br/>

## Instructions
#### To Train the Agent
To train the agent, all the files and folders mentioned in the **Project Files**, should be saved in a directory. Then the **navigation_sovler.py** file should 
be run using a python 3 interpreter. Two things to note while running the project for training:
- The **navigation_sovler.py** assumes that the Unity ML-Agents Banana Collector environment is in the same directory as itself. The location of the 
Banana Collector environment directory can be updated in line no 20 of the **navigation_sovler.py** file. 
- The RUN_NAME (line 17 of **navigation_sovler.py**) corresponds to a specific run, and creates a tensordboard graph and checkpoint file with the given value.
Different runs should have different RUN_NAME values.
  
#### To Run a Trained Agent
Trained agents (network state dictionaries) are stored in the checkpoints folder, containing the name ***RUN_NAME***.pth. Trained means the agent achieved 
average points of 14 over the last 100 episodes in the Banana Collector environment. The checkpoints folder contains three trained agents: Test1.pth, Test2.pth, Test3.pth.
To run a trained agent, update the RUN_NAME in the **run.py** file (line 13) and run the **run.py** file using a python 3 interpreter.
<br/>

## Results
Please check the [report](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/REPORT.md) file for the implementation and result details.
