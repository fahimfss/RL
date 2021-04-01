# Udacity Deep Reinforcement Learning Project: Navigation
## Project Overview 
 In this project, I experimented with the Unity ML-Agents Banana Collector environment. I used Deep Q-Learning in my project, along with Dueling 
 Network Architectures [(link)](https://arxiv.org/abs/1511.06581), Double Q-learning [(link)](https://arxiv.org/abs/1509.06461) and Prioritized Experience Replay 
 [(link)](https://arxiv.org/abs/1511.05952).

#### Project Files
- **navigation_sovler.py:**  This file contains the dqn() method which is used to train the RL agent  
- **agent.py:**  This file contains the Agent class, which is responsible for interacting with the environment, 
store interactions in memory and train the deep neural network.
- **model.py:** This file contains the dueling deep neural network architecture used by the agent.
- **experience_replay.py:** This file contains the PrioritizedExperienceBuffer class, which stores experiences with 
priorities. This class can provide sample experiences based on their priorities, to train the agent
- **run.py:** This file can run a trained Agent on the Banana Collector environment 
- **log/tensorboard:** This folder contains the tensorboard graphs of different training runs
- **checkpoints:** This folder contains saved models of different runs
<br/>

## To Run the Project
#### To Train the Agent
To train the agent, all the files and folders mentioned in the **Project Files**, should be saved in a directory. Then the **navigation_sovler.py** file should 
be run using a python 3 interpreter.  
Two things to note while running the project for training:
- The **navigation_sovler.py** assumes that the Unity ML-Agents Banana Collector environment is in the same directory as itself. The location of the 
Banana Collector environment directory can be updated in line no 20 of the **navigation_sovler.py** file. 
The Banana Collector environment is not included in this github project.
- The RUN_NAME (line 17 of **navigation_sovler.py**) corresponds to a specific run, and creates a tensordboard graph and checkpoint file with the given value.
Different runs should have different RUN_NAME values.
