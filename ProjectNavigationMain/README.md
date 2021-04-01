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
- **log/tensorboard:** This folder contains the training graphs of different runs
- **checkpoints:** This folder contains saved models of different runs


