# Report on Project Navigation

## Overview of the Algorithm

In this project, I used the Deep Q-Learning Algorithm with Dueling Network Architectures, Double Q-learning and Prioritized Experience Replay, for solving the Unity Banana Collector environment.  

The training of the agent takes place in the dqn() method of the [navigation_sovler.py](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/navigation_sovler.py) file. Here's a very basic overview of the Algorithm for training the agent:
- Initially, the Unity Banana Collector environment is initialized. This environment is responsible for providing the state, reward, next-state and done (if an episode is completed) values.
- Then, an agent object is created which is responsible for selecting an action based on the current state. In Deep Q-Learning, the agent uses a Deep Neural Network for action selection. The DNN predicts the Q values for all actions, given a state and usually, the action with the highest Q value is selected. In this project, the agent codes are written in the Agent class [(agent.py)](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/agent.py) and the DNN codes are written in the DuelingQNetwork class [(model.py)](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/model.py)
- The agent picks an action based on the current state provided by the environment. Based on the action, the environment provides next-state, reward, and done values. This process is repeated for a very long time. 
- To choose better actions, the agent needs to learn by using the values provided by the environment. Instead of learning directly from the environment outputs (called **experience**), the agent stores those experiences in a buffer called the replay buffer and samples experiences from the buffer regularly for the learning purpose. Using a buffer has benefits like unbiased sampling (which would not be possible if the agent used experiences directly) and a single experience can be used multiple times. The agent uses an object of the class PrioritizedExperienceBuffer [(experience_replay.py)](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/experience_replay.py) for storing experiences.
- For learning, the agent picks sample experiences from the replay buffer. Then calculates the target Q values using those samples. To calculate the target Q values, the agent uses the immediate reward, which the sample experiences contain and the next-state values. Amazingly, the next-state values are calculated using a DNN, similar to the DNN which chooses the action. The more the agent trains, the values predicted by the DNNs get better. For that, the training also improves because now the agent is using better predictions for training. 
- After the training reaches a certain level (in this environment, when the mean reward reaches the value 14 for the last 100 episodes), the training is finished.

#### Hyperparameters
**dqn() ([navigation_sovler.py](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/navigation_sovler.py)):** state_size=37, action_size=4, n_episodes=2000, max_t=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995  
**Agent ([agent.py](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/agent.py)):** BUFFER_SIZE=100000, BATCH_SIZE=64, START_TRAIN=512, GAMMA=0.99, TAU=1e-3, LR=5e-4, UPDATE_EVERY=4  
**PrioritizedExperienceBuffer ([experience_replay.py](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/experience_replay.py)):** BETA_LAST=60000, beta=0.4, beta_inc=(1-self.beta)/BETA_LAST

## Improvements
### Dueling Network Architectures
The Dueling Network Architectures [(paper link)](https://arxiv.org/abs/1511.06581), modifies the Deep Neural Network used by the agent. Traditionally used  Deep Neural Networks use multiple dense hidden layers and an input and output layer. The size of the input layer matches the shape of states, and the size of the output layer matches the shape of actions. The dueling network also contains a input layer and multiple dense hidden layers. But instead of a single sequence to the output layer, it splits into two parts. According to the authors, one part is responsible for predicting the state-values (output size: 1), and another is responsible for predicting the advantages of each action (output size: number of actions). Predicting state-values and advantage-values separately improves the overall prediction capability of the network.  
Dueling Network is implemented in the DuelingQNetwork class [(model.py)](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/model.py).
  
**Neural Network Architecture**  
The following network architecture is used in the project for creating local and target Q-Networks:  
  
<img src="https://user-images.githubusercontent.com/8725869/113427519-fc5b1e80-93f6-11eb-849c-318771f911d2.png" width="600" height="220">
 
### Double Q-learning
Double Q-learning [(paper link)](https://arxiv.org/abs/1509.06461), improves how the target value is calculated for the agent to learn. Traditionally the following equation is used to calculate the target value by the Q-Learning algorithm:  
![image](https://user-images.githubusercontent.com/8725869/113436312-4e577080-9406-11eb-8869-201f0515257c.png)  
As the target network is used to select both the action and the action value, this results in overestimation according to the authors of the Double Q-Learning paper. The following equation is used for calculating the target values in Double Q-Learning:  
![image](https://user-images.githubusercontent.com/8725869/113436845-44823d00-9407-11eb-9a9b-d4d9121fb54e.png)  
According to this equation, the action for the next state is chosen by the local network, and the action value is selected by the target network.  
Double Q-Learning is implemented in the Agent class's [(agent.py)](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/agent.py) learn method.

### Prioritized Experience Replay
Instead of randomly sampling experiences from the experience replay buffer, we can sample experiences according to their priorities. Priority of a experience can be set according to it's error: higher the difference of an experience's state-action value with the target value (error), the higher it's priority will be. This is the main idea behind Prioritized Experience Replay [(paper link)](https://arxiv.org/abs/1511.05952). 
Prioritized Experience Replay is implemented in the PrioritizedExperienceBuffer class [(experience_replay.py)](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/experience_replay.py).  
(I tried to implement a version of the Prioritized Experience Replay using SumSegmentTree, but unfortunately I could not get it to work properly. It can be found [here](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/experience_replay_sum_tree.py))  

### Exploration vs Exploitation
While running a trained agent, I noticed that often the agent got stuck after collecting 10 rewards. To solve this problem, I reset epsilon to 0.25 once the mean reward reaches 10.5 (line 100, [navigation_sovler.py](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/navigation_sovler.py)) during training. This made the agent to explore more at later episodes and resulted in an overall better policy.  

## Results
The code in its current state was able to achieve a mean score of 14 over 100 episodes in three different runs. Random was not seeded in the different runs. Here's a plot of the mean reward over 100 episodes vs episode number for the three runs:  
![image](https://user-images.githubusercontent.com/8725869/113443937-b745e500-9414-11eb-8748-23029e065d99.png)  
This plot is created using tensorboard, with log files located at "[/log/tensorboard](https://github.com/fahimfss/RL/tree/master/ProjectNavigationMain/log/tensorboard)".  
There is a performance drop at around reward 10.5, because of the added exploration mechanism during that time. 

Here's a video of a trained agent collecting bananas in the environment:  
[VIDEO LINK](https://user-images.githubusercontent.com/8725869/113444334-80bc9a00-9415-11eb-9f55-61d8de9f4804.mp4)  
This video was created by running the [Test3](https://github.com/fahimfss/RL/tree/master/ProjectNavigationMain/checkpoints) agent, using the [run.py](https://github.com/fahimfss/RL/blob/master/ProjectNavigationMain/run.py) file.  

## Future Works
- To solve the environment by implementing the [RAINBOW](https://arxiv.org/abs/1710.02298) paper.  
- To solve the Banana Pixels environment
- I implemented an RL agent to solve my own game before ([Meteors](https://github.com/fahimfss/RL/tree/master/DQN)). I will improve on that project by applying the knowledge learned from this project. 
