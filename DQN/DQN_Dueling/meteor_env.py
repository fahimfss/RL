import numpy as np
import gym
from MeteorGame.meteor_img import MeteorImg
import MeteorGame.parameters as PRM
from PIL import Image
import time

N_DISCRETE_ACTIONS = 9


class MeteorEnv(gym.Env):
    def __init__(self):
        super(MeteorEnv, self).__init__()
        print('CONSOLE RENDER ENV')
        self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(PRM.OUTPUT_HEIGHT, PRM.OUTPUT_WIDTH), dtype=np.uint8)
        self.meteor_env = MeteorImg()
        self.meteor_env.setup()

    def step(self, action):
        obs, reward, is_done, _ = self.meteor_env.one_step(action)
        return obs, reward / 20, is_done, _

    def reset(self):
        return self.meteor_env.reset()

    def render(self, mode='human'):
        pass

    def close(self):
        self.meteor_env = None


# env = MeteorEnv()
# times = []
# for i in range(200):
#     action = env.action_space.sample()
#     st = time.time()
#     o, r, d, _ = env.step(1)
#     en = time.time()
#     times.append((en-st, o.shape, r, d))
#
#     if d:
#         Image.fromarray(o).save("ob " + str(i) + " .png")
#         env.reset()
#
# tm = 0
# for idx, (t, shape, r, d) in enumerate(times):
#     tm += t
#     print(idx, t, shape, r, d)
#
# print(tm/200)
