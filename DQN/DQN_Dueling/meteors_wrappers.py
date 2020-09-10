from DQN_Dueling.meteor_env import MeteorEnv
import gym.spaces
import numpy as np


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=3):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        obs, reward, done, info = None, None, None, None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (1, old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.expand_dims(observation, axis=0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env():
    env = MeteorEnv()
    env = SkipEnv(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)

# TEST WRAPPER
# env = make_env()
# env.reset()
# for i in range(200):
#     new_state, reward, is_done, _ = env.step(1)
#     i1 = np.array(new_state[0, :, :] * 255, dtype=np.uint8)
#     i2 = np.array(new_state[1, :, :] * 255, dtype=np.uint8)
#     i3 = np.array(new_state[2, :, :] * 255, dtype=np.uint8)
#     i4 = np.array(new_state[3, :, :] * 255, dtype=np.uint8)
#
#     img = np.hstack((i1, i2, i3, i4))
#
#     Image.fromarray(img, mode='L').save("ob.png")
#     if is_done:
#         env.reset()
#     time.sleep(0.3)

