import numpy as np
import cv2
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def mean_of_list(func):
    def function_wrapper(*args, **kwargs):
        lists = func(*args, **kwargs)
        return [sum(list) / len(list) for list in lists[:-1]] + [lists[-1]]

    return function_wrapper


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def preprocessing(x):
    img = rgb2gray(x)  # / 255.0 -> Do it later in order to open up more RAM !!!!
    img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    img = img[18:102, :]
    return img


def stack_states(stacked_frames, state, is_new_episode):
    frame = preprocessing(state)

    if is_new_episode:
        stacked_frames = np.stack([frame for _ in range(4)], axis=0)
    else:
        stacked_frames = stacked_frames[1:, ...]
        stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=0)], axis=0)
    return stacked_frames


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def make_mario(env_id):
    main_env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(main_env, SIMPLE_MOVEMENT)
    assert 'SuperMarioBros' in main_env.spec.id
    env = RepeatActionEnv(env)
    return env


class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        super(RepeatActionEnv, self).__init__(env)
        self.env = env
        self.successive_frame = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = self.env.step(action)
            if t == 2:
                self.successive_frame[0] = state
                state = self.successive_frame.max(axis=0)
            elif t == 3:
                self.successive_frame[1] = state
                state = self.successive_frame.max(axis=0)
            reward += r
            if done:
                break

        return state, reward, done, info
