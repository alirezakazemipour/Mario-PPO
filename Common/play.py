import torch
from torch import device
import time
import os
from Common.utils import *


class Play:
    def __init__(self, env, agent, max_episode=1):
        self.env = make_mario(env)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Results"):
            os.mkdir("Results")
        self.VideoWriter = cv2.VideoWriter("Results/" + "ppo" + ".avi", self.fourcc, 20.0,
                                           self.env.observation_space.shape[1::-1])

        self.score = 0

    def evaluate(self):
        stacked_states = np.zeros((4, 84, 84), dtype=np.uint8)
        mean_ep_reward = []
        for ep in range(self.max_episode):
            # self.env.seed(ep)
            s = self.env.reset()
            stacked_states = stack_states(stacked_states, s, True)
            episode_reward = 0
            clipped_ep_reward = 0
            # x = input("Push any button to proceed...")
            done = False
            while not done:
                action, *_ = self.agent.get_actions_and_values(stacked_states, batch=False)
                s_, r, done, info = self.env.step(action[0])
                new_score = info["score"] - self.score
                self.score = info["score"]
                r = r + new_score / 40 + int(info["flag_get"])
                episode_reward += r
                clipped_ep_reward += r / 10
                if info["flag_get"]:
                    print("🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩")

                stacked_states = stack_states(stacked_states, s_, False)
                self.VideoWriter.write(cv2.cvtColor(s_, cv2.COLOR_RGB2BGR))
                self.env.render()
                time.sleep(0.01)
            print(f"episode reward:{episode_reward:.1f}| "
                  f"clipped episode reward:{clipped_ep_reward:.1f}")
            mean_ep_reward.append(episode_reward)
            self.env.close()
            self.VideoWriter.release()
            cv2.destroyAllWindows()
        print(f"Mean episode reward:{sum(mean_ep_reward) / len(mean_ep_reward):0.3f}")
