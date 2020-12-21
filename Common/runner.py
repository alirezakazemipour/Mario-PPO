from Common.utils import *
import os


class Worker:
    def __init__(self, id, **config):
        self._id = id
        self.config = config
        self._state_shape = self.config["state_shape"]
        self._env = make_mario(self.config["env_name"])
        self._stacked_states = np.zeros(self._state_shape, dtype=np.uint8)
        self._score = 0

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("trajectories"):
            os.mkdir("trajectories")
        self.VideoWriter = cv2.VideoWriter("trajectories/" + "worker_" + f"{self.id}" + ".avi", self.fourcc, 20.0,
                                           self._env.observation_space.shape[1::-1])
        self._frames = []
        self.reset()
        print(f"Worker {self.id}: initiated.")

    @property
    def id(self):
        return self._id

    def render(self):
        self._env.render()

    def reset(self):
        state = self._env.reset()
        self._stacked_states = stack_states(self._stacked_states, state, True)
        self._score = 0
        self._frames = []

    def step(self, conn):
        while True:
            conn.send(self._stacked_states)
            action = conn.recv()
            next_state, r, d, info = self._env.step(action)
            new_score = info["score"] - self._score
            self._score = info["score"]
            r = r + new_score / 40  # r + new_score -> would be scaled later.
            if d:
                if info["flag_get"]:
                    r += 350  # 50
                else:
                    r -= 50

            if info["flag_get"]:
                print("---------------------------------------")
                print("🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩\n"
                      f"Worker {self._id}: got the flag!!!!!!!")
                print("---------------------------------------")

                for frame in self._frames:
                    self.VideoWriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # r = r + new_score + int(info["flag_get"])

            self._stacked_states = stack_states(self._stacked_states, next_state, False)
            self._frames.append(next_state)

            conn.send((self._stacked_states, r / 10, d, info))  # np.sign(r)
            if d:
                self.reset()
