from comet_ml import Experiment
from Common.runner import Worker
from Common.play import Play
from Common.config import get_params
from Common.logger import Logger
from torch.multiprocessing import Process, Pipe
import numpy as np
import torch
from Brain.brain import Brain
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from tqdm import tqdm


def run_workers(worker, conn):
    worker.step(conn)


if __name__ == '__main__':
    np.random.seed(123)
    torch.random.manual_seed(123)

    config = get_params()

    config.update({"env_name": "SuperMarioBros-" + str(config['world']) + "-" + str(config['stage']) + "-v0"})
    test_env = gym_super_mario_bros.make(config["env_name"])
    test_env = JoypadSpace(test_env, SIMPLE_MOVEMENT)
    config.update({"n_actions": test_env.action_space.n})
    test_env.close()

    experiment = Experiment(
        api_key="mpH0nJorSD143jz45qMvMYKZI",
        project_name="general",
        workspace="alirezakazemipour")

    brain = Brain(**config)
    logger = Logger(brain, experiment=experiment, **config)

    if config["do_train"]:
        if not config["train_from_scratch"]:
            checkpoint = logger.load_weights()
            brain.set_from_checkpoint(checkpoint)
            running_reward = checkpoint["running_reward"]
            init_iteration = checkpoint["iteration"]
            episode = checkpoint["episode"]
            position = checkpoint["position"]
            logger.running_reward = running_reward
            logger.episode = episode
            logger.position = position
        else:
            init_iteration = 0
            running__reward = 0
            episode = 0
            position = 0

        workers = [Worker(i, **config) for i in range(config["n_workers"])]

        parents = []
        for worker in workers:
            parent_conn, child_conn = Pipe()
            p = Process(target=run_workers, args=(worker, child_conn,))
            parents.append(parent_conn)
            p.start()

        rollout_base_shape = config["n_workers"], config["rollout_length"]

        init_states = np.zeros(rollout_base_shape + config["state_shape"], dtype=np.uint8)
        init_actions = np.zeros(rollout_base_shape, dtype=np.uint8)
        init_action_probs = np.zeros(rollout_base_shape + (config["n_actions"],))
        init_rewards = np.zeros(rollout_base_shape)
        init_dones = np.zeros(rollout_base_shape, dtype=np.bool)
        init_values = np.zeros(rollout_base_shape)
        init_log_probs = np.zeros(rollout_base_shape)
        init_next_states = np.zeros((rollout_base_shape[0],) + config["state_shape"], dtype=np.uint8)

        logger.on()
        episode_reward = 0
        concatenate = np.concatenate
        for iteration in tqdm(range(init_iteration + 1, config["total_iterations"] + 1)):
            total_states = init_states
            total_actions = init_actions
            total_action_probs = init_action_probs
            total_rewards = init_rewards
            total_dones = init_dones
            total_values = init_values
            total_log_probs = init_log_probs
            next_states = init_next_states

            for t in range(config["rollout_length"]):
                for worker_id, parent in enumerate(parents):
                    s = parent.recv()
                    total_states[worker_id, t] = s

                total_actions[:, t], total_values[:, t], total_log_probs[:, t], total_action_probs[:, t] = \
                    brain.get_actions_and_values(total_states[:, t], batch=True)
                for parent, a in zip(parents, total_actions[:, t]):
                    parent.send(a)

                infos = []
                for worker_id, parent in enumerate(parents):
                    s_, r, d, info = parent.recv()
                    infos.append(info)
                    total_rewards[worker_id, t] = r
                    total_dones[worker_id, t] = d
                    next_states[worker_id] = s_

                episode_reward += total_rewards[0, t]
                if total_dones[0, t]:
                    episode += 1
                    position = infos[0]["x_pos"]
                    logger.log_episode(episode, episode_reward, position)
                    episode_reward = 0

            _, next_values, *_ = brain.get_actions_and_values(next_states, batch=True)

            training_logs = brain.train(states=concatenate(total_states),
                                        actions=concatenate(total_actions),
                                        rewards=total_rewards,
                                        dones=total_dones,
                                        values=total_values,
                                        log_probs=concatenate(total_log_probs),
                                        next_values=next_values)
            brain.schedule_lr()
            brain.schedule_clip_range(iteration)

            logger.log_iteration(iteration,
                                 training_logs,
                                 total_action_probs[0].max(-1).mean())

    else:
        checkpoint = logger.load_weights()
        play = Play(config["env_name"], brain, checkpoint)
        play.evaluate()
