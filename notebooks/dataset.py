import gym
import numpy as np
import numpy.typing as npt
import random


ACTION_SPACE_ENV_0 = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
    (-1, 1,   0), (0, 1,   0), (1, 1,   0),
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
    (-1, 0,   0), (0, 0,   0), (1, 0,   0)
]

def random_action(env_version: int):
    if env_version == 0:
        return env.action_space.sample()
    else:
        action_index = random.randrange(len(ACTION_SPACE_ENV_0))
        return ACTION_SPACE_ENV_0[action_index]

    

def generate_dataset(env_version: int, num_observations: int) -> npt.NDArray[np.float64]:
    if env_version == 0:
        env = gym.make('CarRacing-v0')
        dataset = np.empty((num_observations, 96, 96, 3))
    else:
        env = CarRacingV1(
            grayscale=1,
            show_info_panel=0,
            discretize_actions='hard',
            frames_per_state=4,
            num_lanes=1,
            num_tracks=1
        )
        dataset = np.empty((num_observations, 96, 96, 4))

    i = 0
    while i < num_observations:
        observation = env.reset()

        while True:
            action = random_action(env_version)
            observation, reward, done, _ = env.step(action)
            print(observation)
            dataset = dataset.append(observation, axis=0)

            if done:
                break
    env.close()

    print(dataset)

if __name__ == '__main__':
    generate_dataset(env_version=0, num_observations=100)