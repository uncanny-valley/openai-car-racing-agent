import sys
sys.path.append("..")

from argparse import ArgumentParser
import gym
import numpy as np
import numpy.typing as npt
import random
from pyvirtualdisplay import Display


display = Display(visible=0, size=(1400, 900))
display.start()

ACTION_SPACE_ENV_0 = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
    (-1, 1,   0), (0, 1,   0), (1, 1,   0),
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
    (-1, 0,   0), (0, 0,   0), (1, 0,   0)
]

def random_action(env_version: int, env: gym.Env):
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
        from gym.envs.box2d import CarRacingV1
        env = CarRacingV1(
            grayscale=1,
            show_info_panel=0,
            discretize_actions='hard',
            frames_per_state=4,
            num_lanes=1,
            num_tracks=1
        )
        dataset = np.empty((num_observations, 96, 96, 4))

    print(dataset.shape)
    i = 0
    while i < num_observations:
        observation = env.reset()

        while True:
            action = random_action(env_version, env)
            observation, reward, done, _ = env.step(action)
            observation = np.expand_dims(observation, axis=0)
            dataset[i] = observation
            i += 1

            if done or i >= num_observations:
                break

    env.close()
    return dataset


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a dataset of environment state frames')
    parser.add_argument('-n', '--num-observations', type=int, default=100, help='Number of observation frames to generate')
    parser.add_argument('-e', '--env', default=0, type=int, help='The environment version (CarRacing-v0 or CarRacing-v1)')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Name of the .npy file to output the dataset')
    args = parser.parse_args()

    dataset = generate_dataset(env_version=args.env, num_observations=args.num_observations)
    print(args.env, dataset.shape)
    np.save(args.output_file, dataset)
