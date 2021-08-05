
from argparse import ArgumentParser
import logging
import numpy as np
import gym
from gym.envs.box2d import CarRacing, CarRacingV1
from pyvirtualdisplay import Display

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


from agent import CarRacingV0Agent, CarRacingV1Agent
from experiment import Simulation

display = Display(visible=0, size=(1400, 900))
display.start()


def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument('--env', type=int, default=1, help='Either CarRacing-v0 or CarRacing-v1 OpenAI gym environment')
    parser.add_argument('--rng', type=int, default=0, help='Random seed to reproduce agent stochasticity')
    parser.add_argument('-n', '--num-episodes', default=100, help='Number of episodes to simulate with agent')
    parser.add_argument('-m', '--model', type=str, help='Path to load an existing model', required=True)
    parser.add_argument('-r', '--render', action='store_true', help='Whether to render the animated display')
    parser.add_argument('-e', '--epsilon', type=np.float32, default=0.1, help='Rate of exploration when testing the agent')
    parser.add_argument('-g', '--discount-factor', type=np.float32, default=0.95, help='How much the agent considers long-term future rewards relative to immediate rewards [0, 1]')
    parser.add_argument('-p', '--phi-length', type=int, default=3, help='The number of game frames to stack together, given that the environment doesn\'t provide this automatically')
    parser.add_argument('--num-frames-to-skip', type=np.int64, default=3, help='Number of frames to skip. For example, if set to 3, wes process every 4th frame')
    args = parser.parse_args()

    # Default hyperparameters
    hyperparameters = {
        'initial_epsilon': args.epsilon,
        'model': args.model,
        'rng': args.rng,
        'discount_factor': args.discount_factor,
        'optimizer': Adam(learning_rate=0.001, clipnorm=1.0),
        'loss_function': MeanSquaredError(reduction='auto', name='mean_squared_error'),
        'phi_length': args.phi_length,
        'num_frames_to_skip': args.num_frames_to_skip,
        'log_directory': './log',
    }

    if args.env == 0:
        env = gym.make('CarRacing-v0')
        agent = CarRacingV0Agent(env=env, **hyperparameters)
    else:
        env = CarRacingV1(
            grayscale=1,
            show_info_panel=0,
            discretize_actions='hard',
            frames_per_state=4,
            num_lanes=1,
            num_tracks=1
        )
        agent = CarRacingV1Agent(env=env, **hyperparameters)

    agent.load_model(args.model)
    simul = Simulation(env=env, env_version=args.env, agent=agent, epsilon=args.epsilon, render=args.render, frames_to_skip=args.num_frames_to_skip, phi_length=args.phi_length)
    simul.run(num_episodes=args.num_episodes)

    env.close()


if __name__ == '__main__':
    main()