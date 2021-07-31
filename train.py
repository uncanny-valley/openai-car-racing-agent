from argparse import ArgumentParser
import numpy as np
import gym
from gym.envs.box2d import CarRacing
from pyvirtualdisplay import Display

from agent import Agent
from preprocessing import normalize_state

display = Display(visible=0, size=(1400, 900))
display.start()


def train_agent(env: CarRacing, render: bool=False, path_to_model: str=None, **kwargs):
    agent = Agent(env, **kwargs)

    num_episodes = kwargs.get('num_episodes')
    update_frequency = kwargs.get('update_frequency')
    save_frequency = kwargs.get('save_frequency')
    checkpoint_directory = kwargs.get('checkpoint_directory')

    if path_to_model is not None:
        agent.load_model(path_to_model)

    for episode_index in range(num_episodes):
        current_state = normalize_state(env.reset())
        total_reward = 0.
        num_steps = 0

        while True:
            if render:
                env.render()

            # Agent takes an action, a, from state and observes reward and next state s'
            action = agent.act(current_state)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            next_state = normalize_state(next_state)

            # Store transition (s, a, r, s', done) in experience replay memory
            agent.replay_memory.add_transition(current_state.copy(), action, reward, next_state.copy(), done)

            current_state = next_state

            if done:
                print(f'Agent {agent.name}, Epoch={agent.epoch_index}, Episode=(index={index}, total_reward={total_reward}, epsilon={agent._epsilon}, episode_steps={num_steps})')
                agent.log(values=dict(total_reward=total_reward, steps_per_episode=num_steps), step=episode_index)
                break

            agent.maybe_learn()

            num_steps += 1


        if agent.epoch_index % update_frequency == 0 and agent.epoch_index != 0:
            agent.update_target_weights()

        if episode_index % save_frequency == 0 and episode_index != 0:
            agent.save_checkpoint(checkpoint_directory, episode_index=episode_index)

        


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to load an existing model')
    parser.add_argument('-n', '--num_episodes', type=int, default=1000, help='The number of episodes with which to train the agent')
    parser.add_argument('-r', '--render', action='store_true', help='Whether to render the animated display')
    parser.add_argument('-e', '--epsilon', type=np.float32, default=1., help='Initial epsilon for the agent')
    parser.add_argument('-s', '--replay-buffer-size', type=int, default=10000, help='The size of the experience replay memory buffer')
    parser.add_argument('-b', '--minibatch-size', type=int, default=128, help='The size of the minibatch that we will use to intermittently train the agent')
    parser.add_argument('-g', '--discount-factor', type=np.float32, default=0.95, help='How much the agent considers long-term future rewards relative to immediate rewards [0, 1]')
    parser.add_argument('-l', '--learning-rate', type=np.float32, default=1e-3, help='How sensitive the Q-network weights are to estimated errors during training [0, 1]')
    parser.add_argument('--epsilon-min', type=np.float32, default=0.1, help='A lower bound for the agent\'s decaying epsilon value')
    parser.add_argument('--epsilon-decay', type=np.float32, default=0.9999, help='The proportion by which to scale the current epsilon down [0, 1]')
    parser.add_argument('-u', '--update-frequency', type=np.int64, default=5000, help='How often to update the target model\'s weights in epochs')
    args = parser.parse_args()


    env = CarRacing(
        grayscale=1,
        show_info_panel=0,
        discretize_actions='hard',
        frames_per_state=4,
        num_lanes=1,
        num_tracks=1
    )

    hyperparameters = {
        'initial_epsilon': args.epsilon,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'num_episodes': args.num_episodes,
        'replay_buffer_size': args.replay_buffer_size,
        'minibatch_size': args.minibatch_size,
        'discount_factor': args.discount_factor,
        'learning_rate': args.learning_rate,
        'update_frequency': args.update_frequency,
        'save_frequency': 25, # in number of episodes
        'checkpoint_directory': './checkpoint',
        'log_directory': './log',
    }

    train_agent(env, render=args.render, path_to_model=args.model, **hyperparameters)
    env.close()


if __name__ == '__main__':
    main()