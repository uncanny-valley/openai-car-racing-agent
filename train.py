from argparse import ArgumentParser
import logging
import numpy as np
import gym
from gym.envs.box2d import CarRacing, CarRacingV1
from pyvirtualdisplay import Display

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from agent import CarRacingV0Agent, CarRacingV1Agent
from experiment import Experiment
from preprocessing import SubframeQueue

display = Display(visible=0, size=(1400, 900))
display.start()


# def train_agent_v0(env: CarRacing, render: bool=False, path_to_model: str=None, **kwargs):
#     """
#     Args:
#         env (CarRacing): The CarRacing-v0 OpenAI gym environment.
#         render (bool, optional): Whether to render the display. Defaults to False.
#         path_to_model (str, optional): Path to load an existing model. Defaults to None.
#     """
#     agent = CarRacingV0Agent(env, rng=args.get('rng'), **kwargs)

#     num_episodes = kwargs.get('num_episodes')
#     update_frequency = kwargs.get('update_frequency')
#     save_frequency = kwargs.get('save_frequency')
#     checkpoint_directory = kwargs.get('checkpoint_directory')
#     phi = kwargs.get('phi_length')
#     num_frames_to_skip = kwargs.get('num_frames_to_skip')

#     if path_to_model is not None:
#         agent.load_model(path_to_model)

#     for episode_index in range(num_episodes):
#         current_state = env.reset()

#         # Initialize a sub-frame queue to handle overlapping consecutive frames
#         subframe_queue = SubframeQueue(subframes=[current_state] * phi, size=phi)

#         total_episodic_reward = 0.
#         num_steps = 0

#         while True:
#             if render:
#                 env.render()

#             # Agent takes an action, a, from state and observes reward and next state s'
#             current_state_subframes = subframe_queue.to_numpy()
#             action = agent.act(current_state_subframes)

        
    
#             for _ in range(num_frames_to_skip)


#             next_state, reward, done, _ = env.step(action)

#             total_episodic_reward += reward
            
#             subframe_queue.add(next_state)
#             next_state_subframes = subframe_queue.to_numpy()

#             # Store transition (s, a, r, s', done) in experience replay memory
#             agent.replay_memory.add_transition(current_state_subframes, action, reward, next_state_subframes, done)

#             if done:
#                 print(f'Agent {agent.name}, Epoch={agent.epoch_index}, Episode=(index={episode_index}, total_episodic_reward={total_episodic_reward}, epsilon={agent._epsilon}, episode_steps={num_steps})')
#                 agent.log(values=dict(total_episodic_reward=total_episodic_reward, steps_per_episode=num_steps), step=episode_index)
#                 break

#             agent.maybe_learn()

#             num_steps += 1


#         should_update_target_model = agent.episode_index % update_frequency == 0 and agent.episode_index != 0 if args.get('update_by_episodes') else agent.epoch_index % update_frequency == 0 and agent.epoch_index != 0

#         if should_update_target_model:
#             agent.update_target_weights()

#         if episode_index % save_frequency == 0 and episode_index != 0:
#             agent.save_checkpoint(checkpoint_directory, episode_index=episode_index)



# def train_agent_v1(env: CarRacingV1, render: bool=False, path_to_model: str=None, **kwargs):
#     """
#     Args:
#         env (CarRacingV1): The custom V1 of the CarRacing OpenAI gym environment.
#         render (bool, optional): Whether to render the display. Defaults to False.
#         path_to_model (str, optional): Path to load an existing model. Defaults to None.
#     """
#     agent = CarRacingV1Agent(env, rng=args.get('rng'), **kwargs)

#     num_episodes = kwargs.get('num_episodes')
#     update_frequency = kwargs.get('update_frequency')
#     save_frequency = kwargs.get('save_frequency')
#     checkpoint_directory = kwargs.get('checkpoint_directory')

#     if path_to_model is not None:
#         agent.load_model(path_to_model)

#     for episode_index in range(num_episodes):
#         current_state = normalize_state(env.reset())
#         total_episodic_reward = 0.
#         num_steps = 0

#         while True:
#             if render:
#                 env.render()

#             # Agent takes an action, a, from state and observes reward and next state s'
#             action = agent.act(current_state)
#             next_state, reward, done, _ = env.step(action)

#             total_episodic_reward += reward
#             next_state = normalize_state(next_state)

#             # Store transition (s, a, r, s', done) in experience replay memory
#             agent.replay_memory.add_transition(current_state.copy(), action, reward, next_state.copy(), done)

#             current_state = next_state

#             if done:
#                 print(f'Agent {agent.name}, Epoch={agent.epoch_index}, Episode=(index={episode_index}, total_episodic_reward={total_episodic_reward}, epsilon={agent._epsilon}, episode_steps={num_steps})')
#                 agent.log(values=dict(total_reward=total_episodic_reward, steps_per_episode=num_steps), step=episode_index)
#                 break

#             agent.maybe_learn()

#             num_steps += 1

#         should_update_target_model = agent.episode_index % update_frequency == 0 and agent.episode_index != 0 if args.get('update_by_episodes') else agent.epoch_index % update_frequency == 0 and agent.epoch_index != 0

#         if should_update_target_model:
#             agent.update_target_weights()

#         if episode_index % save_frequency == 0 and episode_index != 0:
#             agent.save_checkpoint(checkpoint_directory, episode_index=episode_index)

        


def main():
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument('--env', type=int, default=1, help='Either CarRacing-v0 or CarRacing-v1 OpenAI gym environment')
    parser.add_argument('--rng', type=int, default=0, help='Random seed to reproduce agent stochasticity')
    parser.add_argument('-m', '--model', type=str, help='Path to load an existing model')
    parser.add_argument('-n', '--num_epochs', type=int, default=300, help='The number of epoch with which to train the agent')
    parser.add_argument('--steps_per_epoch', type=int, default=5000, help='The number of steps per epoch with which to train the agent')
    parser.add_argument('-r', '--render', action='store_true', help='Whether to render the animated display')
    parser.add_argument('-e', '--epsilon', type=np.float32, default=1., help='Initial epsilon for the agent')
    parser.add_argument('-s', '--replay-buffer-size', type=int, default=10000, help='The size of the experience replay memory buffer')
    parser.add_argument('-b', '--minibatch-size', type=int, default=128, help='The size of the minibatch that we will use to intermittently train the agent')
    parser.add_argument('-g', '--discount-factor', type=np.float32, default=0.95, help='How much the agent considers long-term future rewards relative to immediate rewards [0, 1]')
    parser.add_argument('-l', '--learning-rate', type=np.float32, default=1e-3, help='How sensitive the Q-network weights are to estimated errors during training [0, 1]')
    parser.add_argument('-p', '--phi-length', type=int, default=3, help='The number of game frames to stack together, given that the environment doesn\'t provide this automatically')
    parser.add_argument('--num-frames-to-skip', type=np.int64, default=3, help='Number of frames to skip. For example, if set to 3, wes process every 4th frame')
    parser.add_argument('--epsilon-min', type=np.float32, default=0.1, help='A lower bound for the agent\'s decaying epsilon value')
    parser.add_argument('--epsilon-decay', type=np.float32, default=0.9999, help='The proportion by which to scale the current epsilon down [0, 1]')
    parser.add_argument('-u', '--update-frequency', type=np.int64, default=2, help='How often to update the target model\'s weights in epochs')
    parser.add_argument('--save-frequency', type=int, default=25, help='How often to save the target model in epochs')
    parser.add_argument('--test-frequency', type=int, default=25, help='How often to test the agent on a hold-out set of states, in epochs')
    parser.add_argument('--update-by-episodes', action='store_true', help='Whether the specified update frequency is in episodes rather than total frames')
    parser.add_argument('--initial-epoch', type=int, default=0, help='The starting epoch')
    parser.add_argument('--initial-episode', type=int, default=0, help='The starting episode if we are running an existing model')
    args = parser.parse_args()

    # Default hyperparameters
    hyperparameters = {
        'initial_epsilon': args.epsilon,
        'model': args.model,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'rng': args.rng,
        'num_epochs': args.num_epochs,
        'steps_per_epoch': args.steps_per_epoch,
        'replay_buffer_size': args.replay_buffer_size,
        'minibatch_size': args.minibatch_size,
        'discount_factor': args.discount_factor,
        'optimizer': Adam(learning_rate=args.learning_rate, clipnorm=1.0),
        'loss_function': MeanSquaredError(reduction='auto', name='mean_squared_error'),
        'phi_length': args.phi_length,
        'num_frames_to_skip': args.num_frames_to_skip,
        'update_by_episodes': args.update_by_episodes,
        'update_frequency': args.update_frequency,
        'save_frequency': args.save_frequency,
        'checkpoint_directory': './checkpoint',
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

    if args.model is not None:
        agent.load_model(args.model)

    experiment = Experiment(env=env, env_version=args.env, agent=agent, render=args.render, frames_to_skip=args.num_frames_to_skip, phi_length=args.phi_length,
                            num_epochs=args.num_epochs, num_steps_per_epoch=args.steps_per_epoch, target_model_update_frequency=args.update_frequency,
                            initial_epoch=args.initial_epoch, initial_episode=args.initial_episode, model_test_frequency=args.test_frequency,
                            model_save_frequency=args.save_frequency, target_model_update_by_episodes=args.update_by_episodes, checkpoint_directory=hyperparameters['checkpoint_directory'])
    experiment.run()

    env.close()


if __name__ == '__main__':
    main()