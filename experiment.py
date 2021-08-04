import gym
import logging
import numpy as np
import numpy.typing as npt
from time import time
from typing import Tuple

from agent import Agent
from preprocessing import SubframeQueue, normalize_state
from util import log_memory_usage_summary


class Experiment:
    def __init__(self, env: gym.Env, env_version: str, agent: Agent, render:bool, num_epochs: int, num_steps_per_epoch: int,
                 frames_to_skip: int, phi_length: int, target_model_update_frequency: int, model_save_frequency: int,
                 model_test_frequency: int, target_model_update_by_episodes: int, checkpoint_directory: str, initial_epoch: int=0, initial_episode: int=0):
        self._agent = agent
        self._env   = env
        self._env_version = env_version 

        self._render = render
        self._frames_to_skip = frames_to_skip
        self._phi_length = phi_length
        self._initial_epoch = initial_epoch
        self._num_epochs = num_epochs
        self._num_steps_per_epoch = num_steps_per_epoch

        self._total_episodes = initial_episode

        self._target_model_update_frequency = target_model_update_frequency
        self._model_save_frequency = model_save_frequency
        self._model_test_frequency = model_test_frequency
        self._target_model_update_by_episodes = target_model_update_by_episodes
        self._checkpoint_directory = checkpoint_directory


    def step(self, action: int) -> Tuple[npt.NDArray[np.float64], np.float64, bool]:
        reward = 0
        for _ in range(self._frames_to_skip):
            next_state, r, done, _ = self._env.step(action)
            if done:
                return next_state, reward + r, done
            else:
                reward += r

        return next_state, reward, done


    def run(self):
        for epoch_index in range(self._initial_epoch, self._num_epochs):
            self.run_epoch(epoch_index, self._num_steps_per_epoch)

    def run_epoch(self, epoch_index: int, num_steps: int):
        start_time = time()
        remaining_steps = num_steps
        episode_index = 0
        total_reward = 0.
        mean_episodic_reward = 0.

        logging.info(f'Starting training epoch: {epoch_index}, total_episodes: {self._total_episodes}')

        while remaining_steps > 0:
            if self._env_version == 0:
                total_episodic_reward, num_steps_in_episode = self.run_episode(epoch_index, episode_index, remaining_steps)
            else: 
                total_episodic_reward, num_steps_in_episode = self.run_episode_v1(epoch_index, episode_index, remaining_steps)

            episode_index += 1
            self._total_episodes += 1

            if self._target_model_update_by_episodes and self._total_episodes % self._target_model_update_frequency == 0:
                logging.info(f'Updating target model weights based on episodic update frequency: {self._target_model_update_frequency}')
                self._agent.update_target_weights()

            total_reward += total_episodic_reward
            mean_episodic_reward = mean_episodic_reward + (total_episodic_reward - mean_episodic_reward) / episode_index
            remaining_steps -= num_steps_in_episode

        wall_time = time() - start_time
        self._agent.log_average_loss(epoch_index)
        self._agent.log(values=dict(num_episodes_per_epoch=episode_index + 1, total_reward=total_reward, mean_episodic_reward_in_epoch=mean_episodic_reward, epoch_wall_time=wall_time), step=epoch_index)

        if not self._target_model_update_by_episodes and (epoch_index + 1) % self._target_model_update_frequency == 0:
            logging.info(f'Updating target model weights based on epoch update frequency: {self._target_model_update_frequency}')
            self._agent.update_target_weights()

        if (epoch_index + 1) % self._model_save_frequency == 0:
            logging.info(f'Saving model based on epoch update frequency: {self._model_save_frequency}')
            self._agent.save_checkpoint(self._checkpoint_directory, epoch_index=epoch_index)

        logging.info(f'Finished training epoch: {epoch_index}, total_episodes: {self._total_episodes}, wall_time: {wall_time}')
        log_memory_usage_summary()


    def run_episode(self, epoch_index: int, episode_index: int, max_steps:int) -> Tuple[np.float64, int]:
        start_time = time()
        current_state = self._env.reset()

        # Initialize a sub-frame queue to handle overlapping consecutive frames
        subframe_queue = SubframeQueue(subframes=[current_state] * self._phi_length, size=self._phi_length)

        total_episodic_reward = 0.
        episodic_step_index = 0

        while True:
            if self._render:
                self._env.render()

            # Agent takes an action, a, from state and observes reward and next state s'
            current_state_subframes = subframe_queue.to_numpy()
            action = self._agent.act(current_state_subframes)

            next_state, reward, done = self.step(action)

            total_episodic_reward += reward
            
            subframe_queue.add(next_state)
            next_state_subframes = subframe_queue.to_numpy()

            # Store transition (s, a, r, s', done) in experience replay memory
            self._agent.replay_memory.add_transition(current_state_subframes, self._agent.action_space.index(action), reward, next_state_subframes, done)

            if done or (episodic_step_index + 1) >= max_steps:
                wall_time = time() - start_time
                logging.info(f'Agent={self._agent.name}, Epoch={epoch_index}, Episode=(index={episode_index}, total_episodes={self._total_episodes}, total_episodic_reward={total_episodic_reward}, epsilon={self._agent._epsilon}, episode_steps={episodic_step_index + 1}, wall_time={wall_time}, terminated={done}, steps_remaining_in_epoch={max_steps - (episodic_step_index + 1)})')
                self._agent.log(values=dict(total_episodic_reward=total_episodic_reward, steps_per_episode=episodic_step_index + 1, episode_wall_time=wall_time), step=self._total_episodes)
                break

            episodic_step_index += 1

            self._agent.maybe_learn()

        return total_episodic_reward, episodic_step_index + 1
        

    def run_episode_v1(self, epoch_index: int, episode_index: int, max_steps:int) -> Tuple[np.float64, int]:
        start_time = time()
        current_state = normalize_state(self._env.reset())
        total_episodic_reward = 0.
        episodic_step_index = 0

        while True:
            if self._render:
                self._env.render()

            # Agent takes an action, a, from state and observes reward and next state s'
            action = self._agent.act(current_state)
            next_state, reward, done = self.step(action)

            total_episodic_reward += reward

            next_state = normalize_state(next_state)

            # Store transition (s, a, r, s', done) in experience replay memory
            self._agent.replay_memory.add_transition(current_state.copy(), action, reward, next_state.copy(), done)

            current_state = next_state

            if done or (episodic_step_index + 1) >= max_steps:
                wall_time = time() - start_time
                logging.info(f'Agent={self._agent.name}, Epoch={epoch_index}, Episode=(index={episode_index}, total_episodes={self._total_episodes}, total_episodic_reward={total_episodic_reward}, epsilon={self._agent._epsilon}, episode_steps={episodic_step_index + 1}, wall_time={wall_time}, terminated={done}, steps_remaining_in_epoch={max_steps - (episodic_step_index + 1)})')
                self._agent.log(values=dict(total_episodic_reward=total_episodic_reward, steps_per_episode=episodic_step_index + 1, episode_wall_time=wall_time), step=self._total_episodes)
                break

            episodic_step_index += 1

            self._agent.maybe_learn()

        return total_episodic_reward, episodic_step_index + 1

