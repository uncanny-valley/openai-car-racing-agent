
import datetime
import gym
import numpy as np
import numpy.typing as npt
import os

from gym.envs.box2d import CarRacing, CarRacingV1
from random import randrange

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D 
from tensorflow.keras.metrics import Mean

from experience_replay import ExperienceReplay

from network import DeepQNet


class Agent:
    def __init__(self, env: gym.Env, **kwargs):
        self._env = env
        self._rng = np.random.RandomState(kwargs.get('rng'))

        self._discount_factor = kwargs.get('discount_factor')
        self._epsilon              = kwargs.get('initial_epsilon')
        self._epsilon_min          = kwargs.get('epsilon_min')
        self._epsilon_decay        = kwargs.get('epsilon_decay')
        self._minibatch_size       = kwargs.get('minibatch_size')
        self._network              = self.build_network(**kwargs)
        self.replay_memory         = ExperienceReplay(size=kwargs.get('replay_memory_size'), batch_shape=env.observation_space.shape)
        self.name                  = datetime.datetime.now().strftime('agent-%Y-%m-%d-%H-%M')
        self.log_dir               = os.path.join(kwargs.get('log_directory'), self.name)
        self._train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        # Size 
        self._training_losses_in_epoch = []

    def build_network(self):
        raise NotImplementedError()

    def act(self, state: npt.NDArray[np.float64]) -> np.uint8:
        raise NotImplementedError()   

    def learn(self):
        raise NotImplementedError()

    def maybe_learn(self):
        # If enough transitions in experience replay, randomly sample minibatch of transitions
        if len(self.replay_memory) >= self._minibatch_size:
            self.learn()

    def update_target_weights(self):
        self._network.update_target_weights()

    def save_checkpoint():
        path = os.path.join(checkpoint_directory, f'{self.name}-episode-{episode_index}.h5')
        self._network.save_checkpoint(path)

    def load_model(self, path_to_model: str):
        self._network.load_model(path_to_model)

    def log(self, values: dict, step: int):
        with self._train_summary_writer.as_default():
            for name, value in values.items():
                tf.summary.scalar(name, value, step=step)

    def log_average_loss(self, epoch_index: int):
        average_training_loss = np.mean(self._training_losses_in_epoch)
        self.log(dict(average_training_loss=average_training_loss), step=epoch_index)

    def _decay_epsilon(self):
        self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)


class CarRacingV1Agent(Agent):
    def __init__(self, env: CarRacingV1, **kwargs):
        super().__init__(env, **kwargs)

    def build_network(self, **kwargs):
        return DeepQNet(
            input_shape=self._env.observation_space.shape,
            output_num=self._env.action_space.n,
            discount_factor=kwargs.get('discount_factor'),
            loss_function=kwargs.get('loss_function'),
            optimizer=kwargs.get('optimizer'))

    def act(self, state: npt.NDArray[np.float64]) -> np.uint8:
        if self._rng.rand() <= self._epsilon:
            return self._env.action_space.sample()
        else:
            return self._network.predict_action(state)

    def learn(self):
        batch = self.replay_memory.sample_minibatch(batch_size=self._minibatch_size)
        state_samples, action_samples, reward_samples, terminal_samples, next_state_samples = batch

        loss = self._network.train(state_samples, action_samples, reward_samples, terminal_samples, next_state_samples)
        self._training_losses_in_epoch.append(loss)

        self._decay_epsilon()


class CarRacingV0Agent(Agent):
    def __init__(self, env: CarRacingV1, **kwargs):
        self.action_space = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
            (-1, 1,   0), (0, 1,   0), (1, 1,   0),
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ]

        super().__init__(env, **kwargs)


    def build_network(self, **kwargs):
        return DeepQNet(
            input_shape=(self._env.observation_space.shape[0], self._env.observation_space.shape[1], kwargs.get('phi_length')),
            output_num=len(self.action_space),
            discount_factor=kwargs.get('discount_factor'),
            loss_function=kwargs.get('loss_function'),
            optimizer=kwargs.get('optimizer'))

    def act(self, state: npt.NDArray[np.float64]) -> np.uint8:
        if self._rng.rand() <= self._epsilon:
            action_index = randrange(len(self.action_space))
        else:
            action_index = self._network.predict_action(state)

        return self.action_space[action_index]

    def learn(self):
        batch = self.replay_memory.sample_minibatch(batch_size=self._minibatch_size)
        state_samples, action_samples, reward_samples, terminal_samples, next_state_samples = batch

        loss = self._network.train(state_samples, action_samples, reward_samples, terminal_samples, next_state_samples)
        self._training_losses_in_epoch.append(loss)

        self._decay_epsilon()
