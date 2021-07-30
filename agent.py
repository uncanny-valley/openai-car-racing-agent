
import datetime
import numpy as np
import numpy.typing as npt
import os

from gym.envs.box2d import CarRacing

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from experience_replay import ExperienceReplay



class Agent:
    def __init__(self, env: CarRacing, **kwargs):
        self._env = env
        self._discount_factor = kwargs.get('discount_factor')
        self._epsilon         = kwargs.get('epsilon')
        self._epsilon_min     = kwargs.get('epsilon_min')
        self._epsilon_decay   = kwargs.get('epsilon_decay')
        self._learning_rate   = kwargs.get('learning_rate')
        self._minibatch_size  = kwargs.get('minibatch_size')

        self._model = self.initialize_model()
        self._model_target = self.initialize_model()

        self.replay_memory   = ExperienceReplay(size=kwargs.get('replay_memory_size'), batch_shape=env.observation_space.shape)
        self.name = datetime.datetime.now().strftime('agent-%Y-%m-%d-%H-%M')

    def save_checkpoint(self, checkpoint_directory: str, episode_index: int):
        path = os.path.join(checkpoint_directory, f'{self.name}-episode-{episode_index}')
        self._model_target.save_weights(path)

    def load_model(self, path_to_model: str):
        self._model.load_weights(path_to_model)
        self._update_target_weights()

    def initialize_model(self):
        model = Sequential([
            Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=self._env.observation_space.shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=12, kernel_size=(4, 4), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(216, activation='relu'),
            Dense(self._env.action_space.n, activation='linear')
        ])
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self._learning_rate, epsilon=1e-7))
        return model

    def act(self, state: npt.NDArray[np.float64]) -> np.uint8:
        return self._env.action_space.sample()

    def learn(self):
        # If enough transitions in experience replay, randomly sample minibatch of transitions
        if len(self.replay_memory) >= self._minibatch_size:
            batch = self.replay_memory.sample_minibatch(batch_size=self._minibatch_size)
            state_samples, action_samples, reward_samples, terminal_samples, next_state_samples = batch

            state_action_values = self._model.predict(state)
            print('state_action_values:', state_action_values.shape)

            future_rewards = self._model_target.predict(next_state_samples)
            print('future_rewards:', future_rewards.shape)

    def update_target_weights(self):
        self._model_target.set_weights(self._model.get_weights())

