
import datetime
import numpy as np
import numpy.typing as npt
import os

from gym.envs.box2d import CarRacing

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D 
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from experience_replay import ExperienceReplay



class Agent:
    def __init__(self, env: CarRacing, **kwargs):
        self._env = env
        self._discount_factor = kwargs.get('discount_factor')
        self._epsilon         = kwargs.get('initial_epsilon')
        self._epsilon_min     = kwargs.get('epsilon_min')
        self._epsilon_decay   = kwargs.get('epsilon_decay')
        self._learning_rate   = kwargs.get('learning_rate')
        self._minibatch_size  = kwargs.get('minibatch_size')

        self._model           = self._initialize_model()
        self._model_target    = self._initialize_model()
        self._optimizer       = Adam(learning_rate=self._learning_rate, clipnorm=1.0)
        self._loss_func       = MeanSquaredError(reduction='auto', name='mean_squared_error')

        self.replay_memory    = ExperienceReplay(size=kwargs.get('replay_memory_size'), batch_shape=env.observation_space.shape)
        self.name             = datetime.datetime.now().strftime('agent-%Y-%m-%d-%H-%M')

    def save_checkpoint(self, checkpoint_directory: str, episode_index: int):
        path = os.path.join(checkpoint_directory, f'{self.name}-episode-{episode_index}')
        self._model_target.save_weights(path)

    def load_model(self, path_to_model: str):
        self._model.load_weights(path_to_model)
        self._update_target_weights()

    def _initialize_model(self):
        visible = Input(shape=self._env.observation_space.shape)
        conv1   = Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu')(visible)
        pool1   = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2   = Conv2D(filters=12, kernel_size=(4, 4), activation='relu')(pool1)
        pool2   = MaxPooling2D(pool_size=(2, 2))(conv2)
        flatten = Flatten()(pool2)
        hidden  = Dense(216, activation='relu')(flatten)
        output  = Dense(self._env.action_space.n, activation='linear')(hidden)
        model   = Model(inputs=visible, outputs=output)
        model.summary()
        return model

    def act(self, state: npt.NDArray[np.float64]) -> np.uint8:
        if np.random.rand() <= self._epsilon:
            return self._env.action_space.sample()
        else:
            state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=np.float64)
            logits = self._model(state_tensor)
            return tf.argmax(logits[0]).numpy()

    def _decay_epsilon(self):
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def maybe_learn(self):
        # If enough transitions in experience replay, randomly sample minibatch of transitions
        if len(self.replay_memory) >= self._minibatch_size:
            self.learn()

    def learn(self):
        batch = self.replay_memory.sample_minibatch(batch_size=self._minibatch_size)
        state_samples, action_samples, reward_samples, terminal_samples, next_state_samples = batch

        # If observation is not terminal, target = reward + discount * expected maximal future reward
        future_reward_samples = self._model_target.predict(next_state_samples)
        targets = reward_samples + self._discount_factor * tf.reduce_max(future_reward_samples, axis=1)

        # If observation is terminal, set the value to its reward, disregarding future rewards
        if any(terminal_samples):
            targets = targets.numpy()
            terminal_mask = terminal_samples.astype(bool)
            targets[terminal_mask] = rewards[terminal_mask]
            targets = tf.convert_to_tensor(targets)

        # Create mask for only the agent-selected actions
        masks = tf.one_hot(action_samples, self._env.action_space.n)

        with tf.GradientTape() as tape:
            q_values = self._model(state_samples, training=True)

            # Only consider Q values for actions that were taken by the agent
            q_values = tf.multiply(q_values, masks)

            # Reduce to array of values of selected actions (minibatch_size, 1)
            q_action = tf.reduce_sum(q_values, axis=1)

            # Compute mean squared error
            loss = self._loss_func(targets, q_action)


        # Backpropagation
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
            
        self._decay_epsilon()

    def update_target_weights(self):
        self._model_target.set_weights(self._model.get_weights())

