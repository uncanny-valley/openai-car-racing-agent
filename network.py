import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D 
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric, Mean as MeanMetric
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.summary import SummaryWriter


from typing import Tuple

class DeepQNet:
    def __init__(self, input_shape: Tuple, output_num: int, discount_factor: np.float64, loss_function: Loss, optimizer: Optimizer):
        self._discount_factor = discount_factor
        self._loss_function   = loss_function
        self._optimizer       = optimizer
        self._num_actions     = output_num

        self._model           = self.initialize_model(input_shape, output_num)
        self._model_target    = self.initialize_model(input_shape, output_num)

    def update_target_weights(self):
        self._model_target.set_weights(self._model.get_weights())  
    
    def load_model(self, path_to_model: str):
        self._model.load_weights(path_to_model)
        self.update_target_weights()
    
    def save_model(self, path: str):
        self._model_target.save_weights(path)

    def initialize_model(self, input_shape: Tuple, output_num: int) -> Model:
        visible = Input(shape=input_shape)
        conv1   = Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu')(visible)
        pool1   = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2   = Conv2D(filters=12, kernel_size=(4, 4), activation='relu')(pool1)
        pool2   = MaxPooling2D(pool_size=(2, 2))(conv2)
        flatten = Flatten()(pool2)
        hidden  = Dense(256, activation='relu')(flatten)
        output  = Dense(output_num, activation='linear')(hidden)
        model   = Model(inputs=visible, outputs=output)
        model.summary()
        return model

    def predict_proba(self, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=np.float64)
        return self._model(state_tensor)

    def predict_action(self, state: npt.NDArray[np.float64]) -> int:
        logits = self.predict_proba(state)
        return tf.argmax(logits[0]).numpy()

    def train(self,
              state_samples: npt.NDArray[np.float64],
              action_samples: npt.NDArray[np.uint8],
              reward_samples: npt.NDArray[np.float64],
              terminal_samples: npt.NDArray[np.uint8],
              next_state_samples: npt.NDArray[np.float64]) -> np.float64:

        # If observation is not terminal, target = reward + discount * expected maximal future reward
        future_reward_samples = self._model_target.predict(next_state_samples)
        targets = reward_samples + self._discount_factor * tf.reduce_max(future_reward_samples, axis=1)

        # If observation is terminal, set the value to its reward, disregarding future rewards
        if any(terminal_samples):
            targets = targets.numpy()
            terminal_mask = terminal_samples.astype(bool)
            targets[terminal_mask] = reward_samples[terminal_mask]
            targets = tf.convert_to_tensor(targets)

        # Create mask for only the agent-selected actions
        masks = tf.one_hot(action_samples, self._num_actions)

        with tf.GradientTape() as tape:
            q_values = self._model(state_samples, training=True)

            # Only consider Q values for actions that were taken by the agent
            q_values = tf.multiply(q_values, masks)

            # Reduce to array of values of selected actions (minibatch_size, 1)
            q_action = tf.reduce_sum(q_values, axis=1)

            # Compute mean squared error
            loss = self._loss_function(targets, q_action)

        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

        return loss