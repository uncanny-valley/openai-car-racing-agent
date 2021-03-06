from collections import deque, namedtuple
import numpy as np
import numpy.typing as npt
import random
from typing import Tuple


Transition = namedtuple('Transition', field_names=[
    'state',
    'action',
    'reward',
    'next_state',
    'is_terminal'])

class ExperienceReplay(object):
    def __init__(self, size: np.int64, batch_shape: Tuple[np.int64]):
        self.max_size = size
        self.batch_shape = batch_shape

        # A deque is inherently circular with a fixed size
        self._memory = deque(maxlen=size)

    def __len__(self):
        return len(self._memory)

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory = value

    def add_transition(self, state: npt.NDArray[np.float64], action: np.uint8, reward: np.float32, next_state: npt.NDArray[np.float64], is_terminal: bool):
        t = Transition(state, action, reward, next_state, is_terminal)
        self._memory.append(t)

    def _reshape_batch_states(self, batch_size: np.int64, batch: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.array(batch).reshape(batch_size, self.batch_shape[0], self.batch_shape[1], self.batch_shape[2])

    def sample_minibatch(self, batch_size: np.int64) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], npt.NDArray[np.float64], npt.NDArray[np.uint8], npt.NDArray[np.float64]]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        states, actions, rewards, next_states, terminals = zip(*[self._memory[idx] for idx in indices])
        return self._reshape_batch_states(batch_size, states), np.array(actions, dtype=np.uint8), np.array(rewards, dtype=np.float64), \
               np.array(terminals, dtype=np.uint8), self._reshape_batch_states(batch_size, next_states)