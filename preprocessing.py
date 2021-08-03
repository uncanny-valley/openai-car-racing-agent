
from collections import deque
import numpy as np
import numpy.typing as npt



def normalize_state(state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return state / 255.

def grayscale(rgb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def process_rgb(state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return normalize_state(grayscale(state))


class SubframeQueue:
    def __init__(self, subframes=[], size=3):
        self._queue = deque([process_rgb(f) for f in subframes], maxlen=size)   

    def add(self, subframe: npt.NDArray[np.float64]):
        processed = process_rgb(subframe)
        self._queue.append(processed)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return np.moveaxis(np.array(self._queue, dtype=np.float64), 0, -1)

