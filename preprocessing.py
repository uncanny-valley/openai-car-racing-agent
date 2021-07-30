
import numpy as np
import numpy.typing as npt


def normalize_state(state: npt.NDArray[np.float64]):
    return state / 255.