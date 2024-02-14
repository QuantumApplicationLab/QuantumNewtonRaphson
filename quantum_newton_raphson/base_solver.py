from typing import Tuple
from typing import Union
import numpy as np
from scipy.sparse import sparray

ValidInputFormat = Union[sparray, Tuple, np.ndarray]


class BaseSolver:
    """Base class for all linear solvers."""

    def __call__(A: ValidInputFormat, b: ValidInputFormat):
        """Solve method.

        Args:
            A (ValidInputFormat): matrix of the linear system
            b (ValidInputFormat): rhs of the LS

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("Implement a _solve method")
