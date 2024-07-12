from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from typing import Tuple
from typing import Union
import numpy as np
from scipy.sparse import sparray
from .result import QUBOResult
from .result import SPLUResult
from .result import VQLSResult

ValidInputFormat = Union[sparray, Tuple, np.ndarray]
ValidOutputFormat = Union[SPLUResult, QUBOResult, VQLSResult]


class BaseSolver(ABC):
    """Base class for all linear solvers."""

    @abstractmethod
    def __call__(A: ValidInputFormat, b: ValidInputFormat) -> ValidInputFormat:
        """Solve method.

        Args:
            A (ValidInputFormat): matrix of the linear system
            b (ValidInputFormat): rhs of the LS

        Returns: solution ofmthe linear system

        """
