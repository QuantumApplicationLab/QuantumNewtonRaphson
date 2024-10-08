from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.sparse.linalg import SuperLU
from vqls_prototype.solver import VQLSLog


@dataclass
class NewtonRaphsonResult:
    """Result of the NewtonRaphson routine."""

    solution: np.ndarray
    n_iter: int
    diff: float
    converged: bool
    linear_solver_results: List


@dataclass
class SPLUResult:
    """Results of the Sparse LU solver."""

    solution: np.ndarray
    splu: SuperLU


@dataclass
class VQLSResult:
    """Result of the VQLS."""

    solution: np.ndarray
    residue: float
    logger: VQLSLog
    ref: np.ndarray


@dataclass
class QUBOResult:
    """Result of the QUBO linear solver."""

    solution: np.ndarray
    # n_iter: int
    # error: float
    # cost_function: np.ndarray


@dataclass
class HHLResult:
    """Result of the HHL linear solver."""

    solution: np.ndarray
    residue: float
    ref: np.array
