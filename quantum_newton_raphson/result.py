from dataclasses import dataclass
import numpy as np 
from typing import List 
from scipy.sparse.linalg  import SuperLU

@dataclass
class NewtonRaphsonResult:
    solution: np.ndarray
    n_iter: int 
    diff: float
    converged: bool
    linear_solver_results: List

@dataclass
class SPLUResult:
    solution: np.ndarray
    splu: SuperLU

@dataclass 
class VQLSResult:
    solution: np.ndarray
    # n_iter: int 
    # error: float 
    # cost_function: np.ndarray

@dataclass 
class QUBOResult:
    solution: np.ndarray
    # n_iter: int 
    # error: float 
    # cost_function: np.ndarray