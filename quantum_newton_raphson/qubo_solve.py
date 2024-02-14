import numpy as np
from qalcore.dwave.qubols.encodings import EfficientEncoding
from qalcore.dwave.qubols.qubols import QUBOLS
from scipy.sparse import sparray
from .base_solver import BaseSolver
from .result import QUBOResult
from .utils import preprocess_data


class QUBO_SOLVER(BaseSolver):
    """Solve the LS using a qubo.

    Args:
        BaseSovler (object): base class
    """

    def __init__(self, **options):
        """Init the solver and options."""
        self.options = options

        # preprocess options
        if "encoding" not in self.options:
            self.options["encoding"] = EfficientEncoding

    def __call__(self, A: sparray, b: np.ndarray) -> QUBOResult:
        """Solve a real system of euqations using QUBO linear solver.

        Args:
            A (sparray): input matrix
            b (np.ndarray): right hand side
            options (Dict, optional): Options for the quantum solver. Defaults to {}.

        Returns:
            QUBOResult: solution of the system
        """
        # convert the input data inot a spsparse compatible format
        A, b = preprocess_data(A, b)

        # preprocess the b vector
        # norm_b = np.linalg.norm(b)
        # bnorm = np.copy(b)
        # bnorm /= norm_b

        # solve
        sol = QUBOLS(self.options).solve(A, b)

        # postporcess solution
        # sol *= norm_b

        return QUBOResult(sol)
