import numpy as np
from qubols.aequbols import AEQUBOLS
from qubols.encodings import EfficientEncoding
from qubols.encodings import RangedEfficientEncoding
from qubols.qubols import QUBOLS
from scipy.sparse import sparray
from .base_solver import BaseSolver
from .result import QUBOResult
from .utils import preprocess_data


class QUBO_SOLVER(BaseSolver):
    """Solve the linear system using a QUBO formulation."""

    def __init__(self, **options):
        """Init the solver and options."""
        self.options = options

        # preprocess options
        if "encoding" not in self.options:
            self.options["encoding"] = RangedEfficientEncoding

        if "range" not in self.options:
            self.options["range"] = 1.0

        if "offset" not in self.options:
            self.options["offset"] = 0.0

        self.normalise_rhs = False
        if "normalize" in self.options:
            self.normalise_rhs = self.options.pop("normalize")

        self._solver = QUBOLS(self.options)
        if "use_aequbols" in self.options:
            self.use_aequbols = self.options.pop("use_aequbols")
            if self.use_aequbols:
                self._solver = AEQUBOLS(self.options)

    def __call__(self, A: sparray, b: np.ndarray) -> QUBOResult:
        """Solve a real system of euqations using QUBO linear solver.

        Args:
            A (sparray): input matrix
            b (np.ndarray): right hand side

        Returns:
            QUBOResult: solution of the system
        """
        # convert the input data into a spsparse compatible format
        A, b = preprocess_data(A, b)

        if self.normalise_rhs:
            # preprocess the b vector
            norm_b = np.linalg.norm(b)
            original_b = np.copy(b)
            b /= norm_b

        # solve
        sol = self._solver.solve(A, b)

        if self.normalise_rhs:
            # postporcess solution
            b = original_b
            sol *= norm_b

        return QUBOResult(sol)
