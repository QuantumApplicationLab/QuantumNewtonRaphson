from abc import ABC
from abc import abstractmethod
from typing import Tuple
from typing import Union
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import diags
from scipy.sparse import sparray
from scipy.sparse import spmatrix

ValidInputFormat = Union[np.ndarray, spmatrix, sparray]


class Preconditioner(ABC):
    """Abstract base class for preconditioners applied to linear systems.

    This class provides the interface for preconditioning methods, which
    preprocess the coefficient matrix A and the right-hand side vector b
    to improve the efficiency of linear solvers. Implementations should
    derive from this class and provide concrete methods for applying and
    reversing the preconditioner effect.

    Attributes:
        A (ValidInputFormat): The coefficient matrix of the linear system.
        b (ValidInputFormat): The right-hand side vector of the linear system.
    """

    def __init__(self, A: ValidInputFormat, b: ValidInputFormat):
        """Initializes the preconditioner with a matrix A and vector b.

        Args:
            A (ValidInputFormat): The coefficient matrix of the linear system.
            b (ValidInputFormat): The right-hand side vector of the linear system.
        """
        if not isinstance(A, (spmatrix, sparray)):
            A = csc_matrix(A)

        self.A, self.b = A, b

    @abstractmethod
    def apply(self) -> Tuple[ValidInputFormat, ValidInputFormat]:
        """Applies the preconditioning to the matrix A and vector b.

        This method should modify the attributes A and b of the instance
        to the preconditioned forms.

        Returns:
            ValidInputFormat: The preconditioned matrix A.
            ValidInputFormat: The preconditioned right-hand side vector b.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def reverse(
        self, A_hat: np.ndarray, b_hat: np.ndarray, x_hat: np.ndarray
    ) -> Tuple[ValidInputFormat, ValidInputFormat, ValidInputFormat]:
        """Transforms the preconditioned system back, including it solution, to the original system.

        Args:
            A_hat (np.ndarray): A preconditioned A matrix.
            b_hat (np.ndarray): A preconditioned right-hand side vector.
            x_hat (np.ndarray): The solution vector of the preconditioned system.

        Returns:
            ValidInputFormat: The original A matrix.
            ValidInputFormat: The original right-hand side vector.
            ValidInputFormat: The solution vector of the original system.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class DiagonalScalingPreconditioner(Preconditioner):
    r"""Implements a diagonal scaling preconditioner for linear systems.

    This preconditioner scales the matrix A and the vector b by the square root
    of the diagonal elements of A, effectively transforming the system into one with a better
    condition number and potentially improving the convergence properties of iterative solvers.

    Starting from the original linear system :math:`A x = b`,
    we apply a two-sided preconditioner:

    .. math::
        P^{-1} A P^{-1} (P x) = P^{-1} b,

    where the preconditioner :math:`P` is given by:

    .. math::
        P = \mathrm{diag}(\sqrt{a_{11}}, \sqrt{a_{22}}, \ldots, \sqrt{a_{nn}}).

    The final preconditioned matrices are :math:`\hat{A} = P^{-1} A P^{-1}`
    and :math:`\hat{b} = P^{-1} b`.
    """

    def __init__(self, A: ValidInputFormat, b: ValidInputFormat):
        """Initializes the Diagonal Scaling Preconditioner class with a matrix A and vector b.

        Args:
            A (ValidInputFormat): The coefficient matrix of the linear system.
            b (ValidInputFormat): The right-hand side vector of the linear system.
        """
        super().__init__(A, b)

        # register the preconditioner and its inverse; to be defined later.
        self.P = None
        self.P_inv = None

    def _get_preconditioner(self, A: ValidInputFormat) -> Tuple[spmatrix, spmatrix]:
        """Calculates the diagonal scaling preconditioner from A.

        Args:
            A (spmatrix): The original sparse matrix A.

        Returns:
            Tuple[spmatrix, spmatrix]: The preconditioner matrix P and its inverse P_inv.
        """
        # extract the diagonal elements of A
        diag_A = A.diagonal()
        sqrt_diag_A = np.sqrt(diag_A)

        # construct the sparse diagonal matrix P
        P = diags(sqrt_diag_A, format="csc")

        # small value to avoid division by zero
        epsilon = 1e-10

        # construct the inverse of the sparse diagonal matrix P
        sqrt_diag_A_inv = 1.0 / (sqrt_diag_A + epsilon)
        P_inv = diags(sqrt_diag_A_inv, format="csc")

        return P, P_inv

    def apply(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""Applies the diagonal scaling preconditioning to the matrix A and vector b.

        Returns:
            np.ndarray: The preconditioned :math:`\hat{A}` matrix.
            np.ndarray: The preconditioned right-hand side vector :math:`\hat{b}`.
        """
        # create preconditioner
        self.P, self.P_inv = self._get_preconditioner(self.A)

        # preconditioned matrices
        A_hat = self.P_inv @ self.A @ self.P_inv
        b_hat = self.P_inv @ self.b

        return A_hat.todense(), b_hat

    def reverse(
        self, A_hat: np.ndarray, b_hat: np.ndarray, x_hat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transforms a solution vector of the preconditioned system back to the original system.

        Args:
            A_hat (np.ndarray): A preconditioned A matrix.
            b_hat (np.ndarray): A preconditioned right-hand side vector.
            x_hat (np.ndarray): The solution vector of the preconditioned system.

        Returns:
            np.ndarray: The original A matrix.
            np.ndarray: The original right-hand side vector.
            np.ndarray: The solution vector of the original system.
        """
        # convert vectors to sparse matrices
        A_hat = csc_matrix(A_hat)

        # check if P has already been set
        if self.P is None or self.P_inv is None:
            raise ValueError("Preconditioner P has not been set.")

        # ensure shapes are compatible
        if self.P.todense().shape[1] != A_hat.todense().shape[0]:
            raise ValueError("Shapes of P and A_hat are not compatible.")
        if self.P.todense().shape[1] != b_hat.shape[0]:
            raise ValueError("Shapes of P and b_hat are not compatible.")
        if self.P_inv.todense().shape[1] != x_hat.shape[0]:
            raise ValueError("Shapes of P_inv and x_hat are not compatible.")

        # convert to original A, b, and x
        A = self.P @ A_hat @ self.P
        b = self.P @ b_hat
        x = self.P_inv @ x_hat

        return A.todense(), b, x
