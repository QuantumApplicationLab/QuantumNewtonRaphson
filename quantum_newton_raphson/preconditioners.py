from abc import ABC
from abc import abstractmethod
from typing import Tuple
from typing import Union
import numpy as np
from scipy.sparse import sparray
from scipy.sparse import spmatrix

ValidATypeFormat = Union[np.ndarray, spmatrix, sparray]
ValidBTypeFormat = ValidATypeFormat


class Preconditioner(ABC):
    """Abstract base class for preconditioners applied to linear systems.

    This class provides the interface for preconditioning methods, which
    preprocess the coefficient matrix A and the right-hand side vector b
    to improve the efficiency of linear solvers. Implementations should
    derive from this class and provide concrete methods for applying and
    reversing the preconditioner effect.

    Attributes:
        A (ValidATypeFormat): The coefficient matrix of the linear system.
        b (ValidBTypeFormat): The right-hand side vector of the linear system.
    """

    def __init__(self, A: ValidATypeFormat, b: ValidBTypeFormat):
        """Initializes the preconditioner with a matrix A and vector b.

        Args:
            A (ValidATypeFormat): The coefficient matrix of the linear system.
            b (ValidBTypeFormat): The right-hand side vector of the linear system.
        """
        self.A, self.b = A, b

    @abstractmethod
    def apply(self) -> Tuple[np.ndarray, np.ndarray]:
        """Applies the preconditioning to the matrix A and vector b.

        This method should modify the attributes A and b of the instance
        to the preconditioned forms.

        Returns:
            np.ndarray: The preconditioned matrix A.
            np.ndarray: The preconditioned right-hand side vector b.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def reverse(
        self,
        A_hat: np.ndarray,
        b_hat: np.ndarray,
        x_hat: np.ndarray
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
        raise NotImplementedError("Subclasses should implement this method.")


class DiagonalScalingPreconditioner(Preconditioner):
    """Implements a diagonal scaling preconditioner for linear systems.

    This preconditioner scales the matrix A and the vector b by the square root
    of the diagonal elements of A, effectively transforming the system into one with a better
    condition number and potentially improving the convergence properties of iterative solvers.
    """
    def __init__(self, A: ValidATypeFormat, b: ValidBTypeFormat):
        """Initializes the Diagonal Scaling Preconditioner class with a matrix A and vector b.

        Args:
            A (Union[np.ndarray, spmatrix]): The coefficient matrix of the linear system.
            b (Union[np.ndarray, spmatrix]): The right-hand side vector of the linear system.
        """
        super().__init__(A, b)

    def _get_preconditioner(self, A: ValidATypeFormat) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the diagonal scaling preconditioner from A.

        The preconditioner matrix P is defined as P = np.diag(np.sqrt(np.diag(A))).

        Args:
            A (ValidATypeFormat): The original A matrix.

        Returns:
            np.ndarray: The preconditioner matrix P
            np.ndarray: The inverse of P
        """
        if isinstance(self.A, (spmatrix, sparray)):
            P = np.diag(np.sqrt(np.diag(A.todense())))
        else:
            P = np.diag(np.sqrt(np.diag(A)))

        # calculate inverse of P
        P_inv = np.diag(1.0 / np.diag(P))

        return P, P_inv

    def apply(self) -> Tuple[np.ndarray, np.ndarray]:
        """Applies the diagonal scaling preconditioning to the matrix A and vector b.

        From the linear system :  Ax = b
        We use a two sided preconditioner :  P^{-1} A P^{-1} (Px) = P^{-1} b.

        with P = np.diag(np.sqrt(np.diag(A)))

        Returns:
            np.ndarray: The preconditioned matrix A_hat.
            np.ndarray: The preconditioned right-hand side vector b_hat.
        """
        # create preconditioner
        self.P, self.P_inv = self._get_preconditioner(self.A)

        return self.P_inv @ self.A @ self.P_inv, self.P_inv @ self.b

    def reverse(
        self, A_hat: np.ndarray,
        b_hat: np.ndarray,
        x_hat: np.ndarray
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
        return self.P @ A_hat @ self.P, self.P @ b_hat, self.P_inv @ x_hat
