from functools import partial
from typing import Callable
from typing import Tuple
from typing import Union
from warnings import warn
import numpy as np
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse import csc_matrix
from scipy.sparse import issparse
from scipy.sparse import sparray
from scipy.sparse._sputils import is_pydata_spmatrix

ValidInputFormat = Union[sparray, Tuple, np.ndarray]


def bind_func_to_grad(grad: Callable, func: Callable) -> Callable:
    """Bind the function argument to the gradient calculator.

    Args:
        grad (Callable): method to compute the gradient
        func (Callable): function we want the gradient of

    Returns:
        Callable: gradient function with the function binded to it
    """
    if grad.__name__ == "finite_difference_grads":
        grad = partial(grad, func=func)
    return grad


def finite_difference_grads(
    input: np.ndarray, func: Callable, eps: float = 1e-6
) -> np.ndarray:
    """Compute the gradient of function at a given poijt using central finite difference.

    Args:
        input (np.ndarray): point at which to evaluate the gradient
        func (Callable): function for which we want the gradients
        eps (float, optional): step size of the fintie difference. Defaults to 1e-6.

    Returns:
        np.ndarray: Jacobian of the matrix
    """
    size = input.shape[0]
    out = np.zeros((size, size))

    for ix in range(size):
        input_temp = np.copy(input)
        input_temp[ix] += eps
        grad_ix = func(input_temp)
        input_temp[ix] -= 2 * eps
        grad_ix -= func(input_temp)

        out[:, ix] = grad_ix / eps / 2
    return out


def preprocess_data(
    A: ValidInputFormat, b: ValidInputFormat
) -> Tuple[ValidInputFormat, ValidInputFormat]:
    """Convert the input data in a type compatible with scipy sparse arrays.

    Args:
        A (ValidInputFormat): input matrix
        b (ValidInputFormat): input vector
    Returns:
        Tuple[ValidInputFormat, ValidInputFormat]: converted input data
    """
    if is_pydata_spmatrix(A):
        A = A.to_scipy_sparse().tocsc()

    if not (issparse(A) and A.format in ("csc", "csr")):
        A = csc_matrix(A)
        warn("spsolve requires A be CSC or CSR matrix format", SparseEfficiencyWarning)

    # b is a vector only if b have shape (n,) or (n, 1)
    b_is_sparse = issparse(b) or is_pydata_spmatrix(b)
    if not b_is_sparse:
        b = np.asarray(b)

    return A, b


def pad_input(A: ValidInputFormat, y: ValidInputFormat) -> Tuple[sparray, sparray, int]:
    """Process the input data to pad to power of two size.

    Args:
        A (sparray): matrix of the linear system
        y (np.ndarray): rhs of the linear system
    """
    input_size = A.shape[0]
    if np.log2(input_size).is_integer():
        return A, y, input_size
    else:
        # new size
        new_size = 2 ** int(np.ceil(np.log2(input_size)))

        # pad matrix
        Afull = np.eye(new_size)
        Afull[:input_size, :input_size] = A.todense()

        # convert pad matrix into a sparse matrix
        Afull = csc_matrix(Afull)

        # pad vector
        yfull = np.zeros(new_size)
        yfull[:input_size] = y
        return Afull, yfull, input_size


def post_process_solution(
    A: ValidInputFormat,
    y: ValidInputFormat,
    x: ValidInputFormat,
    original_input_size: int,
) -> Tuple[ValidInputFormat, ValidInputFormat, ValidInputFormat]:
    """Retreive the  norm and direction of the solution vector.

    VQLS provides a normalized form of the solution vector
    that can also have a -1 prefactor. This routine retrieves
    the un-normalized solution vector with the correct prefactor.

    Args:
        A (np.ndarray): matrix of the linear system
        y (np.ndarray): rhs of the linear system
        x (np.ndarray): proposed solution
        original_input_size (int): size of the original vector
    """
    # unpad the data
    A = A[:original_input_size, :original_input_size]
    x = x[:original_input_size]
    y = y[:original_input_size]

    # compute the prefactor
    Ax = A @ x
    normy = np.linalg.norm(y)
    normAx = np.linalg.norm(Ax)
    prefac = normy / normAx

    # possible flip sign
    if np.dot(Ax * prefac, y) < 0:
        prefac *= -1
    sol = prefac * x
    return A, y, sol
