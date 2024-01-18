import numpy as np
from typing import Callable
from functools import partial


def bind_func_to_grad(grad: Callable, func: Callable) -> Callable:
    """Bind the function argument to the gradient calculator

    Args:
        grad (Callable): function to compue the gradient
        func (Callable): function

    Returns:
        Callable: gradient function with the function binded to it
    """

    if grad.__name__ == "finite_difference_grads":
        grad = partial(grad, func=func)
    return grad


def finite_difference_grads(
    input: np.ndarray, func: Callable, eps: float = 1e-6
) -> np.ndarray:
    """Compute the gradient of function at a given poijt using central finite difference

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
