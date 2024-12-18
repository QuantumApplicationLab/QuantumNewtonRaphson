from typing import Callable
import numpy as np
from numpy.linalg import norm
from .result import NewtonRaphsonResult
from .splu_solver import SPLU_SOLVER
from .utils import bind_func_to_grad
from .utils import finite_difference_grads


def newton_raphson(
    func: Callable,
    initial_guess: np.ndarray,
    grad: Callable = finite_difference_grads,
    tol: float = 1e-10,
    max_iter: int = 100,
    func_options: dict = {},
    linear_solver: Callable = SPLU_SOLVER(),
    verbose: bool = False,
):
    """Newton Raphson routine.

    Orignal adapted from PyPSA
    Solve f(x) = 0 with initial guess for x and dfdx(x).
    dfdx(x) should return a sparse Jacobian.  Terminate if error on norm
    of f(x) is < x_tol or there were more than lim_iter iterations.

    Args:
        func (Callable): function to minimize
        initial_guess (np.ndarray): initial guess for the solution
        grad (Callable): gradients of the function to minimize
        tol (float, optional): convergence parameters. Defaults to 1e-10.
        max_iter (int, optional): maximum number of iterations. Defaults to 100.
        func_options (dict, optional): options to pass to the callbale functions
        linear_solver (callable, optional): linear solver used to solve the system
        linear_solver_options (dict, optional): options for the linear system
        verbose (bool): print info regarding the solver

    Returns:
        OptimizationResult: Result of the optimization
    """
    # init trackers
    converged = False
    n_iter = 0
    linear_solver_results = []

    # take care of binding if necessary
    grad = bind_func_to_grad(grad, func)

    # init the function values
    current_solution = initial_guess
    func_values = func(current_solution, **func_options)

    # init the error values
    error = norm(func_values, np.Inf)

    # loop while not converged
    while error > tol and n_iter < max_iter:

        if verbose:
            print(f"{n_iter}/{max_iter} error = {error}({tol})")

        n_iter += 1

        # solve linear system
        result = linear_solver(grad(current_solution, **func_options), func_values)
        linear_solver_results.append(result)

        # update solution
        current_solution = current_solution - result.solution

        # update func values
        func_values = func(current_solution, **func_options)

        # compute error
        error = norm(func_values, np.Inf)

    # check for convergence
    if error > tol:
        print(
            f"Warning, we didn't reach the required tolerance within {n_iter} iterations, error is at {error}"
        )
    elif not np.isnan(error):
        converged = True

    return NewtonRaphsonResult(
        current_solution, n_iter, error, converged, linear_solver_results
    )
