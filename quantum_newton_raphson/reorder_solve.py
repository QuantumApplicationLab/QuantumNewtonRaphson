import numpy as np 

def reorder_solve(A, b, options={}):
    """Solve the linear system by reordering the system of eq.

    Args:
        A (_type_): _description_
        b (_type_): _description_
        options (dict, optional): _description_. Defaults to {}.
    """

    order = get_ordering(A)
    A = A[np.ix_(order,order)]