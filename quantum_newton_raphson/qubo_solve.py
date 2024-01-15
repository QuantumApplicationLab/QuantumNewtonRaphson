import numpy  as np 
from typing import List, Dict

from scipy.sparse import csr_matrix

from qalcore.dwave.qubols.qubols import QUBOLS 

from .result import QUBOResult

def qubosolve_complex(Agraph, blist, quantum_solver_options):
    """Solve the linear system using QUBO

    to deal with the complex matrix we solve

    Ar -Ac    xr      br
    Ac  Ar    xc   =  bc

    Args:
        Agraph (tuple): graph representing the A matrix
        blist (List): b vector
        quantum_solver_options (dict): optins for the solver
    """

    # create a dense matrix from the graph
    mat = csr_matrix(Agraph)#.todense()
    size = mat.shape[0]

    # create the real/imag 
    A = np.block([[mat.real, -mat.imag],[mat.imag, mat.real]])

    # preprocess the b vector
    b = np.array(blist)
    b = np.block([b.real, b.imag])
    norm_b = np.linalg.norm(b)
    b /= norm_b

    # solve
    update = QUBOLS(quantum_solver_options).solve(A, b)

    # postporcess solution 
    update *= norm_b

    return QUBOResult(update[:size] + 1.0j*update[size:])

def qubosolve_real(Agraph, blist, quantum_solver_options):
    """Solve the linear system using QUBO

    to deal with the complex matrix we solve


    Args:
        Agraph (tuple): graph representing the A matrix
        blist (List): b vector
        quantum_solver_options (dict): optins for the solver
    """

    # create a dense matrix from the graph
    mat = csr_matrix(Agraph) #.todense()

    # preprocess the b vector
    b = np.array(blist)
    norm_b = np.linalg.norm(b)
    b /= norm_b

    # solve
    update = QUBOLS(quantum_solver_options).solve(mat, b)

    # postporcess solution 
    update *= norm_b

    return QUBOResult(update)