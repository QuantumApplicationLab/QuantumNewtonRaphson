[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/QuantumApplicationLab/quantumnewtonraphson) 
[![github license badge](https://img.shields.io/github/license/QuantumApplicationLab/quantumnewtonraphson)](https://github.com/QuantumApplicationLab/quantumnewtonraphson)
[![Python](https://img.shields.io/badge/Python-3.8-informational)](https://www.python.org/)
[![Code style: Black](https://img.shields.io/badge/Code%20style-Black-000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/quantumapplicationlab/qubols/actions/workflows/build.yml/badge.svg)](https://github.com/quantumapplicationlab/qubols/actions/workflows/build.yml)
[![Coverage Status](https://coveralls.io/repos/github/QuantumApplicationLab/qubols/badge.svg?branch=master)](https://coveralls.io/github/QuantumApplicationLab/qubols?branch=master)

<p align="center">
<img width="460" height="300" src=./docs/qnr.png>
</p>

## Quantum Newton Raphson
The `QuantunNewtonRaphson` allows to offload the gradient calculation of the Newton-Raphson algorithm to a variety of quantum linear solvers.  


## Installation

To install quantum_newton_raphson from GitHub repository, do:

```console
git clone git@github.com:QuantumApplicationLab/quantumnewtonraphson.git
cd quantumnewtonraphson
python -m pip install .
```

## Example

```python
from qiskit.primitives import Estimator, Sampler
from quantum_newton_raphson.newton_raphson import newton_raphson
from quantum_newton_raphson.hhl_solver import HHL_SOLVER

def func(input):
  """non linear function to be solved: func(x) = 0."""
  ...
  return output

estimator = Estimator()
sampler = Sampler()

# define the linear solver with the reorder solver
solver = HHL_SOLVER(estimator=estimator, sampler=sampler)

# Solve the newton raphson problem
res = newton_raphson(
    func, np.random.rand(2), linear_solver=solver, verbose=True, max_iter=25
)
```

## Contributing

If you want to contribute to the development of quantum_newton_raphson,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
