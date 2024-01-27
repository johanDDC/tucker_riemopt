# Tucker Riemopt

Python implementation of the Tucker toolbox. Package allows users to manipulate
tensors in Tucker and SF-Tucker [[1]]() formats. It also provides tools 
for implementing first-order optimization methods of the Riemannian 
optimization on the manifolds of tensors of fixed Tucker rank or fixed SF-Tucker rank.
For instance, package implements a method for efficiently computing the Riemannian
gradient of any smooth function via automatic differentiation.

The library is compatible with several computation frameworks, such as PyTorch and JAX, and can be easily integrated with other frameworks.

## Installation
NumPy, SciPy and [opt-einsum](https://pypi.org/project/opt-einsum/)
are required for installation. Additionally, you need to install your special
computation framework: PyTorch or JAX.

Package may be installed using

`pip install tucker_riemopt[torch/jax]`

with corresponding computation framework.

## Use cases
See [this repository](https://github.com/johanDDC/R-TuckER) for examples of package usage.

Default computation framework is PyTorch. For using JAX you should

1. Install JAX;
2. Enable JAX backend using

```python
from tucker_riemopt import set_backend
set_backend("jax")
```

<!-- ## Quick start
See `examples` folder to dive into `tucker_riemopt` basics.

* [backend](https://github.com/johanDDC/tucker_riemopt/blob/master/examples/backend.ipynb) notebook contains a guide, how to use different computational frameworks for both routine operations and computations requires autodiff;
* [eigenvalues](https://github.com/johanDDC/tucker_riemopt/blob/master/examples/eigenvalues.ipynb) notebook contains a basic guide for performing riemannian optimization on manifold of tensors of fixed multilinear rank using this package; -->

[//]: # (## Structure overview)

[//]: # ()
[//]: # (The main classes representing Tucker tensors and Tucker matrices are [`Tucker`]&#40;https://github.com/johanDDC/tucker_riemopt/blob/master/tucker_riemopt/tucker.py&#41; and [`TuckerMatrix`]&#40;https://github.com/johanDDC/tucker_riemopt/blob/master/tucker_riemopt/matrix.py&#41;. )

[//]: # (Also we've implemented [`SparseTucker`]&#40;https://github.com/johanDDC/tucker_riemopt/blob/master/tucker_riemopt/tucker.py&#41; class for sparse representation of Tucker tensor. May be useful for tensor completion task or RecSys.)

## Documentation

Detailed information may be found [here](https://johanddc.github.io/tucker_riemopt/).

## Contribution policy

We warmly welcome contributions from all developers, provided that they are willing to adhere to the [GitFlow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

## License
MIT License
