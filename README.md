# Tucker riemopt

Implementation of toolbox for riemannian optimization on manifold of tensors of fixed multilinear (Tucker) rank.
Package supports several computation frameworks (PyTorch and JAX for now), and has convenoent interface for adding new ones.
## Installation
NumPy, SciPy and [opt-einsum](https://pypi.org/project/opt-einsum/) are required for installation. Additionally you need to install your special computation framework (JAX by default).
## Quick start
See `examples` folder to dive into `tucker_riemopt` basics.

* [backend](https://github.com/johanDDC/tucker_riemopt/blob/master/examples/backend.ipynb) notebook contains a guide, how to use different computational frameworks for both routine operations and computations requires autodiff;
* [eigenvalues](https://github.com/johanDDC/tucker_riemopt/blob/master/examples/eigenvalues.ipynb) notebook contains a basic guide for performing riemannian optimization on manifold of tensors of fixed multilinear rank using this package;

## Structure overview

The main classes representing Tucker tensors and Tucker matrices are [`Tucker`](https://github.com/johanDDC/tucker_riemopt/blob/master/tucker_riemopt/tucker.py) and [`TuckerMatrix`](https://github.com/johanDDC/tucker_riemopt/blob/master/tucker_riemopt/matrix.py). 
Also we've implemented [`SparseTucker`](https://github.com/johanDDC/tucker_riemopt/blob/master/tucker_riemopt/tucker.py) class for sparse representation of Tucker tensor. May be useful for tensor completion task or RecSys.

## License
MIT License

