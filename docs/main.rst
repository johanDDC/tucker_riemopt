Python implementation of the Tucker toolbox. Package allows users to manipulate
tensors in Tucker and SF-Tucker [[1]]() formats. It also provides tools
for implementing first-order optimization methods of the Riemannian
optimization on the manifolds of tensors of fixed Tucker rank or fixed SF-Tucker rank.
For instance, package implements a method for efficiently computing the Riemannian
gradient of any smooth function via automatic differentiation.

The library is compatible with several computation frameworks, such as PyTorch and
JAX, and can be easily integrated with other frameworks.

Installation
============

NumPy, SciPy, PyTorch and `opt-einsum <https://pypi.org/project/opt-einsum/>`_
are required for installation. Additionally, you need to install your special
computation framework (e.g. JAX).

Install `tucker_riemopt` from PyPi:

.. code-block:: python

    TBD

Quick start
=============

See `examples <https://github.com/johanDDC/tucker_riemopt/tree/master/examples>`_ to dive into `tucker_riemopt` basics.

* `backend <https://github.com/johanDDC/tucker_riemopt/blob/master/examples/backend.ipynb>`_ notebook contains a guide, how to use different computational frameworks for both routine operations and computations requires autodiff;
* `eigenvalues <https://github.com/johanDDC/tucker_riemopt/blob/master/examples/eigenvalues.ipynb>`_ notebook contains a basic guide for performing riemannian optimization on manifold of tensors of fixed multilinear rank using this package;