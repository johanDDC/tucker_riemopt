Python implementation of the Tucker toolbox. Package allows users to manipulate
tensors in Tucker and SF-Tucker [[1]]() formats. It also provides tools 
for implementing first-order optimization methods of the Riemannian 
optimization on the manifolds of tensors of fixed Tucker rank or fixed SF-Tucker rank.
For instance, package implements a method for efficiently computing the Riemannian
gradient of any smooth function via automatic differentiation.

The library is compatible with several computation frameworks, such as PyTorch and JAX, 
and can be easily integrated with other frameworks.

Installation
============

NumPy, SciPy and [opt-einsum](https://pypi.org/project/opt-einsum/)
are required for installation. Additionally, you need to install your special
computation framework: PyTorch or JAX.

Package may be installed using

`pip install tucker_riemopt[torch/jax]`

with corresponding computation framework.

.. code-block:: python

    TBD

Use cases
=============

See [this repository](https://github.com/johanDDC/R-TuckER) for examples of package usage.

Default computation framework is PyTorch. For using JAX you should

1. Install JAX;
2. Enable JAX backend using

.. code-block:: python
    from tucker_riemopt import set_backend
    set_backend("jax")
