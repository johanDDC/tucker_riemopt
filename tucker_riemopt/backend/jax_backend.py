try:
    import jax
    import jax.numpy as jnp
except ImportError as error:
    message = ("Impossible to import Jax.\n"
               "To use tucker_riemopt with the Jax backend, "
               "you must first install Jax!")
    raise ImportError(message) from error

import numpy
import copy
import typing

from opt_einsum import contract

from .backend import Backend


class JaxBackend(Backend, backend_name="jax"):
    @staticmethod
    def type():
        return type(JaxBackend.tensor([]))

    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None, **kwargs):
        return jnp.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, jnp.ndarray)

    @staticmethod
    def to_numpy(tensor):
        return numpy.asarray(tensor)

    @staticmethod
    def copy(tensor):
        return copy.deepcopy(tensor)

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def dot(a, b):
        return a.dot(b)

    @staticmethod
    def lstsq(a, b):
        x, residuals, _, _ = jnp.linalg.lstsq(a, b, rcond=None, numpy_resid=True)
        return x, residuals

    def khatri_rao(self, matrices, weights=None, mask=None):
        n_columns = matrices[0].shape[1]
        n_factors = len(matrices)

        start = ord("a")
        common_dim = "z"
        target = "".join(chr(start + i) for i in range(n_factors))
        source = ",".join(i + common_dim for i in target)
        operation = source + "->" + target + common_dim

        if weights is not None:
            matrices = [m if i else m*self.reshape(weights, (1, -1)) for i, m in enumerate(matrices)]

        m = mask.reshape((-1, 1)) if mask is not None else 1
        return self.einsum(operation, *matrices).reshape((-1, n_columns))*m

    @staticmethod
    def sort(tensor, axis, descending = False):
        if descending:
            return jnp.flip(jnp.sort(tensor, axis=axis), axis = axis)
        else:
            return jnp.sort(tensor, axis=axis)

    @staticmethod
    def argsort(tensor, axis, descending = False):
        if descending:
            return jnp.argsort(-1 * tensor, axis=axis)
        else:
            return jnp.argsort(tensor, axis=axis)

    @staticmethod
    def grad(func: typing.Callable, argnums: typing.Union[int, typing.Sequence[int]] = 0, retain_graph=False):
        return jax.grad(func, argnums=argnums)

    @staticmethod
    def pad(tensor, pad_width, mode, **kwargs):
        return jnp.pad(tensor, pad_width, mode=mode, **kwargs)

    @staticmethod
    def einsum(subscripts, *operands):
        return contract(subscripts, *operands, backend="jax")

    @staticmethod
    def cho_factor(A, upper=False, **kwargs):
        """
            Additionally wraps result of jax.scipy.linalg.cho_factor in list
        """
        return [jax.scipy.linalg.cho_factor(A, lower=not upper, check_finite=False, **kwargs)]

    @staticmethod
    def cho_solve(B, L, upper=False, **kwargs):
        return jax.scipy.linalg.cho_solve(L, B, check_finite=False, **kwargs)

    @staticmethod
    def lu_factor(A, pivot=True):
        return jax.scipy.linalg.lu_factor(A)

    @staticmethod
    def lu_solve(lu_pivots, B, left=True):
        return jax.scipy.linalg.lu_solve(lu_pivots, B)


for name in ["int64", "int32", "float64", "float32", "complex128", "complex64", "reshape",
             "where", "transpose", "arange", "ones", "zeros", "flip", "trace", "any",
             "zeros_like", "eye", "kron", "concatenate", "max", "min", "matmul",
             "all", "mean", "sum", "cumsum", "count_nonzero", "prod", "sign", "abs", "sqrt", "argmin",
             "argmax", "stack", "conj", "diag", "clip", "log2", "sin", "cos", "squeeze"]:
    JaxBackend.register_method(name, getattr(jnp, name))

for name in ["solve", "qr", "svd", "eigh"]:
    JaxBackend.register_method(name, getattr(jnp.linalg, name))
