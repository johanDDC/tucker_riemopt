from unittest import TestCase

import jax
import jax.config
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from src.tucker import Tucker
from src.matrix import TuckerMatrix
from src.riemopt import group_cores
from src.riemopt import compute_gradient_projection
from src.riemopt import optimize


class Test(TestCase):

    def testGradProjection(self):
        np.random.seed(229)

        @jax.jit
        def f_full(A):
            return (A ** 2 - A).sum()

        @jax.jit
        def f(T: Tucker):
            A = T.full()
            return (A ** 2 - A).sum()

        @jax.jit
        def g(T1, core, factors):
            new_factors = [jnp.concatenate([T1.factors[i], factors[i]], axis=1) for i in range(T1.ndim)]
            new_core = group_cores(core, T1.core)

            T = Tucker(new_core, new_factors)
            return f(T)

        full_grad = jax.grad(f_full, argnums=0)

        A = np.random.random((4, 4, 4))
        T = Tucker.full2tuck(A)

        eucl_grad = full_grad(T.full())
        riem_grad = compute_gradient_projection(T, g)

        assert(np.allclose(eucl_grad, riem_grad.full()))

    def testRiemopt(self):
        np.random.seed(123)

        A = np.diag([1, 2, 3, 4])
        Q, _ = np.linalg.qr(np.random.random((4, 4)))
        A = Q @ A @ Q.T
        A = TuckerMatrix.full2tuck(A.reshape([2] * 4), [2] * 2, [2] * 2)

        @jax.jit
        def f(T):
            return (T.flat_inner(A @ T)) / T.flat_inner(T)

        @jax.jit
        def g(T1, core, factors):
            d = T1.ndim
            r = T1.rank

            new_factors = [jnp.concatenate([T1.factors[i], factors[i]], axis=1) for i in range(T1.ndim)]
            new_core = group_cores(core, T1.core)

            T = Tucker(new_core, new_factors)
            return f(T)

        x = np.random.random(4)
        X = Tucker.full2tuck(x.reshape([2] * 2))

        X, _ = optimize(f, g, X, maxiter=100)
        x = X.full().reshape(4)
        A = A.full().reshape((4, 4))

        assert(np.allclose(4 * x, A @ x))


