import numpy as np

from unittest import TestCase

from tucker_riemopt import backend as back
from tucker_riemopt.symmetric.tucker import Tucker
from tucker_riemopt.symmetric.riemopt import compute_gradient_projection


class RiemoptTest(TestCase):
    def createTestTensor(self, n=4):
        """
            A = [G; U, V, V], V = ones(n x n)
        """
        common_factor = np.random.randn(n, n)
        common_factor = back.tensor(common_factor)
        common_factor = back.qr(common_factor)[0]
        symmetric_factor = back.tensor(np.random.randn(n, n))
        symmetric_factor = back.qr(symmetric_factor)[0]
        symmetric_modes = [1, 2]
        core = back.tensor(np.random.randn(n, n, n))
        return Tucker(core, [common_factor], symmetric_modes, symmetric_factor)

    def testGradProjection(self):
        np.random.seed(229)

        def f_full(A):
            return (A ** 2 - A).sum()

        def f(T: Tucker):
            A = T.full()
            return (A ** 2 - A).sum()

        full_grad = back.grad(f_full, argnums=0)

        T = self.createTestTensor(4)

        eucl_grad = full_grad(T.full())
        riem_grad, _ = compute_gradient_projection(f, T)

        assert(np.allclose(back.to_numpy(eucl_grad), back.to_numpy(riem_grad.full()), atol=1e-5))