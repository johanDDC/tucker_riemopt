import numpy as np

from unittest import TestCase

from tucker_riemopt import backend as back, set_backend
from tucker_riemopt import Tucker
from tucker_riemopt.tucker.riemannian import grad


class RiemannianTest(TestCase):

    def createTestTensor(self, n=4):
        """
            A = [G; U, V, V], V = ones(n x n)
        """
        common_factor = np.random.randn(n, n)
        common_factor = back.tensor(common_factor)
        common_factor = back.qr(common_factor)[0]
        symmetric_factor = back.tensor(np.random.randn(n, n))
        symmetric_factor = back.qr(symmetric_factor)[0]
        core = back.tensor(np.random.randn(n, n, n))
        return Tucker(core, [common_factor, symmetric_factor, symmetric_factor])

    def testGradProjection(self):
        set_backend("pytorch")
        np.random.seed(229)

        def f_full(A):
            return (A ** 2 - A).sum()

        def f(T: Tucker):
            A = T.to_dense()
            return (A ** 2 - A).sum()

        full_grad = back.grad(f_full, argnums=0)

        T = self.createTestTensor(4)

        eucl_grad = full_grad(T.to_dense())
        riem_grad, _ = grad(f, T)
        riem_grad = riem_grad.construct()

        assert(np.allclose(back.to_numpy(eucl_grad), back.to_numpy(riem_grad.to_dense()), atol=1e-5))


