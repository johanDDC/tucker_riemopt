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
        core = back.tensor(np.random.randn(n, n, n))
        return Tucker(core, [common_factor], 2, symmetric_factor)

    @staticmethod
    def project_on_tangent(Z, G, U, V):
        def unfold(X, i):
            X = X.transpose([i] + list(range(i)) + list(range(i + 1, len(X.shape))))
            return X.reshape(X.shape[0], -1)

        def tucker(G, U, V, W):
            return back.einsum('abc,ia,jb,kc->ijk', G, U, V, W)

        dot_G = tucker(Z, U.T, U.T, V.T)

        dot_V = tucker(Z, U.T, U.T, np.eye(V.shape[0]))
        G_2 = unfold(G, 2)
        dot_V = unfold(dot_V, 2) @ (G_2.T @ np.linalg.inv(G_2 @ G_2.T))
        dot_V = dot_V - V @ V.T @ dot_V

        P = np.eye(U.shape[0]) - U @ U.T
        Z_1 = np.einsum('abc,def,ad,bg,cf->ge', Z, G, U, P, V)
        Z_2 = np.einsum('abc,def,ag,be,cf->gd', Z, G, P, U, V)

        G_1 = np.einsum('aij,bij->ab', G, G)
        G_2 = np.einsum('iaj,ibj->ab', G, G)

        dot_U = (Z_1 + Z_2) @ np.linalg.inv(G_1 + G_2)
        return tucker(dot_G, U, U, V) + \
            tucker(G, dot_U, U, V) + \
            tucker(G, U, dot_U, V) + \
            tucker(G, U, U, dot_V)

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
        projected_eucl_grad = RiemoptTest.project_on_tangent(eucl_grad, T.core, T.symmetric_factor, T.common_factors[0])

        assert(np.allclose(back.to_numpy(eucl_grad), back.to_numpy(riem_grad.full()), atol=1e-5))
        assert (np.allclose(back.to_numpy(projected_eucl_grad), back.to_numpy(riem_grad.full()), atol=1e-5))