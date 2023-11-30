import numpy as np

from unittest import TestCase

from tucker_riemopt import backend as back
from tucker_riemopt import SFTucker, SFTuckerMatrix
from tucker_riemopt import SFTuckerRiemannian


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
        return SFTucker(core, [common_factor], 2, symmetric_factor)

    @staticmethod
    def project_on_tangent(Z, G, U, V):
        def unfold(X, i):
            X = back.transpose(X, [i] + list(range(i)) + list(range(i + 1, len(X.shape))))
            return X.reshape(X.shape[0], -1)

        def tucker(G, U, V, W):
            return back.einsum('abc,ia,jb,kc->ijk', G, U, V, W)

        dot_G = tucker(Z, U.T, U.T, V.T)

        dot_V = tucker(Z, U.T, U.T, back.eye(V.shape[0]))
        G_2 = unfold(G, 2)
        dot_V = unfold(dot_V, 2) @ (G_2.T @ np.linalg.inv(G_2 @ G_2.T))
        dot_V = dot_V - V @ V.T @ dot_V

        P = back.eye(U.shape[0]) - U @ U.T
        Z_1 = back.einsum('abc,def,ad,bg,cf->ge', Z, G, U, P, V)
        Z_2 = back.einsum('abc,def,ag,be,cf->gd', Z, G, P, U, V)

        G_1 = back.einsum('aij,bij->ab', G, G)
        G_2 = back.einsum('iaj,ibj->ab', G, G)

        dot_U = back.tensor(back.to_numpy(Z_1 + Z_2) @ np.linalg.inv(G_1 + G_2))
        return tucker(dot_G, U, U, V) + \
            tucker(G, dot_U, U, V) + \
            tucker(G, U, dot_U, V) + \
            tucker(G, U, U, dot_V)

    @staticmethod
    def f(T: SFTucker):
        A = T.to_dense()
        return (A ** 2 - A).sum()

    def testGrad(self):
        np.random.seed(229)

        def f_full(A):
            return (A ** 2 - A).sum()

        full_grad = back.grad(f_full, argnums=0)

        T = self.createTestTensor(4)

        eucl_grad = full_grad(T.to_dense())
        tucker_eucl_grad = SFTucker.from_dense(eucl_grad, ds=2)
        riem_grad, _ = SFTuckerRiemannian.grad(self.f, T)
        riem_grad = riem_grad.construct()
        projected_eucl_grad = RiemoptTest.project_on_tangent(eucl_grad, T.core, T.shared_factor, T.regular_factors[0])

        assert (np.allclose(back.to_numpy(eucl_grad), back.to_numpy(riem_grad.to_dense()), atol=1e-5))
        assert (np.allclose(back.to_numpy(projected_eucl_grad), back.to_numpy(riem_grad.to_dense()), atol=1e-5))
        assert np.allclose(back.to_numpy(riem_grad.to_dense()),
                           back.to_numpy(SFTuckerRiemannian.project(T, tucker_eucl_grad).construct().to_dense()),
                           atol=1e-5)

    def testProject(self):
        np.random.seed(229)

        T = self.createTestTensor(4)
        tg_vector, _ = SFTuckerRiemannian.grad(self.f, T)
        tg_vector_proj = SFTuckerRiemannian.project(T, tg_vector.construct())
        assert np.allclose(back.to_numpy(tg_vector.construct().to_dense()),
                           back.to_numpy(tg_vector_proj.construct().to_dense()), atol=1e-5)


    def testZeroTangentVector(self):
        np.random.seed(229)

        T = self.createTestTensor(4)

        tg_vector1, _ = SFTuckerRiemannian.grad(self.f, T)
        zero_tg_vector = SFTuckerRiemannian.TangentVector(tg_vector1.point, back.zeros_like(tg_vector1.point.core))
        tg_vector2 = tg_vector1.linear_comb(xi=zero_tg_vector)

        assert np.allclose(back.to_numpy(tg_vector1.construct().to_dense()),
                           back.to_numpy(tg_vector2.construct().to_dense()), atol=1e-5)

    def testAdd(self):
        np.random.seed(229)

        T = self.createTestTensor(4)
        tg_vector1, _ = SFTuckerRiemannian.grad(self.f, T)
        tg_vector2 = SFTuckerRiemannian.TangentVector(T, back.randn(T.core.shape),
                                                      [back.randn(T.regular_factors[0].shape)],
                                                      back.randn(T.shared_factor.shape))
        addition = tg_vector1 + tg_vector2
        dumb_addition = tg_vector1.construct() + tg_vector2.construct()
        assert (addition.construct() - dumb_addition).norm(qr_based=True) / dumb_addition.norm(qr_based=True) <= 1e-6

    def testScalarMultiplication(self):
        np.random.seed(229)

        T = self.createTestTensor(4)
        tg_vector, _ = SFTuckerRiemannian.grad(self.f, T)
        rmul = 420 * tg_vector
        dumb_rmul = 420 * tg_vector.construct()
        assert (rmul.construct() - dumb_rmul).norm(qr_based=True) / dumb_rmul.norm(qr_based=True) <= 1e-6

    def testLinearComb(self):
        np.random.seed(229)

        a = 2
        b = 3
        T = self.createTestTensor(4)
        tg_vector1, _ = SFTuckerRiemannian.grad(self.f, T)
        tg_vector2 = SFTuckerRiemannian.TangentVector(T, back.randn(T.core.shape),
                                                      [back.randn(T.regular_factors[0].shape)],
                                                      back.randn(T.shared_factor.shape))

        dumb_combination = a * tg_vector1.construct() + b * tg_vector2.construct()
        wise_combination = tg_vector1.linear_comb(a, b, tg_vector2)
        tangent_manifold_point = SFTuckerRiemannian.TangentVector(T)
        dumb_point_combination = a * tg_vector1.construct() + b * T
        wise_point_combination = tg_vector1.linear_comb(a, b)

        assert np.allclose(back.to_numpy(dumb_combination.to_dense()),
                           back.to_numpy(wise_combination.construct().to_dense()), atol=1e-5)
        assert np.allclose(back.to_numpy(T.to_dense()),
                           back.to_numpy(tangent_manifold_point.construct().to_dense()), atol=1e-5)
        assert np.allclose(back.to_numpy(dumb_point_combination.to_dense()),
                           back.to_numpy(wise_point_combination.construct().to_dense()), atol=1e-5)


    def testMatrixGrad(self):
        for num_shared_factors in [1, 3]:
            eye = back.ones((8, 8))
            matrix = back.copy(eye)
            eye = back.reshape(eye, (2, 2, 2, 2, 2, 2))
            eye = back.transpose(eye, (0, 3, 1, 4, 2, 5))
            eye = back.reshape(eye, (4, 4, 4))
            eye = SFTuckerMatrix.from_dense(eye, num_shared_factors, (2, 2, 2), (2, 2, 2), eps=1e-7)
            x = back.ones(8)
            x = back.reshape(x, (2, 2, 2))
            x_dense = back.reshape(x, (8,))

            loss = lambda A: back.norm(A @ x) ** 2
            loss_dense = lambda A: back.norm(A @ x_dense) ** 2

            eucl_grad = back.grad(loss_dense, argnums=0)(matrix)
            riem_grad, _ = SFTuckerRiemannian.grad(loss, eye)
            riem_grad = riem_grad.construct()

            riem_grad = riem_grad.to_dense()
            riem_grad = back.reshape(riem_grad, (2, 2, 2, 2, 2, 2))
            riem_grad = back.transpose(riem_grad, (0, 2, 4, 1, 3, 5))
            riem_grad = back.reshape(riem_grad, (8, 8))

            assert(np.allclose(back.to_numpy(eucl_grad), back.to_numpy(riem_grad.to_dense()), atol=1e-5))

