import numpy as np

from unittest import TestCase

from tucker_riemopt import backend as back, set_backend
from tucker_riemopt import Tucker
from tucker_riemopt import TuckerRiemannian


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

    @staticmethod
    def f(T: Tucker):
        A = T.to_dense()
        return (A ** 2 - A).sum()

    def testGradProjection(self):
        set_backend("pytorch")
        np.random.seed(229)

        def f_full(A):
            return (A ** 2 - A).sum()

        full_grad = back.grad(f_full, argnums=0)

        T = self.createTestTensor(4)

        eucl_grad = full_grad(T.to_dense())
        riem_grad, _ = TuckerRiemannian.grad(self.f, T)
        riem_grad = riem_grad.construct()

        assert(np.allclose(back.to_numpy(eucl_grad), back.to_numpy(riem_grad.to_dense()), atol=1e-5))

    def testProject(self):
        np.random.seed(229)

        T = self.createTestTensor(4)
        tg_vector, _ = TuckerRiemannian.grad(self.f, T)
        tg_vector_proj = TuckerRiemannian.project(T, tg_vector.construct())
        assert np.allclose(back.to_numpy(tg_vector.construct().to_dense()),
                           back.to_numpy(tg_vector_proj.construct().to_dense()), atol=1e-5)


    def testZeroTangentVector(self):
        np.random.seed(229)

        T = self.createTestTensor(4)

        tg_vector1, _ = TuckerRiemannian.grad(self.f, T)
        zero_tg_vector = TuckerRiemannian.TangentVector(tg_vector1.point, back.zeros_like(tg_vector1.point.core))
        tg_vector2 = tg_vector1.linear_comb(xi=zero_tg_vector)

        assert np.allclose(back.to_numpy(tg_vector1.construct().to_dense()),
                           back.to_numpy(tg_vector2.construct().to_dense()), atol=1e-5)

    def testLinearComb(self):
        np.random.seed(229)

        a = 2
        b = 3
        T = self.createTestTensor(4)
        tg_vector1, _ = TuckerRiemannian.grad(self.f, T)
        tg_vector2 = TuckerRiemannian.TangentVector(T, back.randn(T.core.shape),
                                                      [back.randn(T.regular_factors[0].shape) for _ in range(T.ndim)])

        dumb_combination = a * tg_vector1.construct() + b * tg_vector2.construct()
        wise_combination = tg_vector1.linear_comb(a, b, tg_vector2)
        tangent_manifold_point = TuckerRiemannian.TangentVector(T)
        dumb_point_combination = a * tg_vector1.construct() + b * T
        wise_point_combination = tg_vector1.linear_comb(a, b)

        assert np.allclose(back.to_numpy(dumb_combination.to_dense()),
                           back.to_numpy(wise_combination.construct().to_dense()), atol=1e-5)
        assert np.allclose(back.to_numpy(T.to_dense()),
                           back.to_numpy(tangent_manifold_point.construct().to_dense()), atol=1e-5)
        assert np.allclose(back.to_numpy(dumb_point_combination.to_dense()),
                           back.to_numpy(wise_point_combination.construct().to_dense()), atol=1e-5)

