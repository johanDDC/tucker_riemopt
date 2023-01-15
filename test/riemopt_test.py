import numpy as np

from unittest import TestCase

from tucker_riemopt import backend as back
from tucker_riemopt import Tucker
from tucker_riemopt.riemopt import compute_gradient_projection


class RiemoptTest(TestCase):

    def testGradProjection(self):
        np.random.seed(229)

        def f_full(A):
            return (A ** 2 - A).sum()

        def f(T: Tucker):
            A = T.full()
            return (A ** 2 - A).sum()

        full_grad = back.grad(f_full, argnums=0)

        A = back.randn((100, 100, 100))
        T = Tucker.full2tuck(A)

        eucl_grad = full_grad(T.full())
        riem_grad = compute_gradient_projection(f, T)

        assert(np.allclose(back.to_numpy(eucl_grad), back.to_numpy(riem_grad.full()), atol=1e-5))


