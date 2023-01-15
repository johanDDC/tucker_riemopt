import numpy as np

from unittest import TestCase

from tucker_riemopt import Tucker
from tucker_riemopt import TuckerMatrix
from tucker_riemopt import backend as back


class TuckerMatrixTest(TestCase):
    def testShape(self):
        A = back.randn((4, 4, 4))

        M1 = TuckerMatrix.full2tuck(A, [4, 4], [4])
        M2 = TuckerMatrix.full2tuck(A, [4], [4, 4])

        res = M1 @ M2
        assert(res.shape == (4, 4, 4, 4))

    def testMatMul(self):
        A1 = back.randn((3, 3))
        A2 = back.randn((3, 3))

        M1 = TuckerMatrix.full2tuck(A1, [3], [3])
        M2 = TuckerMatrix.full2tuck(A2, [3], [3])

        res = M1 @ M2
        assert(np.allclose(res.full(), A1 @ A2))

    def testMatVec(self):
        v = back.randn((4,))
        A = back.randn((4, 4))

        V = Tucker.full2tuck(v.reshape([2] * 2))
        M = TuckerMatrix.full2tuck(A.reshape([2] * 4), [2] * 2, [2] * 2)

        res = M @ V
        assert(np.allclose(res.full().flatten(), A @ v))

    def testAddMul(self):
        A1 = back.randn((3, 3))
        A2 = back.randn((3, 3))

        M1 = TuckerMatrix.full2tuck(A1, [3], [3])
        M2 = TuckerMatrix.full2tuck(A2, [3], [3])

        res = M1 + 2 * M2
        assert(np.allclose(res.full(), A1 + 2 * A2))
