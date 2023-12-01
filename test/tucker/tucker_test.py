import numpy as np

from unittest import TestCase

from tucker_riemopt import Tucker
from tucker_riemopt import backend as back


class TuckerTensorTest(TestCase):
    n = 4

    def createTestTensor(self, n=4):
        """
            A_ijk = i + j + k
        """
        x = back.arange(n) + 1
        e = back.ones(n, dtype=back.float64)
        A = back.einsum("i,j,k->ijk", x, e, e) + \
            back.einsum("i,j,k->ijk", e, x, e) + \
            back.einsum("i,j,k->ijk", e, e, x)
        return A

    def testFull2Tuck(self):
        A = self.createTestTensor(self.n)
        A_tuck = Tucker.from_dense(A, eps=1e-6)
        assert np.allclose(A, A_tuck.to_dense())

    def testAdd(self):
        A = self.createTestTensor(self.n)
        A = Tucker.from_dense(A, eps=1e-6)
        A2 = A + A
        self.assertEqual(A2.rank, (4, 4, 4))
        A2 = A2.round(A.rank)
        self.assertEqual(A2.rank, (2, 2, 2))
        assert np.allclose((2 * A).to_dense(), A2.to_dense())

    def testMul(self):
        A = self.createTestTensor(self.n)
        A_tuck = Tucker.from_dense(A, eps=1e-6)
        self.assertEqual(A_tuck.rank, (2, 2, 2))
        A_tuck = A_tuck * A_tuck
        self.assertEqual(A_tuck.rank, (4, 4, 4))
        A_tuck = A_tuck.round(eps=1e-6)
        self.assertEqual(A_tuck.rank, (3, 3, 3))

    def testNorm(self):
        A = self.createTestTensor(self.n)
        A_tuck = Tucker.from_dense(A, eps=1e-6)
        assert np.allclose(A_tuck.norm(qr_based=False), back.norm(A))
        assert np.allclose(A_tuck.norm(qr_based=True), back.norm(A))

    def testModeProd(self):
        A = self.createTestTensor(self.n)
        Z = back.zeros((self.n, self.n, self.n))
        A_tuck = Tucker.from_dense(A, eps=1e-6)
        M = back.zeros((self.n, self.n), dtype=A.dtype)
        assert np.allclose(A_tuck.k_mode_product(0, M).to_dense(), Z)
        assert np.allclose(A_tuck.k_mode_product(1, M).to_dense(), Z)
        assert np.allclose(A_tuck.k_mode_product(2, M).to_dense(), Z)
