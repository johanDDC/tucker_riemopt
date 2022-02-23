from unittest import TestCase
from src.tucker import Tucker
import numpy as np
from src import backend as back

def createTestTensor(n = 4):
    """
        A_ijk = i + j + k
    """
    x = back.arange(n) + 1
    e = back.ones(n, dtype=back.float64)
    A = back.einsum("i,j,k->ijk", x, e, e) + \
        back.einsum("i,j,k->ijk", e, x, e) + \
        back.einsum("i,j,k->ijk", e, e, x)
    return A

class TuckerTensorTest(TestCase):
    n = 4

    def testFull2Tuck(self):
        A = createTestTensor(self.n)
        A_tuck = Tucker.full2tuck(A, 1e-6)
        assert np.allclose(A, A_tuck.full())
        # self.assertAllClose(A, A_tuck.full(), rtol=1e-14)

    def testAdd(self):
        A = createTestTensor(self.n)
        A = Tucker.full2tuck(A, 1e-6)
        A2 = A + A
        self.assertEqual(A2.rank, (4, 4, 4))
        A2 = A2.round(1e-6)
        self.assertEqual(A2.rank, (2, 2, 2))
        assert np.allclose((2 * A).full(), A2.full())
        # self.assertAllClose((2 * A).full(), A2.full())

    def testMul(self):
        A = createTestTensor(self.n)
        A_tuck = Tucker.full2tuck(A, 1e-6)
        self.assertEqual(A_tuck.rank, (2, 2, 2))
        A_tuck = A_tuck * A_tuck
        self.assertEqual(A_tuck.rank, (4, 4, 4))
        A = A * A
        A_tuck = A_tuck.round(1e-6)
        self.assertEqual(A_tuck.rank, (3, 3, 3))

    def testNorm(self):
        A = createTestTensor(self.n)
        A_tuck = Tucker.full2tuck(A, 1e-6)
        assert np.allclose(A_tuck.norm(qr_based=False), back.norm(A))
        assert np.allclose(A_tuck.norm(qr_based=True), back.norm(A))

    def testModeProd(self):
        A = createTestTensor(self.n)
        Z = back.zeros((self.n, self.n, self.n))
        A_tuck = Tucker.full2tuck(A, 1e-6)
        M = back.zeros((self.n, self.n))
        assert np.allclose(A_tuck.k_mode_product(0, M).full(), Z)
        assert np.allclose(A_tuck.k_mode_product(1, M).full(), Z)
        assert np.allclose(A_tuck.k_mode_product(2, M).full(), Z)
        # self.assertAllClose(A_tuck.k_mode_product(0, M).full(), Z)
        # self.assertAllClose(A_tuck.k_mode_product(1, M).full(), Z)
        # self.assertAllClose(A_tuck.k_mode_product(2, M).full(), Z)




