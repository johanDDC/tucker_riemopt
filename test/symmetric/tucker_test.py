import numpy as np

from unittest import TestCase

from tucker_riemopt import backend as back
from tucker_riemopt.symmetric.tucker import Tucker

class TuckerTensorTest(TestCase):
    n = 4

    def createTestTensor(self, n=4):
        """
            A = [G; U, V, V], V = ones(n x n)
        """
        common_factor = np.random.randn(n, n)
        common_factor = back.tensor(common_factor)
        symmetric_factor = back.ones((n, n))
        core = np.random.randn(n, n, n)
        return Tucker(core, [common_factor], 2, symmetric_factor)

    def testAdd(self):
        A = self.createTestTensor(self.n)
        A2 = A + A
        self.assertEqual(A2.rank, (8, 8, 8))
        assert np.allclose((2 * A).full(), A2.full())

    def testNorm(self):
        A = self.createTestTensor(self.n)
        assert np.allclose(A.norm(qr_based=False), back.norm(A.full()))
        assert np.allclose(A.norm(qr_based=True), back.norm(A.full()))

    def testModeProd(self):
        A = self.createTestTensor(self.n)
        Z = back.zeros((self.n, self.n, self.n))
        M = back.zeros((self.n, self.n), dtype=A.dtype)
        assert np.allclose(A.k_mode_product(0, M).full(), Z)
        assert np.allclose(A.k_mode_product(1, M).full(), Z)
        assert np.allclose(A.k_mode_product(2, M).full(), Z)