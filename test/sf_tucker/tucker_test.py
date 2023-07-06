import numpy as np

from unittest import TestCase

from tucker_riemopt import backend as back
from tucker_riemopt.sf_tucker.sf_tucker import SFTucker

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
        return SFTucker(core, [common_factor], 2, symmetric_factor)

    def testAdd(self):
        A = self.createTestTensor(self.n)
        A2 = A + A
        self.assertEqual(A2.rank, [8, 8])
        assert np.allclose((2 * A).to_dense(), A2.to_dense())

    def testNorm(self):
        A = self.createTestTensor(self.n)
        assert np.allclose(A.norm(qr_based=False), back.norm(A.to_dense()))
        assert np.allclose(A.norm(qr_based=True), back.norm(A.to_dense()))

    def testFlatInner(self):
        A = self.createTestTensor(self.n)
        assert np.allclose(back.to_numpy(A.norm(qr_based=True) ** 2), back.to_numpy(A.flat_inner(A)), atol=1e-5)

    def testModeProd(self):
        A = self.createTestTensor(self.n)
        Z = back.zeros((self.n, self.n, self.n))
        M = back.zeros((self.n, self.n), dtype=A.dtype)
        assert np.allclose(A.k_mode_product(0, M).to_dense(), Z)
        assert np.allclose(A.shared_modes_product(M).to_dense(), Z)