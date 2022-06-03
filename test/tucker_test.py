from unittest import TestCase
import numpy as np

from tucker_riemopt import Tucker, SparseTensor
from tucker_riemopt import backend as back
from tucker_riemopt import set_backend


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
        A_tuck = Tucker.full2tuck(A, eps=1e-6)
        assert np.allclose(A, A_tuck.full())

    def testAdd(self):
        A = self.createTestTensor(self.n)
        A = Tucker.full2tuck(A, eps=1e-6)
        A2 = A + A
        self.assertEqual(A2.rank, (4, 4, 4))
        A2 = A2.round(A.rank)
        self.assertEqual(A2.rank, (2, 2, 2))
        assert np.allclose((2 * A).full(), A2.full())

    def testMul(self):
        A = self.createTestTensor(self.n)
        A_tuck = Tucker.full2tuck(A, eps=1e-6)
        A_rank = A_tuck.rank
        self.assertEqual(A_tuck.rank, (2, 2, 2))
        A_tuck = A_tuck * A_tuck
        self.assertEqual(A_tuck.rank, (4, 4, 4))
        A_tuck = A_tuck.round(eps=1e-6)
        self.assertEqual(A_tuck.rank, (3, 3, 3))

    def testNorm(self):
        A = self.createTestTensor(self.n)
        A_tuck = Tucker.full2tuck(A, eps=1e-6)
        assert np.allclose(A_tuck.norm(qr_based=False), back.norm(A))
        assert np.allclose(A_tuck.norm(qr_based=True), back.norm(A))

    def testModeProd(self):
        A = self.createTestTensor(self.n)
        Z = back.zeros((self.n, self.n, self.n))
        A_tuck = Tucker.full2tuck(A, eps=1e-6)
        M = back.zeros((self.n, self.n), dtype=A.dtype)
        assert np.allclose(A_tuck.k_mode_product(0, M).full(), Z)
        assert np.allclose(A_tuck.k_mode_product(1, M).full(), Z)
        assert np.allclose(A_tuck.k_mode_product(2, M).full(), Z)


class SparseTensorTest(TestCase):
    def creareTestTensor(self):
        return back.tensor([
            [
                [0, 1, 0, 0],
                [2, 0, 0, 1],
                [3, 0, 1, 0],
            ],
            [
                [0, 0, 1, 1],
                [0, 3, 1, 0],
                [0, 0, 0, 0],
            ]
        ])

    def testToDense(self):
        inds = (
            back.tensor([0, 1, 0, 0, 0, 1, 0, 1, 1], dtype=back.int32),
            back.tensor([1, 0, 2, 0, 1, 1, 2, 0, 1], dtype=back.int32),
            back.tensor([0, 3, 0, 1, 3, 1, 2, 2, 2], dtype=back.int32)
        )
        vals = back.tensor([2, 1, 3, 1, 1, 3, 1, 1, 1], dtype=back.int32)
        shape = (2, 3, 4)
        A = SparseTensor(shape, inds, vals)
        assert (A.to_dense() == self.creareTestTensor()).all()

    def testFromDense(self):
        A = back.randn((3, 4, 5))
        assert (SparseTensor.dense2sparse(A).to_dense() == A).all()

    def testUnfolding(self):
        unfolding0 = back.tensor([[0, 2, 3, 1, 0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 3, 0, 1, 1, 0, 1, 0, 0]])
        unfolding1 = back.tensor([[0, 0, 1, 0, 0, 1, 0, 1], [2, 0, 0, 3, 0, 1, 1, 0], [3, 0, 0, 0, 1, 0, 0, 0]])
        unfolding2 = back.tensor([[0, 0, 2, 0, 3, 0], [1, 0, 0, 3, 0, 0], [0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0]])
        A = self.creareTestTensor()
        A = SparseTensor.dense2sparse(A)
        assert (back.tensor(A.unfolding(0).todense()) == unfolding0).all()
        assert (back.tensor(A.unfolding(1).todense()) == unfolding1).all()
        assert (back.tensor(A.unfolding(2).todense()) == unfolding2).all()

    def testContraction(self):
        Xs = (
            back.arange(1 * 2).reshape((1, 2)),
            back.arange(2 * 3).reshape((2, 3)),
            back.arange(1 * 4).reshape((1, 4)),
        )
        A = self.creareTestTensor()
        A = SparseTensor.dense2sparse(A)
        res = back.tensor([[[5], [35]]])
        assert np.allclose(A.contract({0: Xs[0], 1: Xs[1], 2: Xs[2]}).to_dense(), res)


class SparseTuckerTest(TestCase):
    def creareTestTensor(self):
        return back.tensor([
            [
                [0, 1, 0, 0],
                [2, 0, 0, 1],
                [3, 0, 1, 0],
            ],
            [
                [0, 0, 1, 1],
                [0, 3, 1, 0],
                [0, 0, 0, 0],
            ]
        ])

    def testSparseHOSVD(self):
        set_backend("pytorch")
        A = self.creareTestTensor()
        A = SparseTensor.dense2sparse(A)
        A_tuck = Tucker.sparse2tuck(A, max_rank=(1, 2, 3), maxiter=None)
        assert back.norm(A.to_dense() - A_tuck.full()) / back.norm(A.to_dense()) <= 0.71

    def testHOOI(self):
        set_backend("pytorch")
        A = self.creareTestTensor()
        A = SparseTensor.dense2sparse(A)
        A_tuck = Tucker.sparse2tuck(A, max_rank=(1, 2, 3))
        assert back.norm(A.to_dense() - A_tuck.full()) / back.norm(A.to_dense()) <= 0.68
