import numpy as np

from unittest import TestCase

from tucker_riemopt import backend as back
from tucker_riemopt.sparse import SparseTensor


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