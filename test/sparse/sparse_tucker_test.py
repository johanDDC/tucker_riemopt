import numpy as np

from unittest import TestCase

from tucker_riemopt import backend as back, Tucker
from tucker_riemopt.sparse import SparseTensor


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
        A = self.creareTestTensor()
        A = SparseTensor.dense2sparse(A)
        A_tuck = Tucker.sparse2tuck(A, max_rank=(2, 3, 4), maxiter=None)
        assert back.norm(A.to_dense() - A_tuck.to_dense()) / back.norm(A.to_dense()) <= 1e-6

    def testHOOI(self):
        A = self.creareTestTensor()
        A = SparseTensor.dense2sparse(A)
        A_tuck = Tucker.sparse2tuck(A, max_rank=(2, 3, 4))
        assert back.norm(A.to_dense() - A_tuck.to_dense()) / back.norm(A.to_dense()) <= 1e-8
