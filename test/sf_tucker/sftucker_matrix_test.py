import numpy as np

from unittest import TestCase

from tucker_riemopt import SFTuckerMatrix
from tucker_riemopt import backend as back


class SFTuckerMatrixTest(TestCase):
    def testDenseMatmul(self):
        matrix_dense = back.randn((8, 8))
        x_dense = back.randn((8,))
        x = back.reshape(x_dense, (2, 2, 2))
        y_dense = matrix_dense @ x_dense
        matrix_dense = back.reshape(matrix_dense, (2, 2, 2, 2, 2, 2))
        matrix_dense = back.transpose(matrix_dense, (0, 3, 1, 4, 2, 5))
        matrix_dense = back.reshape(matrix_dense, (4, 4, 4))
        for num_shared_factors in [1, 2, 3]:
            matrix_tucker = SFTuckerMatrix.from_dense(matrix_dense, num_shared_factors, (2, 2, 2), (2, 2, 2))
            y = matrix_tucker @ x
            assert np.allclose(back.to_numpy(y_dense), back.to_numpy(back.reshape(y, (8,))), atol=1e-5)

    def testDenseBatchMatmul(self):
        matrix_dense = back.randn((8, 8))
        x_dense = back.randn((10, 8))
        y_dense = (matrix_dense @ x_dense.T).T
        x = back.reshape(x_dense, (10, 2, 2, 2))
        matrix_dense = back.reshape(matrix_dense, (2, 2, 2, 2, 2, 2))
        matrix_dense = back.transpose(matrix_dense, (0, 3, 1, 4, 2, 5))
        matrix_dense = back.reshape(matrix_dense, (4, 4, 4))
        for num_shared_factors in [1, 2, 3]:
            matrix_tucker = SFTuckerMatrix.from_dense(matrix_dense, num_shared_factors, (2, 2, 2), (2, 2, 2))
            y = matrix_tucker @ x
            assert np.allclose(back.to_numpy(y_dense), back.to_numpy(back.reshape(y, (10, 8))), atol=1e-5)
