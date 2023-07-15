import numpy as np

from unittest import TestCase

from tucker_riemopt import Tucker
from tucker_riemopt import TuckerMatrix
from tucker_riemopt import backend as back


class TuckerMatrixTest(TestCase):
    def testDenseMatmul(self):
        eye = back.eye(8)
        eye = back.reshape(eye, (2, 2, 2, 2, 2, 2), order="F")
        eye = back.transpose(eye, (0, 3, 1, 4, 2, 5))
        eye = back.reshape(eye, (4, 4, 4), order="F")
        eye = TuckerMatrix.from_dense(eye, (2, 2, 2), (2, 2, 2))
        x = back.randn((2, 2, 2))
        y = eye @ x
        assert np.allclose(back.to_numpy(y), back.to_numpy(x))
