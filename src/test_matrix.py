from matrix import TuckerMatrix
from tucker import Tucker

import numpy as np

A = np.random.random((4, 4, 4))

M1 = TuckerMatrix.full2tuck(A, [4, 4], [4])
M2 = TuckerMatrix.full2tuck(A, [4], [4, 4])

res = M1.__matmul__(M2)
assert(res.shape == (4, 4, 4, 4))
print('Test #1 passed!')

A1 = np.random.random((3, 3))
A2 = np.random.random((3, 3))

M1 = TuckerMatrix.full2tuck(A1, [3], [3])
M2 = TuckerMatrix.full2tuck(A2, [3], [3])

res = M1.__matmul__(M2)
assert(np.allclose(res.full(), A1 @ A2))
print('Test #2 passed!')

v = np.random.random(4)
A = np.random.random((4, 4))

V = Tucker.full2tuck(v.reshape([2] * 2))
M = TuckerMatrix.full2tuck(A.reshape([2] * 4), [2] * 2, [2] * 2)

res = M.__matmul__(V)
assert(np.allclose(res.full().flatten(), A @ v))
print('Test #3 passed!')

A1 = np.random.random((3, 3))
A2 = np.random.random((3, 3))

M1 = TuckerMatrix.full2tuck(A1, [3], [3])
M2 = TuckerMatrix.full2tuck(A2, [3], [3])

res = M1 + 2 * M2
assert(np.allclose(res.full(), A1 + 2 * A2))
print('Test #4 passed!')

print('All tests passed!')