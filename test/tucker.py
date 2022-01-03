import jax.numpy as jnp
import jax.test_util as jtu

from src.tucker import Tucker

def createTestTensor(n = 4):
    """
        A_ijk = i + j + k
    """
    x = jnp.arange(n) + 1
    e = jnp.ones(n)
    A = jnp.einsum("i,j,k->ijk", x, e, e) + \
        jnp.einsum("i,j,k->ijk", e, x, e) + \
        jnp.einsum("i,j,k->ijk", e, e, x)
    return A

class TuckerTensorTest(jtu.JaxTestCase):
    n = 1000

    def testFull2Tuck(self):
        A = createTestTensor(self.n)
        A_tuck = Tucker.full2tuck(A, 1e-6)
        self.assertAllClose(A, A_tuck.full())

    def testAdd(self):
        A = createTestTensor(self.n)
        A = Tucker.full2tuck(A, 1e-6)
        A2 = A + A
        self.assertEqual(A2.rank, (4, 4, 4))
        A2 = A2.round(1e-6)
        self.assertEqual(A2.rank, (2, 2, 2))
        self.assertAllClose((2 * A).full(), A2.full())

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
        self.assertAlmostEqual(A_tuck.norm(qr_based=False), jnp.linalg.norm(A), places=4)
        self.assertAlmostEqual(A_tuck.norm(qr_based=True), jnp.linalg.norm(A), places=4)

    def testModeProd(self):
        A = createTestTensor(self.n)
        Z = jnp.zeros((self.n, self.n, self.n))
        A_tuck = Tucker.full2tuck(A, 1e-6)
        M = jnp.zeros((self.n, self.n))
        self.assertAllClose(A_tuck.k_mode_product(0, M).full(), Z)
        self.assertAllClose(A_tuck.k_mode_product(1, M).full(), Z)
        self.assertAllClose(A_tuck.k_mode_product(2, M).full(), Z)




