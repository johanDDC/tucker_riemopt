import jax
import jax.numpy as jnp
import jax.test_util as jtu

import src.riemanian as riemann
from src.tucker import Tucker

class RiemannTest(jtu.JaxTestCase):
    n = 4

    def testProjection(self):
        pass

    def testGrad(self):
        rng = jax.random.PRNGKey(0)
        tensor = Tucker(jax.random.normal(rng, (3, 3, 3)),
                        [jax.random.normal(rng, (self.n, 3)),
                            jax.random.normal(rng, (self.n, 3)),
                            jax.random.normal(rng, (self.n, 3))])
        f = lambda x: 0.5 * tensor.flat_inner(x) ** 2
        actual_grad = tensor
        riemann_grad = riemann.grad(f)(tensor)
        self.assertAllClose(actual_grad.full(), riemann_grad.full())
