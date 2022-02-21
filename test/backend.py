import jax.test_util as jtu
from src import set_backend

import tucker

class TuckerTensorTest(jtu.JaxTestCase):
    def testJaxBackend(self):
        set_backend("jax")
        instance = tucker.TuckerTensorTest()
        instance.testFull2Tuck()
        instance.testAdd()
        instance.testMul()
        instance.testNorm()
        instance.testModeProd()


    def testPytorchBackend(self):
        set_backend("pytorch")
        instance = tucker.TuckerTensorTest()
        instance.testFull2Tuck()
        instance.testAdd()
        instance.testMul()
        instance.testNorm()
        instance.testModeProd()

    def testBackend(self):
        self.testJaxBackend()
        self.testPytorchBackend()