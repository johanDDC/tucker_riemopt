from unittest import TestCase
from tucker_riemopt import set_backend

from tucker import tucker_test

class BackendTest(TestCase):
    def testJaxBackend(self):
        set_backend("jax")
        instance = tucker_test.TuckerTensorTest()
        instance.testFull2Tuck()
        instance.testAdd()
        instance.testMul()
        instance.testNorm()
        instance.testModeProd()


    def testPytorchBackend(self):
        set_backend("pytorch")
        instance = tucker_test.TuckerTensorTest()
        instance.testFull2Tuck()
        instance.testAdd()
        instance.testMul()
        instance.testNorm()
        instance.testModeProd()

    def testBackend(self):
        self.testJaxBackend()
        self.testPytorchBackend()