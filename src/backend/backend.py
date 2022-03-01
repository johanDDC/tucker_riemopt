import warnings

import numpy as np
import typing

class Backend(object):
    _available_backends = dict()

    def __init_subclass__(cls, backend_name, **kwargs):
        super().__init_subclass__(**kwargs)

        if backend_name != "":
            cls._available_backends[backend_name.lower()] = cls
            cls.backend_name = backend_name
        else:
            warnings.warn(f"Creating a subclass of BaseBackend ({cls.__name__}) with no name.")

    def __repr__(self):
        return f"Tucker riemopt {self.backend_name}-backend"

    @classmethod
    def register_method(cls, name, func):
        setattr(cls, name, staticmethod(func))

    @property
    def int64(self):
        raise NotImplementedError

    @property
    def int32(self):
        raise NotImplementedError

    @property
    def float64(self):
        raise NotImplementedError

    @property
    def float32(self):
        raise NotImplementedError

    @property
    def complex128(self):
        raise NotImplementedError

    @property
    def complex64(self):
        raise NotImplementedError

    @staticmethod
    def type():
        raise NotImplementedError

    @staticmethod
    def check_random_state(seed):
        if seed is None:
            return np.random.mtrand._rand
        elif isinstance(seed, int):
            return np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            return seed

        raise ValueError("Seed should be None, int or np.random.RandomState")

    def randn(self, shape, seed=None, **context):
        rng = self.check_random_state(seed)
        random_tensor = rng.randn(*shape)
        random_tensor = self.tensor(random_tensor, **context)
        return random_tensor

    @staticmethod
    def context(tensor):
        raise NotImplementedError

    @staticmethod
    def tensor(data, **context):
        raise NotImplementedError

    @staticmethod
    def is_tensor(obj):
        raise NotImplementedError

    @staticmethod
    def shape(tensor):
        raise NotImplementedError

    @staticmethod
    def ndim(tensor):
        raise NotImplementedError

    @staticmethod
    def to_numpy(tensor):
        raise NotImplementedError

    @staticmethod
    def copy(tensor):
        raise NotImplementedError

    @staticmethod
    def concatenate(tensors, axis=0):
        raise NotImplementedError

    @staticmethod
    def reshape(tensor, newshape, order="C"):
        raise NotImplementedError

    @staticmethod
    def transpose(tensor):
        raise NotImplementedError

    @staticmethod
    def arange(start=0, stop=None, step=1):
        raise NotImplementedError

    @staticmethod
    def ones(shape, dtype=None):
        raise NotImplementedError

    @staticmethod
    def zeros(shape, dtype=None):
        raise NotImplementedError

    @staticmethod
    def zeros_like(tensor):
        raise NotImplementedError

    @staticmethod
    def diag(diagnoal):
        raise NotImplementedError

    @staticmethod
    def eye(N):
        raise NotImplementedError

    @staticmethod
    def count_nonzero(tensor):
        raise NotImplementedError

    @staticmethod
    def trace(tensor):
        raise NotImplementedError

    @staticmethod
    def cumsum(tensor, axis=None):
        raise NotImplementedError

    @staticmethod
    def where(condition, x, y):
        raise NotImplementedError

    @staticmethod
    def any(tensor, axis=None, keepdims=False, **kwargs):
        return tensor.any(axis=axis, keepdims=keepdims, **kwargs)

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        raise NotImplementedError

    @staticmethod
    def max(tensor):
        raise NotImplementedError

    @staticmethod
    def min(tensor):
        raise NotImplementedError

    @staticmethod
    def argmax(tensor):
        raise NotImplementedError

    @staticmethod
    def argmin(tensor):
        raise NotImplementedError

    @staticmethod
    def all(tensor):
        raise NotImplementedError

    @staticmethod
    def mean(tensor, axis=None):
        raise NotImplementedError

    @staticmethod
    def sum(tensor, axis=None):
        raise NotImplementedError

    @staticmethod
    def prod(tensor, axis=None):
        raise NotImplementedError

    @staticmethod
    def sign(tensor):
        raise NotImplementedError

    @staticmethod
    def abs(tensor):
        raise NotImplementedError

    @staticmethod
    def sqrt(tensor):
        raise NotImplementedError

    def norm(self, tensor, order=2, axis=None):
        if axis == ():
            axis = None

        if order == "inf":
            return self.max(self.abs(tensor), axis=axis)
        if order == 1:
            return self.sum(self.abs(tensor), axis=axis)
        elif order == 2:
            return self.sqrt(self.sum(self.abs(tensor) ** 2, axis=axis))
        else:
            return self.sum(self.abs(tensor) ** order, axis=axis) ** (1 / order)

    @staticmethod
    def dot(a, b):
        raise NotImplementedError

    @staticmethod
    def matmul(a, b):
        raise NotImplementedError

    @staticmethod
    def solve(a, b):
        raise NotImplementedError

    @staticmethod
    def lstsq(a, b):
        raise NotImplementedError

    @staticmethod
    def qr(a):
        raise NotImplementedError

    @staticmethod
    def stack(arrays, axis=0):
        raise NotImplementedError

    def eps(self, dtype):
        return self.finfo(dtype).eps

    def finfo(self, dtype):
        return np.finfo(self.to_numpy(self.tensor([], dtype=dtype)).dtype)

    @staticmethod
    def conj(x, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def sort(tensor, axis, descending=False):
        raise NotImplementedError

    @staticmethod
    def argsort(tensor, axis, descending=False):
        raise NotImplementedError

    @staticmethod
    def einsum(subscripts, *operands):
        raise NotImplementedError

    def kron(self, a, b):
        s1, s2 = self.shape(a)
        s3, s4 = self.shape(b)
        a = self.reshape(a, (s1, 1, s2, 1))
        b = self.reshape(b, (1, s3, 1, s4))
        return self.reshape(a * b, (s1 * s3, s2 * s4))

    def khatri_rao(self, matrices, weights=None, mask=None):
        if len(matrices) < 2:
            raise ValueError(f"kr requires a list of at least 2 matrices, but {len(matrices)} given.")

        n_col = self.shape(matrices[0])[1]
        for i, e in enumerate(matrices[1:]):
            if not i:
                if weights is None:
                    res = matrices[0]
                else:
                    res = matrices[0] * self.reshape(weights, (1, -1))
            s1, s2 = self.shape(res)
            s3, s4 = self.shape(e)
            if not s2 == s4 == n_col:
                raise ValueError("All matrices should have the same number of columns.")

            a = self.reshape(res, (s1, 1, s2))
            b = self.reshape(e, (1, s3, s4))
            res = self.reshape(a * b, (-1, n_col))

        m = self.reshape(mask, (-1, 1)) if mask is not None else 1
        return res * m

    def svd(self, matrix):
        raise NotImplementedError

    def eigh(self, matrix):
        raise NotImplementedError

    @staticmethod
    def log2(x):
        raise NotImplementedError

    @staticmethod
    def sin(x):
        raise NotImplementedError

    @staticmethod
    def cos(x):
        raise NotImplementedError

    @staticmethod
    def grad(func: typing.Callable, argnums: typing.Union[int, typing.Sequence[int]] = 0):
        raise NotImplementedError

    @staticmethod
    def pad(tensor, pad_width, constant_values):
        """
        Only constant mode is supported
        :param tensor: tensor to pad
        :param pad_width: tuple of axis to pad in PRIMAL order
        :param constant_values: values to fill padded region
        :return: padded tensor
        """
        raise NotImplementedError
