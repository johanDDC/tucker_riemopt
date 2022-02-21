from flax import struct
from string import ascii_letters
from copy import copy

from src import backend as back
from src.tucker import Tucker

@struct.dataclass
class TuckerMatrix(Tucker):
    n: back.type
    m: back.type

    @classmethod
    def full2tuck(cls, T : back.type, n=None, m=None, max_rank=10, eps=1e-14):
        T = Tucker.full2tuck(T, max_rank=max_rank, eps=eps)
        return cls(T.core, T.factors, back.tensor(n), back.tensor(m))

    def __matmul__(self, other):
        """
        Multiplication of two Tucker-matrices
        """
        d1 = len(self.n)
        d2 = len(self.m)

        if isinstance(other, TuckerMatrix):
            d3 = len(other.n)
            d4 = len(other.m)
        else:
            d3 = other.ndim
            d4 = 0

        if d2 != d3:
            raise ValueError(f"Cant multiply tensors with dimensions {d1}x{d2} and {d3}x{d4}")

        letters = ascii_letters
        l1 = letters[:d1]
        letters = letters[d1:]

        l2 = letters[:d2]
        letters = letters[d2:]

        l3 = letters[:d2]
        letters = letters[d2:]

        l4 = letters[:d2]
        letters = letters[d2:]

        l5 = letters[:d4]
        letters = letters[d4:]

        einsum_str = ''
        einsum_str += l1 + l2 + ','
        for i in range(d2):
            einsum_str += l3[i] + l2[i] + ','
        einsum_str += l4 + l5 + ','
        for i in range(d2):
            einsum_str += l3[i] + l4[i] + ','
        einsum_str = einsum_str[:-1] + '->' + l1 + l5

        new_core = back.einsum(einsum_str, self.core, *self.factors[-d2:], other.core, *other.factors[:d2], optimize='optimal')
        new_factors = self.factors[:d1]
        if d4 != 0:
            new_factors += other.factors[-d4:]

        return TuckerMatrix(new_core, new_factors, d2, d4)

    def __rmul__(self, other):
        if hasattr(other, '__matmul__') and not isinstance(other, back.type):
            return other.__matmul__(self)
        else:
            new_tensor = copy(self)
            return TuckerMatrix(other * new_tensor.core, new_tensor.factors, new_tensor.n, new_tensor.m)
