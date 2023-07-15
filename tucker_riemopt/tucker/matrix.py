from string import ascii_letters
from dataclasses import dataclass
from typing import Sequence, Union

from tucker_riemopt import backend as back
from tucker_riemopt import Tucker


@dataclass()
class TuckerMatrix(Tucker):
    n: Union[Sequence[int], None] = None
    m: Union[Sequence[int], None] = None

    @classmethod
    def from_dense(cls, T: back.type(), n: Union[Sequence[int], None] = None, m: Union[Sequence[int], None] = None,
                   eps=1e-14):
        if n is None or m is None:
            raise ValueError("n and m parameter must be specialized for matrices")
        T = cls._hosvd(T, eps=eps)
        return cls(T.core, T.factors, n, m)

    def __matmul__(self, other: Union[back.type(), Tucker, "TuckerMatrix"]):
        """
        Performs matrix multiplication of tensors.

        :param other: can be dense tensor, Tucker or TuckerMatrix. If dense tensor, then treated as vector and matvec
        operation is performed. The result is also a dense tensor. If `Tucker`, then treated as vector and matvec
        operation is performed. The result is also a Tucker. Else matmul operation is performed and another matrix is
        returned.

        :return: dense tensor, Tucker or TuckerMatrix, depends on type of `other`.
        """
        if type(other) == back.type():
            core_letters = ascii_letters[:self.ndim]
            factors_letters = []
            operand_letters = []
            result_letters = []
            reshaped_factors = []
            for i in range(self.ndim):
                reshaped_factors.append(back.reshape(self.factors[i], (self.n[i], self.m[i], -1), order="F"))
                factors_letters.append(ascii_letters[self.ndim + 2 * i: self.ndim + 2 * (i + 1)] + core_letters[i])
                operand_letters.append(factors_letters[-1][0])
                result_letters.append(factors_letters[-1][1])
            return back.einsum(f"{core_letters},{','.join(factors_letters)},{''.join(operand_letters)}->{''.join(result_letters)}",
                               self.core, *reshaped_factors, other)
