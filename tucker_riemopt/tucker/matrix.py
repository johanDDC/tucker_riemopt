from string import ascii_letters
from dataclasses import dataclass
from typing import Sequence, Union

from tucker_riemopt import backend as back
from tucker_riemopt import Tucker


@dataclass()
class TuckerMatrix(Tucker):
    """Tucker representation for operators.

    :param core: The core tensor of the decomposition.
    :param factors: The list of factors of the decomposition.
    :param n: Dimensionality of source space (see ``from_dense`` method).
    :param m: Dimensionality of target space (see ``from_dense`` method).
    """
    n: Union[Sequence[int], None] = None
    m: Union[Sequence[int], None] = None

    @classmethod
    def from_dense(cls, dense_tensor: back.type(), n: Union[Sequence[int], None] = None, m: Union[Sequence[int], None] = None,
                   eps=1e-14):
        """Convert dense tensor into `TuckerMatrix` representation. The resul is tensor that represents linear operator
        from R^N to R^M.

        :param dense_tensor: Dense tensor that will be factorized.
        :param n: Sequence of ints (n_1, ..., n_a) such that N = n_1 * ... * n_a.
        :param m: Sequence of ints (m_1, ..., m_b) such that M = m_1 * ... * m_b.
        :param eps: Precision of approximation.
        :return: `TuckerMatrix` representation of the provided dense tensor.
        """
        if n is None or m is None:
            raise ValueError("n and m parameter must be specialized for matrices")
        dense_tensor = cls._hosvd(dense_tensor, eps=eps)
        return cls(dense_tensor.core, dense_tensor.factors, n, m)

    def __matmul__(self, other: Union[back.type(), Tucker, "TuckerMatrix"]):
        """Perform matrix multiplication of tensors. If other's ndim > matrix ndim, then perform batch matmul over last
        ndim modes.

        :param other: Can be dense tensor, Tucker or TuckerMatrix. If dense tensor, then treated as vector and matvec
        operation is performed. The result is also a dense tensor. If `Tucker`, then treated as vector and matvec
        operation is performed. The result is also a Tucker. Else matmul operation is performed and another matrix is
        returned.
        :return: Dense tensor, Tucker or TuckerMatrix, depends on type of `other`.
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
            batch_letters = []
            for i in range(len(other.shape) - self.ndim):
                batch_letters.append(ascii_letters[2 * (self.ndim + 1) + i + 1])
            operand_letters = batch_letters + operand_letters
            result_letters =  batch_letters + result_letters
            return back.einsum(f"{core_letters},{','.join(factors_letters)},{''.join(operand_letters)}->{''.join(result_letters)}",
                               self.core, *reshaped_factors, other)
