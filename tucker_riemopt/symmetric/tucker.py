import numpy as np

from typing import List, Union, Sequence, Dict
from dataclasses import dataclass, field
from string import ascii_letters
from copy import deepcopy

from scipy.sparse.linalg import LinearOperator, svds

from tucker_riemopt import backend as back, Tucker as RegularTucker

try:
    import torch
except ImportError as error:
    message = ("Impossible to import PyTorch.\n"
               "To use tucker_riemopt with the PyTorch backend, "
               "you must first install PyTorch!")
    raise ImportError(message) from error

ML_rank = Union[int, Sequence[int]]


@dataclass()
class Tucker:
    core: back.type() = field(default_factory=back.tensor)
    common_factors: List[back.type()] = field(default_factory=list)
    num_symmetric_modes: int = 0
    symmetric_factor: back.type() = field(default_factory=back.tensor)

    @classmethod
    def full2tuck(cls, T: back.type(), eps=1e-14):
        tucker = RegularTucker.full2tuck(T, eps)
        return cls(core=tucker.core, common_factors=tucker.factors,
                   symmetric_modes=[], symmetric_factor=None)

    @property
    def shape(self):
        """
            Get the tuple representing the shape of Tucker tensor.
        """
        shapes = []
        for i in range(self.ndim - self.num_symmetric_modes):
            shapes.append(self.common_factors[i].shape[0])
        shapes += [self.symmetric_factor.shape[0] for _ in range(self.num_symmetric_modes)]
        return tuple(shapes)

    @property
    def rank(self):
        """
            Get multilinear rank of the Tucker tensor.

            Returns
            -------
            rank: int or Sequence[int]
                tuple, represents multilinear rank of tensor
        """
        return self.core.shape

    @property
    def ndim(self):
        """
            Get the number of dimensions of the Tucker tensor.
        """
        return len(self.core.shape)

    @property
    def dtype(self):
        """
            Get dtype of the elements in Tucker tensor.
        """
        return self.core.dtype

    def __add__(self, other):
        """
            Add two `Tucker` tensors. Result rank is doubled.
        """
        if self.num_symmetric_modes != other.num_symmetric_modes:
            return self.to_regular_tucker() + other.to_regular_tucker()
        r1 = self.rank
        r2 = other.rank
        padded_core1 = back.pad(self.core, [(0, r2[j]) if j > 0 else (0, 0) for j in range(self.ndim)],
                                mode="constant", constant_values=0)
        padded_core2 = back.pad(other.core, [(r1[j], 0) if j > 0 else (0, 0) for j in range(other.ndim)],
                                mode="constant", constant_values=0)
        core = back.concatenate((padded_core1, padded_core2), axis=0)
        common_factors = []
        for i in range(len(self.common_factors)):
            common_factors.append(back.concatenate((self.common_factors[i],
                                                    other.common_factors[i]), axis=1))
        symmetric_factor = back.concatenate((self.symmetric_factor,
                                                other.symmetric_factor), axis=1)
        return Tucker(core=core, common_factors=common_factors,
                      num_symmetric_modes=self.num_symmetric_modes, symmetric_factor=symmetric_factor)

    def __rmul__(self, a):
        """
            Elementwise multiplication of `Tucker` tensor by scalar.
        """
        new_tensor = deepcopy(self)
        return Tucker(a * new_tensor.core, new_tensor.common_factors,
                      self.num_symmetric_modes, new_tensor.symmetric_factor)

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other):
        other = -other
        return self + other

    def flat_inner(self, other):
        """
            Calculate inner product of given `Tucker` tensors.
        """
        if self.num_symmetric_modes != other.num_symmetric_modes:
            return self.to_regular_tucker().flat_inner(other.to_regular_tucker())
        factors = []
        transposed_factors = []
        core_letters = ascii_letters[:self.ndim]
        factors_letters = []
        transposed_letters = []
        intermediate_core_letters = []
        symmetric_factor = other.symmetric_factor.T @ self.symmetric_factor
        rev_letters = ascii_letters[self.ndim:][::-1]
        for i in range(1, self.ndim + 1):
            j = i - 1
            if j < self.ndim - self.num_symmetric_modes:
                factors.append(self.common_factors[j])
                factors_letters.append(ascii_letters[self.ndim + i] + core_letters[j])
                transposed_factors.append(other.common_factors[j].T)
                transposed_letters.append(rev_letters[j] + ascii_letters[self.ndim + i])
                intermediate_core_letters.append(rev_letters[j])
            else:
                factors.append(symmetric_factor)
                factors_letters.append(ascii_letters[self.ndim + i] + core_letters[j])
                intermediate_core_letters.append(ascii_letters[self.ndim + i])

        source = ",".join([core_letters] + factors_letters + transposed_letters)
        intermediate_core = back.einsum(source + "->" + "".join(intermediate_core_letters),
                                        self.core, *factors, *transposed_factors)
        return (intermediate_core * other.core).sum()

    def k_mode_product(self, k: int, mat: back.type()):
        """
        K-mode tensor-matrix product.

        Parameters
        ----------
        k: int
            mode id from 0 to ndim - 1
        mat: matrix of backend tensor type
            matrix with which Tucker tensor is contracted by k mode
        """
        if k < 0 or k >= self.ndim:
            raise ValueError(f"k shoduld be from 0 to {self.ndim - 1}")
        if k >= self.ndim - self.num_symmetric_modes:
            return self.to_regular_tucker().k_mode_product(k, mat)
        new_tensor = deepcopy(self)
        new_tensor.common_factors[k] = mat @ new_tensor.common_factors[k]
        return new_tensor

    def symmetric_modes_product(self, mat: back.type()):
        """
            Tensor-matrix product for all symmetric modes
        """
        new_tensor = deepcopy(self)
        new_tensor.symmetric_factor = mat @ new_tensor.symmetric_factor
        return new_tensor

    def norm(self, qr_based: bool = False):
        """
        Frobenius norm of `Tucker`.

        Parameters
        ----------
        qr_based: bool
            whether to use stable QR-based implementation of norm, which is not differentiable,
            or unstable but differentiable implementation based on inner product. By default differentiable implementation
            is used
        Returns
        -------
        F-norm: float
            non-negative number which is the Frobenius norm of `Tucker` tensor
        """
        if qr_based:
            common_factors = []
            for i in range(len(self.common_factors)):
                common_factors.append(back.qr(self.common_factors[i])[1])
            symmetric_factor = back.qr(self.symmetric_factor)[1]
            new_tensor = Tucker(self.core, common_factors,
                                self.num_symmetric_modes, symmetric_factor)
            new_tensor = new_tensor.full()
            return back.norm(new_tensor)

        return back.sqrt(self.flat_inner(self))

    def full(self):
        """
            Dense representation of `Tucker`.
        """
        core_letters = ascii_letters[:self.ndim]
        factor_letters = ""
        tensor_letters = ""
        factors = []
        curr_common_factor = 0
        for i in range(self.ndim):
            if i < self.ndim - self.num_symmetric_modes:
                factor_letters += f"{ascii_letters[self.ndim + i]}{ascii_letters[i]},"
                tensor_letters += ascii_letters[self.ndim + i]
                factors.append(self.common_factors[i])
            else:
                factors.append(self.symmetric_factor)
                factor_letters += f"{ascii_letters[self.ndim + i]}{ascii_letters[i]},"
                tensor_letters += ascii_letters[self.ndim + i]
        einsum_str = core_letters + "," + factor_letters[:-1] + "->" + tensor_letters
        return back.einsum(einsum_str, self.core, *factors)

    def to_regular_tucker(self):
        factors = []
        for i in range(self.ndim - self.num_symmetric_modes):
            factors.append(self.common_factors[i])
        factors += [self.symmetric_factor for _ in range(self.num_symmetric_modes)]
        return RegularTucker(core=self.core, factors=factors)

    def __deepcopy__(self, memodict={}):
        new_core = back.copy(self.core)
        common_factors = [back.copy(factor) for factor in self.common_factors]
        symmetic_factor = back.copy(self.symmetric_factor)
        return self.__class__(new_core, common_factors,
                              self.num_symmetric_modes, symmetic_factor)


TangentVector = Tucker
