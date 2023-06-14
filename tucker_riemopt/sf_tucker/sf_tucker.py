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
class SFTucker:
    core: back.type() = field(default_factory=back.tensor)
    regular_factors: List[back.type()] = field(default_factory=list)
    num_shared_factors: int = 0
    shared_factor: back.type() = field(default_factory=back.tensor)

    @classmethod
    def __sf_hosvd(cls, dense_tensor: back.type(), ds: int, sft_rank=None, eps=1e-14):
        """
        Converts dense tensor into SF-Tucker representation.
        .. math:: (1): \quad \|A - T_{optimal}\|_F = \eps \|A\|_F

        :param ds: amount of shared factors.
        :param sft_rank: desired SFT-rank. If `None`, then parameter `eps` must be provided.
        :param eps: if `sft_rank` is `None`, then `eps` represents precision of approximation as specified at (1).
         If `sft_rank` is not `None`, then ignored.
        :return: SF-Tucker representation of the provided dense tensor.
        """
        d = len(dense_tensor.shape)
        dt = d - ds
        tensor_letters = ascii_letters[:d]

        def truncate_unfolding(u, s, i):
            if sft_rank is None:
                eps_svd = eps / np.sqrt(d) * back.sqrt(s @ s)
                rank = back.cumsum(back.flip(s))
                rank = back.flip(~(rank <= eps_svd))
            else:
                rank = sft_rank[i]
            return u[:, rank]

        modes = list(np.arange(0, d))
        factors = []
        factor_letters = []
        for i in range(dt):
            unfolding = back.transpose(dense_tensor, [modes[i], *(modes[:i] + modes[i + 1:])])
            unfolding = back.reshape(unfolding, (dense_tensor.shape[i], -1), order="F")
            u, s, _ = back.svd(unfolding, full_matrices=False)
            u = truncate_unfolding(u, s, i)
            factors.append(u)
            factor_letters.append(f"{ascii_letters[i]}{ascii_letters[d + i]}")

        core_letters = ascii_letters[d: 2 * d]
        core_concat = None
        for i in range(ds):
            j = dt + i
            unfolding = back.transpose(dense_tensor, [modes[j], *(modes[:j] + modes[j + 1:])])
            unfolding = back.reshape(unfolding, (dense_tensor.shape[j], -1), order="F")
            core_concat = back.concatenate([core_concat, unfolding], axis=1) if core_concat is not None else unfolding
            factor_letters.append(f"{ascii_letters[dt + i]}{ascii_letters[d + dt + i]}")
        u, s, _ = back.svd(core_concat, full_matrices=False)
        u = truncate_unfolding(u, s, -1)
        factors.extend([u] * ds)

        einsum_str = tensor_letters + "," + ",".join(factor_letters) + "->" + core_letters
        core = back.einsum(einsum_str, dense_tensor, *factors)
        return cls(core, factors, ds, factors[-1])

    @classmethod
    def from_dense(cls, dense_tensor: back.type(), eps=1e-14):
        """
        Converts dense tensor into SF-Tucker representation.
        .. math::
            (1): \quad \|A - T_{optimal}\|_F = \eps \|A\|_F

        :param eps: if `sft_rank` is `None`, then `eps` represents precision of approximation as specified at (1).
         If `sft_rank` is not `None`, then ignored.
        :return: SF-Tucker representation of the provided dense tensor.
        """
        shape = dense_tensor.shape
        reversed_shape = shape[::-1]
        num_equal_modes = 0
        while reversed_shape[num_equal_modes] != shape[-1]:
            num_equal_modes += 1

        return cls.__sf_hosvd(dense_tensor, num_equal_modes, eps=eps)

    @property
    def ndim(self) -> int:
        """
        :return: dimensionality of tensor
        """
        return len(self.core.shape)

    @property
    def dt(self) -> int:
        """
        :return: amount of modes with regular factors
        """
        return len(self.regular_factors)

    @property
    def ds(self) -> int:
        """
        :return: amount of modes with shared factors
        """
        return self.ndim - self.dt

    @property
    def shape(self) -> Sequence[int]:
        """
        :return: sequence represents the shape of SF-Tucker tensor.
        """
        return [self.regular_factors[i].shape[0] for i in range(self.dt)] + [self.shared_factor.shape[0]] * self.ds

    @property
    def rank(self) -> Sequence[int]:
        """
        Get SFT-rank of the SF-Tucker tensor.

        :return: sequence represents the SFT-rank of tensor.
        """
        return self.core.shape

    @property
    def dtype(self) -> type:
        """
        Get dtype of the elements in SF-Tucker tensor.

        :return: dtype
        """
        return self.core.dtype

    def __add__(self, other):
        """
            Add two `Tucker` tensors. Result rank is doubled.
        """
        if self.num_shared_factors != other.num_shared_factors:
            return self.to_regular_tucker() + other.to_regular_tucker()
        r1 = self.rank
        r2 = other.rank
        padded_core1 = back.pad(self.core, [(0, r2[j]) if j > 0 else (0, 0) for j in range(self.ndim)],
                                mode="constant", constant_values=0)
        padded_core2 = back.pad(other.core, [(r1[j], 0) if j > 0 else (0, 0) for j in range(other.ndim)],
                                mode="constant", constant_values=0)
        core = back.concatenate((padded_core1, padded_core2), axis=0)
        common_factors = []
        for i in range(len(self.regular_factors)):
            common_factors.append(back.concatenate((self.regular_factors[i],
                                                    other.regular_factors[i]), axis=1))
        symmetric_factor = back.concatenate((self.shared_factor,
                                             other.shared_factor), axis=1)
        return SFTucker(core=core, regular_factors=common_factors,
                        num_shared_factors=self.num_shared_factors, shared_factor=symmetric_factor)

    def __rmul__(self, a):
        """
            Elementwise multiplication of `Tucker` tensor by scalar.
        """
        new_tensor = deepcopy(self)
        return SFTucker(a * new_tensor.core, new_tensor.regular_factors,
                        self.num_shared_factors, new_tensor.shared_factor)

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other):
        other = -other
        return self + other

    def round(self, max_rank: ML_rank = None, eps=1e-14):
        if eps < 0:
            raise ValueError("eps should be greater or equal than 0")
        if max_rank is None:
            max_rank = self.rank
        elif type(max_rank) is int:
            max_rank = [max_rank] * self.ndim

        factors = [None] * (self.ndim - self.num_shared_factors + 1)
        intermediate_factors = [None] * (self.ndim - self.num_shared_factors + 1)
        for i in range(self.ndim - self.num_shared_factors):
            factors[i], intermediate_factors[i] = back.qr(self.regular_factors[i])
        factors[-1], intermediate_factors[-1] = back.qr(self.shared_factor)

        intermediate_tensor = SFTucker(self.core, intermediate_factors[:-1],
                                       self.num_shared_factors, intermediate_factors[-1]).full()
        modes = list(np.arange(0, self.ndim))
        for i in range(self.ndim - self.num_shared_factors):
            unfolding = back.transpose(intermediate_tensor, [modes[i], *(modes[:i] + modes[i + 1:])])
            unfolding = back.reshape(unfolding, (intermediate_tensor.shape[i], -1), order="F")
            u, _, _ = back.svd(unfolding, full_matrices=False)
            u = u[:, :max_rank[i]]
            factors[i] @= u
        core_concat = None
        for i in range(self.num_shared_factors):
            j = len(self.regular_factors) + i
            unfolding = back.transpose(intermediate_tensor, [modes[j], *(modes[:j] + modes[j + 1:])])
            unfolding = back.reshape(unfolding, (intermediate_tensor.shape[j], -1), order="F")
            if core_concat is None:
                core_concat = unfolding
            else:
                core_concat = back.concatenate([core_concat, unfolding], axis=1)
        u, _, _ = back.svd(core_concat, full_matrices=False)
        u = u[:, :max_rank[-1]]
        factors[-1] @= u
        core = self
        for i in range(self.ndim - self.num_shared_factors):
            core = core.k_mode_product(i, factors[i].T)
        core = core.symmetric_modes_product(factors[-1].T).full()
        return SFTucker(core, factors[:-1], self.num_shared_factors, factors[-1])

    def flat_inner(self, other):
        """
            Calculate inner product of given `Tucker` tensors.
        """
        if self.num_shared_factors != other.num_shared_factors:
            return self.to_regular_tucker().flat_inner(other.to_regular_tucker())
        factors = []
        transposed_factors = []
        core_letters = ascii_letters[:self.ndim]
        factors_letters = []
        transposed_letters = []
        intermediate_core_letters = []
        symmetric_factor = other.shared_factor.T @ self.shared_factor
        rev_letters = ascii_letters[self.ndim:][::-1]
        for i in range(1, self.ndim + 1):
            j = i - 1
            if j < self.ndim - self.num_shared_factors:
                factors.append(self.regular_factors[j])
                factors_letters.append(ascii_letters[self.ndim + i] + core_letters[j])
                transposed_factors.append(other.regular_factors[j].T)
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
        if k >= self.ndim - self.num_shared_factors:
            return self.to_regular_tucker().k_mode_product(k, mat)
        new_tensor = deepcopy(self)
        new_tensor.regular_factors[k] = mat @ new_tensor.regular_factors[k]
        return new_tensor

    def symmetric_modes_product(self, mat: back.type()):
        """
            Tensor-matrix product for all sf_tucker modes
        """
        new_tensor = deepcopy(self)
        new_tensor.shared_factor = mat @ new_tensor.shared_factor
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
            for i in range(len(self.regular_factors)):
                common_factors.append(back.qr(self.regular_factors[i])[1])
            symmetric_factor = back.qr(self.shared_factor)[1]
            new_tensor = SFTucker(self.core, common_factors,
                                  self.num_shared_factors, symmetric_factor)
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
            if i < self.ndim - self.num_shared_factors:
                factor_letters += f"{ascii_letters[self.ndim + i]}{ascii_letters[i]},"
                tensor_letters += ascii_letters[self.ndim + i]
                factors.append(self.regular_factors[i])
            else:
                factors.append(self.shared_factor)
                factor_letters += f"{ascii_letters[self.ndim + i]}{ascii_letters[i]},"
                tensor_letters += ascii_letters[self.ndim + i]
        einsum_str = core_letters + "," + factor_letters[:-1] + "->" + tensor_letters
        return back.einsum(einsum_str, self.core, *factors)

    def to_regular_tucker(self):
        factors = []
        for i in range(self.ndim - self.num_shared_factors):
            factors.append(self.regular_factors[i])
        factors += [self.shared_factor for _ in range(self.num_shared_factors)]
        return RegularTucker(core=self.core, factors=factors)

    def __deepcopy__(self, memodict={}):
        new_core = back.copy(self.core)
        common_factors = [back.copy(factor) for factor in self.regular_factors]
        symmetic_factor = back.copy(self.shared_factor)
        return self.__class__(new_core, common_factors,
                              self.num_shared_factors, symmetic_factor)
