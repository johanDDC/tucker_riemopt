import numpy as np

from typing import List, Union, Sequence
from dataclasses import dataclass, field
from string import ascii_letters
from copy import deepcopy

from tucker_riemopt import backend as back
from tucker_riemopt.tucker.tucker import Tucker


@dataclass()
class SFTucker(Tucker):
    num_shared_factors: int = 0
    shared_factor: Union[back.type(), None] = None

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
        return cls(core, factors[:dt], ds, factors[-1])

    @classmethod
    def from_dense(cls, dense_tensor: back.type(), ds: Union[int, None] = None, eps=1e-14):
        """
        Converts dense tensor into SF-Tucker representation.
        .. math::
            (1): \quad \|A - T_{optimal}\|_F = \eps \|A\|_F

        :param ds: number of shared modes. If `None`, then ds=N, where N is a number of last equal modes.
        :param eps: precision of approximation as specified at (1).
        :return: SF-Tucker representation of the provided dense tensor.
        """
        shape = dense_tensor.shape
        reversed_shape = shape[::-1]
        num_equal_modes = 0
        while num_equal_modes < len(shape) and reversed_shape[num_equal_modes] == shape[-1]:
            num_equal_modes += 1
        if ds is None:
            ds = num_equal_modes
        if ds > num_equal_modes:
            raise ValueError(f"ds cannot be larger then {num_equal_modes}.")


        return cls.__sf_hosvd(dense_tensor, ds, eps=eps)

    @property
    def ndim(self) -> int:
        """
        :return: dimensionality of the tensor
        """
        return len(self.core.shape)

    @property
    def regular_factors(self):
        """
        Alias for `factors`.

        :return: factors
        """
        return self.factors

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
        :return: sequence represents the shape of the SF-Tucker tensor.
        """
        return [self.regular_factors[i].shape[0] for i in range(self.dt)] + [self.shared_factor.shape[0]] * self.ds

    @property
    def rank(self) -> List[int]:
        """
        Get SFT-rank of the SF-Tucker tensor.

        :return: sequence represents the SFT-rank of tensor.
        """
        return list(self.core.shape)[:self.dt] + [self.core.shape[-1]]

    @property
    def dtype(self) -> type:
        """
        Get dtype of the elements in the SF-Tucker tensor.

        :return: dtype
        """
        return self.core.dtype

    def __add__(self, other: "SFTucker"):
        """
        Add two `Tucker` tensors. The ML-rank of the result is the sum of ML-ranks of the operands.

        :param other: `Tucker` tensor.
        :return: `Tucker` tensor.
        :raise: ValueError if amount of shared factors of `self` and `other` doesn't match.
        """
        if self.num_shared_factors != other.num_shared_factors:
            raise ValueError("Amount of shared factors doesn't match. You probably should convert tensors to regular"
                             "Tucker format.")
        r1 = self.rank + [self.core.shape[-1]] * (self.ds - 1)
        r2 = other.rank + [other.core.shape[-1]] * (other.ds - 1)
        padded_core1 = back.pad(self.core, [(0, r2[j]) if j > 0 else (0, 0) for j in range(self.ndim)],
                                mode="constant", constant_values=0)
        padded_core2 = back.pad(other.core, [(r1[j], 0) if j > 0 else (0, 0) for j in range(other.ndim)],
                                mode="constant", constant_values=0)
        core = back.concatenate((padded_core1, padded_core2), axis=0)
        regular_factors = [
            back.concatenate((self.regular_factors[i],
                              other.regular_factors[i]), axis=1) for i in range(self.dt)
        ]
        shared_factor = back.concatenate((self.shared_factor, other.shared_factor), axis=1)
        return SFTucker(core, regular_factors, self.num_shared_factors, shared_factor)

    def __rmul__(self, a: float):
        """
        Elementwise multiplication of `SF-Tucker` tensor by scalar.

        :param a: scalar value.
        :return: `SF-Tucker` tensor.
        """
        # new_tensor = deepcopy(self)
        # return SFTucker(a * new_tensor.core, new_tensor.regular_factors,
        #                 self.num_shared_factors, new_tensor.shared_factor)
        return SFTucker(a * self.core, self.regular_factors, self.num_shared_factors, self.shared_factor)

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other):
        other = -other
        return self + other

    def round(self, max_rank: Union[None, Sequence[int]] = None, eps=1e-14):
        """
                Perform rounding procedure. The `SF-Tucker` tensor will be approximated by `SF-Tucker` tensor with rank
                at most `max_rank`.
                .. math::
                    (1): \quad \|A - T_{optimal}\|_F = \eps \|A\|_F

                :param max_rank: maximum possible `SFT rank` of the approximation. Expects a sequence of integers. If
                 provided a sequence of two elements, then the first element is treated as a value of a multilinear part
                 of rank with all equal components. If `None` provided, will be performed approximation with precision
                 `eps`. For example, if `max_rank=[2, 5]` for three-dimensional tensor, then actual SFT rank is treated
                 as `[2, 2, 5]'.
                :param eps: precision of approximation as specified at (1).
                :return: `Tucker` tensor.
        """
        if eps < 0:
            raise ValueError("eps should be greater or equal than 0")

        factors = [None] * self.dt
        intermediate_factors = [None] * self.dt
        for i in range(self.dt):
            factors[i], intermediate_factors[i] = back.qr(self.regular_factors[i])
        shared_Q, shared_R = back.qr(self.shared_factor)
        intermediate_core = SFTucker(self.core, intermediate_factors, self.num_shared_factors, shared_R).to_dense()
        intermediate_core = self.__sf_hosvd(intermediate_core, self.ds, eps=eps)

        if max_rank is None:
            max_rank = intermediate_core.rank
        elif len(max_rank) == 2:
            max_rank = [max_rank[0]] * self.dt + [max_rank[1]]

        rank_slices = []
        for i in range(self.dt):
            rank_slices.append(slice(0, max_rank[i]))
            factors[i] = factors[i] @ intermediate_core.regular_factors[i]
            factors[i] = factors[i][:, :max_rank[i]]
        rank_slices.extend([slice(0, max_rank[-1])] * self.ds)
        shared_factor = shared_Q @ intermediate_core.shared_factor
        shared_factor = shared_factor[:, :max_rank[-1]]
        return SFTucker(intermediate_core.core[tuple(rank_slices)], factors,
                        self.num_shared_factors, shared_factor)

    def flat_inner(self, other: "SFTucker"):
        """
        Calculate inner product of given `SF-Tucker` tensors.

        :param other: `SF-Tucker` tensor.
        :return: result of inner product.
        """
        if self.num_shared_factors != other.num_shared_factors:
            raise ValueError("Amount of shared factors doesn't match. You probably should convert tensors to regular"
                             "Tucker format.")
        core_letters = ascii_letters[:self.ndim]
        rev_letters = ascii_letters[self.ndim:][::-1]

        factors = []
        factors_letters = []
        transposed_letters = []
        intermediate_core_letters = []
        shared_factor = other.shared_factor.T @ self.shared_factor
        for i in range(1, self.ndim + 1):
            if i <= self.dt:
                factors.append(self.regular_factors[i - 1])
                factors_letters.append(f"{ascii_letters[self.ndim + i]}{core_letters[i - 1]}")
                transposed_letters.append(f"{ascii_letters[self.ndim + i]}{rev_letters[i - 1]}")
                intermediate_core_letters.append(rev_letters[i - 1])
            else:
                factors.append(shared_factor)
                factors_letters.append(ascii_letters[self.ndim + i] + core_letters[i - 1])
                intermediate_core_letters.append(ascii_letters[self.ndim + i])

        source = ",".join([core_letters] + factors_letters + transposed_letters)
        intermediate_core = back.einsum(source + "->" + "".join(intermediate_core_letters),
                                        self.core, *factors, *other.regular_factors)
        return (intermediate_core * other.core).sum()

    def k_mode_product(self, k: int, matrix: back.type()):
        """
        k-mode tensor-matrix contraction.

        :param k: from 0 to d-1.
        :param matrix: must contain `self.rank[k]` columns.
        :return: `SFTucker` tensor.
        """
        if k < 0 or k >= self.ndim:
            raise ValueError(f"k should be from 0 to {self.ndim - 1}")
        if k >= self.dt:
            raise ValueError(f"You are trying to contract by shared mode. Use method `shared_modes_product`.")

        # new_tensor = deepcopy(self)
        # new_tensor.regular_factors[k] = mat @ new_tensor.regular_factors[k]
        # return new_tensor
        regular_factors = self.regular_factors[:k] + [matrix @ self.regular_factors[k]] + self.regular_factors[k + 1:]
        return SFTucker(self.core, regular_factors, self.num_shared_factors, self.shared_factor)

    def shared_modes_product(self, matrix: back.type()):
        """
        Tensor-matrix contraction by shared modes.

        :param matrix: must contain `self.rank[-1]` columns.
        :return: `SF-Tucker` tensor.
        """
        # new_tensor = deepcopy(self)
        # new_tensor.shared_factor = mat @ new_tensor.shared_factor
        # return new_tensor
        new_shared_factor = matrix @ self.shared_factor
        return SFTucker(self.core, self.regular_factors, self.num_shared_factors, new_shared_factor)

    def norm(self, qr_based: bool = False):
        """
            Frobenius norm of `SF-Tucker` tensor.

            :param qr_based: whether to use stable QR-based implementation of norm, which is not differentiable,
                or unstable but differentiable implementation based on inner product. By default, differentiable
                implementation is used.
            :return: non-negative number which is the Frobenius norm of `SF-Tucker` tensor.
        """
        if qr_based:
            common_factors = [back.qr(self.regular_factors[i])[1] for i in range(self.dt)]
            shared_factor = back.qr(self.shared_factor)[1]
            new_tensor = SFTucker(self.core, common_factors,
                                  self.num_shared_factors, shared_factor)
            new_tensor = new_tensor.to_dense()
            return back.norm(new_tensor)

        return back.sqrt(self.flat_inner(self))

    def to_dense(self):
        """
        Convert `SF-Tucker` tensor to dense representation.

        :return: dense d-dimensional representation of `SF-Tucker` tensor.
        """
        core_letters = ascii_letters[:self.ndim]
        factor_letters = [f"{ascii_letters[self.ndim + i]}{ascii_letters[i]}" for i in range(self.ndim)]
        tensor_letters = ascii_letters[self.ndim:2 * self.ndim]
        factors = self.regular_factors + [self.shared_factor] * self.ds
        einsum_str = core_letters + "," + ",".join(factor_letters) + "->" + tensor_letters
        return back.einsum(einsum_str, self.core, *factors)

    def __deepcopy__(self, memodict={}):
        new_core = back.copy(self.core)
        common_factors = [back.copy(factor) for factor in self.regular_factors]
        shared_factor = back.copy(self.shared_factor)
        return self.__class__(new_core, common_factors,
                              self.num_shared_factors, shared_factor)
