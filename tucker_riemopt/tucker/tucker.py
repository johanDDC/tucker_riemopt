import numpy as np
import warnings

from typing import Union, Sequence, List
from dataclasses import dataclass, field
from string import ascii_letters
from copy import deepcopy
from scipy.sparse.linalg import LinearOperator, svds

from tucker_riemopt import backend as back
from tucker_riemopt.sparse import SparseTensor

ML_rank = Union[int, Sequence[int]]


@dataclass()
class Tucker:
    core: back.type() = field(default_factory=back.tensor)
    factors: List[back.type()] = field(default_factory=list)
    """
    Tucker tensor factorisation. We represent Tucker tensor as SF-Tucker tensor with no shared factors. One may note,
    that more valid formulation is to represent Tucker as SF-Tucker with one shared factor, but we do not employ
    this approach for the sake of the convenient coding.
    """

    @staticmethod
    def __HOOI(sparse_tensor, sparse_tucker, contraction_dict, maxiter):
        # contraction dict should contain transposed factors
        iteration = 1
        while iteration <= maxiter:
            for k in range(sparse_tensor.ndim):
                r_k = contraction_dict.pop(k).shape[0]
                W = sparse_tensor.contract(contraction_dict)
                W_unfolding = W.unfolding(k)
                if r_k >= min(W_unfolding.shape):
                    # here we have to construct unfolding matrix in dense format, because there is no
                    # method in libraries, which allows to get full svd (at least all left singular vectors)
                    # of sparse matrix. If such method will be found, calculations here gain sufficient boost in
                    # memory
                    W_unfolding = back.tensor(W_unfolding.todense())
                    if r_k > W_unfolding.shape[1]:
                        factor = back.svd(W_unfolding)[0][:, :r_k]
                    else:
                        factor = back.svd(W_unfolding, full_matrices=False)[0]
                else:
                    W_unfolding_linop = LinearOperator(W_unfolding.shape, matvec=lambda x: W_unfolding @ x,
                                                       rmatvec=lambda x: W_unfolding.T @ x)
                    factor = back.tensor(svds(W_unfolding_linop, r_k, return_singular_vectors="u")[0])

                contraction_dict[k] = factor.T
                sparse_tucker.factors[k] = factor

            tucker_core = sparse_tensor.contract(contraction_dict)
            tucker_core = tucker_core.to_dense()
            sparse_tucker = Tucker(tucker_core, sparse_tucker.factors)
            iteration += 1

        return sparse_tucker
    
    @classmethod
    def _hosvd(cls, dense_tensor: back.type(), ml_rank=None, eps=1e-14):
        """
        Converts dense tensor into Tucker representation.
        .. math:: (1): \quad \|A - T_{optimal}\|_F = \eps \|A\|_F

        :param ml_rank: desired ML-rank. If `None`, then parameter `eps` must be provided.
        :param eps: if `ml_rank` is `None`, then `eps` represents precision of approximation as specified at (1).
         If `ml_rank` is not `None`, then ignored.
        :return: Tucker representation of the provided dense tensor.
        """
        d = len(dense_tensor.shape)
        tensor_letters = ascii_letters[:d]

        def truncate_unfolding(u, s, i):
            if ml_rank is None:
                eps_svd = eps / np.sqrt(d) * back.sqrt(s @ s)
                if (eps_svd == float("inf")).any():
                    warnings.warn("You are probably dealing with ill-conditioned tensors. Computations may be numericaly unstable.")
                    s_max = s.max()
                    s_new = s / s_max
                    eps_svd = eps / np.sqrt(d) * s_max * back.sqrt(s_new @ s_new)
                rank = back.cumsum(back.flip(s))
                rank = back.flip(~(rank <= eps_svd))
            else:
                max_rank = ml_rank[i]
                rank = back.tensor([False] * u.shape[1], dtype=bool)
                rank[:max_rank] = True
            return u[:, rank]

        modes = list(np.arange(0, d))
        factors = []
        factor_letters = []
        for i in range(d):
            unfolding = back.transpose(dense_tensor, [modes[i], *(modes[:i] + modes[i + 1:])])
            unfolding = back.reshape(unfolding, (dense_tensor.shape[i], -1), order="F")
            u, s, _ = back.svd(unfolding, full_matrices=False)
            u = truncate_unfolding(u, s, i)
            factors.append(u)
            factor_letters.append(f"{ascii_letters[i]}{ascii_letters[d + i]}")

        core_letters = ascii_letters[d: 2 * d]
        einsum_str = tensor_letters + "," + ",".join(factor_letters) + "->" + core_letters
        core = back.einsum(einsum_str, dense_tensor, *factors)
        return cls(core, factors)

    @classmethod
    def from_dense(cls, dense_tensor: back.type(), eps=1e-14):
        """
        Converts dense tensor into Tucker representation.
        .. math::
            (1): \quad \|A - T_{optimal}\|_F = \eps \|A\|_F

        :param eps: precision of approximation as specified at (1).
        :return: Tucker representation of the provided dense tensor.
        """
        return cls._hosvd(dense_tensor, eps=eps)

    @classmethod
    def sparse2tuck(cls, sparse_tensor: SparseTensor, max_rank: ML_rank = None, maxiter: Union[int, None] = 5):
        """
            Gains sparse tensor and constructs its Sparse Tucker decomposition.

            Parameters
            ----------
            sparse_tensor: SparseTensor
                Tensor in dense format

            max_rank: int, Sequence[int] or None

                - If a number, than defines the maximal `rank` of the result.

                - If a list of numbers, than `max_rank` length should be d
                  (number of dimensions) and `max_rank[i]`
                  defines the (i)-th `rank` of the result.

                  The following two versions are equivalent

                  - ``max_rank = r``

                  - ``max_rank = [r] * d``

            maxiter: int or None

                - If int, than HOOI algorithm will be launched, until provided number of iterations not reached

                - If None, no additional algorithms will be launched (note, that error in that case can be large)
        """
        factors = []
        contraction_dict = dict()
        for i in range(sparse_tensor.ndim):
            unfolding = sparse_tensor.unfolding(i)
            unfolding_linop = LinearOperator(unfolding.shape, matvec=lambda x: unfolding @ x,
                                             rmatvec=lambda x: unfolding.T @ x)
            factors.append(
                back.tensor(svds(unfolding_linop, max_rank[i], solver="propack", return_singular_vectors="u")[0]))
            contraction_dict[i] = factors[-1].T

        core = sparse_tensor.contract(contraction_dict)
        sparse_tucker = cls(core=back.tensor(core.to_dense()), factors=factors)
        if maxiter is not None:
            sparse_tucker = cls.__HOOI(sparse_tensor, sparse_tucker, contraction_dict, maxiter)
        return sparse_tucker

    @property
    def ndim(self) -> int:
        """
        :return: dimensionality of the tensor.
        """
        return len(self.core.shape)

    @property
    def shape(self) -> Sequence[int]:
        """
        :return: sequence represents the shape of the Tucker tensor.
        """
        return [self.factors[i].shape[0] for i in range(self.ndim)]

    @property
    def rank(self) -> Sequence[int]:
        """
        Get ML-rank of the Tucker tensor.

        :return: sequence represents the ML-rank of tensor.
        """
        return self.core.shape

    @property
    def dtype(self) -> type:
        """
        Get dtype of the elements in the Tucker tensor.

        :return: dtype.
        """
        return self.core.dtype

    def __add__(self, other: "Tucker"):
        """
        Add two `Tucker` tensors. The ML-rank of the result is the sum of ML-ranks of the operands.

        :param other: `Tucker` tensor.
        :return: `Tucker` tensor.
        """
        r1 = self.rank
        r2 = other.rank
        padded_core1 = back.pad(self.core, [(0, r2[j]) if j > 0 else (0, 0) for j in range(self.ndim)],
                                mode="constant", constant_values=0)
        padded_core2 = back.pad(other.core, [(r1[j], 0) if j > 0 else (0, 0) for j in range(other.ndim)],
                                mode="constant", constant_values=0)
        core = back.concatenate((padded_core1, padded_core2), axis=0)
        factors = [back.concatenate((self.factors[i], other.factors[i]), axis=1) for i in range(self.ndim)]
        return Tucker(core, factors)

    def __mul__(self, other: "Tucker"):
        """
        Elementwise multiplication of two `Tucker` tensors.

        :param other: `Tucker` tensor.
        :return: `Tucker` tensor.
        """
        core = back.kron(self.core, other.core)
        factors = []
        for i in range(self.ndim):
            contraction = back.einsum('ia,ib->iab', self.factors[i], other.factors[i])
            contraction = back.reshape(contraction, (self.factors[i].shape[0], -1), order="F")
            factors.append(contraction)
        return Tucker(core, factors)

    def __rmul__(self, a: float):
        """
        Elementwise multiplication of `Tucker` tensor by scalar.

        :param a: scalar value.
        :return: `Tucker` tensor.
        """
        return Tucker(a * self.core, self.factors)
        # new_tensor = deepcopy(self)
        # return Tucker(a * new_tensor.core, new_tensor.factors)

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other: "Tucker"):
        other = -other
        return self + other

    def __getitem__(self, key):
        """
            Returns element or a batch of element on positions provided in key parameter.

            Parameters
            ----------
            key : Sequence[Sequence[int]] or  Sequence[Sequence[Sequence[int]]]
                arrays of indices in dense tensor, or batch of indices.
                For instance A[[i], [j], [k]] will return element on (i, j, k) position
                A[[i1, i2], [j1, j2], [k1, k2]] will return 2 elements on positions (i1, j1, k1) and
                (i2, j2, k2) correspondingly.
        """
        if type(key[0]) is int:
            return back.einsum("ijk,i,j,k->", self.core, self.factors[0][key[0]],
                               self.factors[1][key[1]], self.factors[2][key[2]])
        else:
            new_factors = [self.factors[i][key[:, i], :] for i in np.arange(self.ndim)]
            tensor_letters = ascii_letters[:self.ndim]
            einsum_rule = tensor_letters + ',' + ','.join('A' + c for c in tensor_letters) + '->A'

            return back.einsum(einsum_rule, self.core, *new_factors)

    def round(self, max_rank: ML_rank = None, eps=1e-14):
        """
        Perform rounding procedure. The `Tucker` tensor will be approximated by `Tucker` tensor with rank
        at most `max_rank`.
        .. math::
            (1): \quad \|A - T_{optimal}\|_F = \eps \|A\|_F

        :param max_rank: maximum possible `ML-rank` of the approximation. Expects a sequence of integers,
         but if a single number is provided, it will be treated as a sequence with all components equal. If
         `None` provided, will be performed approximation with precision `eps`.
        :param eps: precision of approximation as specified at (1).
        :return: `Tucker` tensor.
        """

        if eps < 0:
            raise ValueError("eps should be greater or equal than 0")

        factors = [None] * self.ndim
        intermediate_factors = [None] * self.ndim
        for i in range(self.ndim):
            factors[i], intermediate_factors[i] = back.qr(self.factors[i])
        intermediate_core = Tucker(self.core, intermediate_factors)
        intermediate_core = self._hosvd(intermediate_core.to_dense(), ml_rank=max_rank, eps=eps)

        if max_rank is None:
            max_rank = intermediate_core.rank
        elif type(max_rank) is int:
            max_rank = [max_rank] * self.ndim

        rank_slices = []
        for i in range(self.ndim):
            rank_slices.append(slice(0, max_rank[i]))
            factors[i] = factors[i] @ intermediate_core.factors[i]
            factors[i] = factors[i][:, :max_rank[i]]
        return Tucker(intermediate_core.core[tuple(rank_slices)], factors)

    def flat_inner(self, other: "Tucker") -> float:
        """
        Calculate inner product of given `Tucker` tensors.

        :param other: `Tucker` tensor.
        :return: result of inner product.
        """
        core_letters = ascii_letters[:self.ndim]
        rev_letters = ascii_letters[self.ndim:][::-1]

        factors_letters = []
        transposed_letters = []
        intermediate_core_letters = []
        for i in range(1, self.ndim + 1):
            factors_letters.append(f"{ascii_letters[self.ndim + i]}{core_letters[i - 1]}")
            transposed_letters.append(f"{ascii_letters[self.ndim + i]}{rev_letters[i]}")
            intermediate_core_letters.append(rev_letters[i])

        source = ",".join([core_letters] + factors_letters + transposed_letters)
        intermediate_core = back.einsum(source + "->" + "".join(intermediate_core_letters),
                                        self.core, *self.factors, *other.factors)
        return (intermediate_core * other.core).sum()

    def k_mode_product(self, k: int, matrix: back.type()):
        """
        k-mode tensor-matrix contraction.

        :param k: from 0 to d-1.
        :param matrix: must contain `self.rank[k]` columns.
        :return: `Tucker` tensor.
        """
        if k < 0 or k >= self.ndim:
            raise ValueError(f"k should be from 0 to {self.ndim - 1}")

        new_factors = self.factors[:k] + [matrix @ self.factors[k]] + self.factors[k + 1:]
        # new_tensor = deepcopy(self)
        # new_tensor.factors[k] = matrix @ new_tensor.factors[k]
        # return new_tensor
        return Tucker(self.core, new_factors)

    def norm(self, qr_based: bool = False) -> float:
        """
        Frobenius norm of `Tucker` tensor.

        :param qr_based: whether to use stable QR-based implementation of norm, which is not differentiable,
            or unstable but differentiable implementation based on inner product. By default, differentiable
            implementation used.
        :return: non-negative number which is the Frobenius norm of `Tucker` tensor.
        """
        if qr_based:
            core_factors = [back.qr(self.factors[i])[1] for i in range(self.ndim)]
            new_tensor = Tucker(self.core, core_factors)
            new_tensor = new_tensor.to_dense()
            return back.norm(new_tensor)

        return back.sqrt(self.flat_inner(self))

    def to_dense(self) -> back.type():
        """
        Convert `Tucker` tensor to dense representation.

        :return: dense d-dimensional representation of `Tucker` tensor.
        """
        core_letters = ascii_letters[:self.ndim]
        factor_letters = [f"{ascii_letters[self.ndim + i]}{ascii_letters[i]}" for i in range(self.ndim)]
        tensor_letters = ascii_letters[self.ndim:2 * self.ndim]
        einsum_str = core_letters + "," + ",".join(factor_letters) + "->" + tensor_letters
        return back.einsum(einsum_str, self.core, *self.factors)

    def __deepcopy__(self, memodict={}):
        new_core = back.copy(self.core)
        new_factors = [back.copy(factor) for factor in self.factors]
        return self.__class__(new_core, new_factors)
