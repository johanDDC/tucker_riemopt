import numpy as np

from typing import List, Union, Sequence, Dict
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
    def full2tuck(cls, T: back.type(), eps=1e-14):
        """
            Convert full tensor T to Tucker by applying HOSVD algorithm.

            Parameters
            ----------
            T: backend tensor type
                Tensor in dense format

            max_rank: int, Sequence[int] or None

                - If a number, than defines the maximal `rank` of the result.

                - If a list of numbers, than `max_rank` length should be d
                  (number of dimensions) and `max_rank[i]`
                  defines the (i)-th `rank` of the result.

                  The following two versions are equivalent

                  - ``max_rank = r``

                  - ``max_rank = [r] * d``

            eps: float

                - If `max_rank` is not provided, then constructed tensor would be guarantied to be `epsilon`-close to `T`
                  in terms of relative Frobenius error:

                    `||T - A_tucker||_F / ||T||_F <= eps`

                - If `max_rank` is provided, than this parameter is ignored
        """
        if eps < 0:
            raise ValueError("eps should be greater or equal than 0")
        d = len(T.shape)

        def eps_trunctation(unfolding_svd):
            u, s, _ = unfolding_svd
            eps_svd = eps / np.sqrt(d) * back.sqrt(s @ s)
            cumsum = back.cumsum(back.flip(s))
            cumsum = back.flip(~(cumsum <= eps_svd))
            return u[:, cumsum]

        modes = list(np.arange(0, d))
        factors = []
        tensor_letters = ascii_letters[:d]
        factor_letters = ""
        core_letters = ""
        for i in range(d):
            unfolding = back.transpose(T, [modes[i], *(modes[:i] + modes[i + 1:])])
            unfolding = back.reshape(unfolding, (T.shape[i], -1), order="F")
            unfolding_svd = back.svd(unfolding, full_matrices=False)
            u = eps_trunctation(unfolding_svd)
            factors.append(u)
            factor_letters += f"{ascii_letters[i]}{ascii_letters[d + i]},"
            core_letters += ascii_letters[d + i]

        einsum_str = tensor_letters + "," + factor_letters[:-1] + "->" + core_letters
        core = back.einsum(einsum_str, T, *factors)
        return cls(core, factors)

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
    def shape(self):
        """
            Get the tuple representing the shape of Tucker tensor.
        """
        return tuple([factor.shape[0] for factor in self.factors])

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
        factors = []
        r1 = self.rank
        r2 = other.rank
        padded_core1 = back.pad(self.core, [(0, r2[j]) if j > 0 else (0, 0) for j in range(self.ndim)],
                                mode="constant", constant_values=0)
        padded_core2 = back.pad(other.core, [(r1[j], 0) if j > 0 else (0, 0) for j in range(other.ndim)],
                                mode="constant", constant_values=0)
        core = back.concatenate((padded_core1, padded_core2), axis=0)
        for i in range(self.ndim):
            factors.append(back.concatenate((self.factors[i], other.factors[i]), axis=1))

        return Tucker(core, factors)

    def __mul__(self, other):
        """
            Elementwise multiplication of two `Tucker` tensors.
        """
        core = back.kron(self.core, other.core)
        factors = []
        for i in range(self.ndim):
            factors.append(back.einsum('ia,ib->iab', self.factors[i], other.factors[i]) \
                           .reshape(self.factors[i].shape[0], -1))

        return Tucker(core, factors)

    def __rmul__(self, a):
        """
            Elementwise multiplication of `Tucker` tensor by scalar.
        """
        new_tensor = deepcopy(self)
        return Tucker(a * new_tensor.core, new_tensor.factors)

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other):
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
        HOSVD rounding procedure, returns a Tucker with smaller ranks.

        Parameters
        ----------
            max_rank: int, Sequence[int] or None

                - If a number, than defines the maximal `rank` of the result.

                - If a list of numbers, than `max_rank` length should be d
                  (number of dimensions) and `max_rank[i]`
                  defines the (i)-th `rank` of the result.

                  The following two versions are equivalent

                  - ``max_rank = r``

                  - ``max_rank = [r] * d``

            eps: float

                - If `max_rank` is not provided, then result would be guarantied to be `epsilon`-close to unrounded tensor
                  in terms of relative Frobenius error:

                    `||A - A_round||_F / ||A||_F <= eps`

                - If `max_rank` is provided, than this parameter is ignored
        """

        if eps < 0:
            raise ValueError("eps should be greater or equal than 0")
        factors = [None] * self.ndim
        intermediate_factors = [None] * self.ndim
        for i in range(self.ndim):
            factors[i], intermediate_factors[i] = back.qr(self.factors[i])

        intermediate_tensor = Tucker(self.core, intermediate_factors)
        intermediate_tensor = Tucker.full2tuck(intermediate_tensor.full(), eps)
        if max_rank is None:
            max_rank = intermediate_tensor.rank
        elif type(max_rank) is int:
            max_rank = [max_rank] * self.ndim
        core = intermediate_tensor.core
        rank_slices = []
        for i in range(self.ndim):
            rank_slices.append(slice(0, max_rank[i]))
            factors[i] = factors[i] @ intermediate_tensor.factors[i]
            factors[i] = factors[i][:, :max_rank[i]]
        return Tucker(core[tuple(rank_slices)], factors)

    def flat_inner(self, other):
        """
            Calculate inner product of given `Tucker` tensors.
        """
        factors = []
        transposed_factors = []
        core_letters = ascii_letters[:self.ndim]
        factors_letters = []
        transposed_letters = []
        intermediate_core_letters = []
        for i in range(self.ndim):
            factors.append(self.factors[i])
            factors_letters.append(ascii_letters[self.ndim + i] + core_letters[i])
            transposed_factors.append(other.factors[i].T)
            transposed_letters.append(ascii_letters[self.ndim + 2 * i] + ascii_letters[self.ndim + i])
            intermediate_core_letters.append(ascii_letters[self.ndim + 2 * i])

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
        new_tensor = deepcopy(self)
        new_tensor.factors[k] = mat @ new_tensor.factors[k]
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
            core_factors = []
            for i in range(self.ndim):
                core_factors.append(back.qr(self.factors[i])[1])

            new_tensor = Tucker(self.core, core_factors)
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
        for i in range(self.ndim):
            factor_letters += f"{ascii_letters[self.ndim + i]}{ascii_letters[i]},"
            tensor_letters += ascii_letters[self.ndim + i]

        einsum_str = core_letters + "," + factor_letters[:-1] + "->" + tensor_letters

        return back.einsum(einsum_str, self.core, *self.factors)

    def __deepcopy__(self, memodict={}):
        new_core = back.copy(self.core)
        new_factors = [back.copy(factor) for factor in self.factors]
        return self.__class__(new_core, new_factors)


TangentVector = Tucker
