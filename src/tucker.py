from typing import List, Union, Sequence
import numpy as np
from flax import struct
from string import ascii_letters
from copy import copy
from src import backend as back

ML_rank = Union[int, Sequence[int]]

@struct.dataclass
class Tucker:
    core: back.type()
    factors: List[back.type()]

    @classmethod
    def full2tuck(cls, T : back.type(), max_rank: ML_rank=None, eps=1e-14):
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
        def rank_trucation(unfolding_svd, factor_id):
            u, s, _ = unfolding_svd
            rank = max_rank if type(max_rank) is int else max_rank[factor_id]
            return u[:, :rank]

        def eps_trunctation(unfolding_svd):
            u, s, _ = unfolding_svd
            eps_svd = eps / np.sqrt(d) * np.sqrt(s.T @ s)
            cumsum = np.cumsum(list(reversed(s)))
            cumsum = (cumsum <= back.to_numpy(eps_svd))
            rank = len(s) - cumsum.argmin()
            return u[:, :rank]

        if eps < 0:
            raise ValueError("eps should be greater or equal than 0")
        d = len(T.shape)
        modes = list(np.arange(0, d))
        factors = []
        UT = []
        tensor_letters = ascii_letters[:d]
        factor_letters = ""
        core_letters = ""
        for i, k in enumerate(range(d)):
            unfolding = back.transpose(T, [modes[k], *(modes[:k] + modes[k+1:])])
            unfolding = back.reshape(unfolding, (T.shape[k], -1), order="F")
            unfolding_svd = back.svd(unfolding, full_matrices=False)
            u = rank_trucation(unfolding_svd, i) if max_rank is not None else eps_trunctation(unfolding_svd)
            factors.append(u)
            UT.append(u.T)
            factor_letters += f"{ascii_letters[d + i]}{ascii_letters[i]},"
            core_letters += ascii_letters[d + i]

        einsum_str = tensor_letters + "," + factor_letters[:-1] + "->" + core_letters
        core = back.einsum(einsum_str, T, *UT)
        return cls(core, factors)

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
        padded_core1 = back.pad(self.core, [(0, r2[j]) if j > 0 else (0, 0) for j in range(self.ndim)], 0)
        padded_core2 = back.pad(other.core, [(r1[j], 0) if j > 0 else (0, 0) for j in range(other.ndim)], 0)
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
            factors.append(back.einsum('ia,ib->iab', self.factors[i], other.factors[i])\
                           .reshape(self.factors[i].shape[0], -1))

        return Tucker(core, factors)

    def __rmul__(self, a):
        """
            Elementwise multiplication of `Tucker` tensor by scalar.
        """
        new_tensor = copy(self)
        return Tucker(a * new_tensor.core, new_tensor.factors)

    def __neg__(self):
        new_tensor = copy(self)
        return (-1) * new_tensor

    def __sub__(self, other):
        other = -other
        return self + other

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
        intermediate_tensor = Tucker.full2tuck(intermediate_tensor.full(), max_rank, eps)
        core = intermediate_tensor.core
        for i in range(self.ndim):
            factors[i] = factors[i] @ intermediate_tensor.factors[i]
        return Tucker(core, factors)

    def flat_inner(self, other):
        """
            Calculate inner product of given `Tucker` tensors.
        """
        new_tensor = copy(self)
        for i in range(self.ndim):
            new_tensor.factors[i] = other.factors[i].T @ new_tensor.factors[i]

        inds = ascii_letters[:self.ndim]
        return back.squeeze(back.einsum(f"{inds},{inds}->", new_tensor.full(), other.core))

    def k_mode_product(self, k:int, mat: back.type()):
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
        new_tensor = copy(self)
        new_tensor.factors[k] = mat @ new_tensor.factors[k]
        return new_tensor

    def norm(self, qr_based:bool =False):
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
            return np.linalg.norm(new_tensor)

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

TangentVector = Tucker