from typing import List
import jax.numpy as jnp
import numpy as np
from flax import struct
from string import ascii_letters
from copy import copy


@struct.dataclass
class Tucker:
    core: jnp.array
    factors: List[jnp.array]

    @classmethod
    def full2tuck(cls, T, eps=1e-14):
        """
        Convert full tensor T to Tucker by applying HOSVD algorithm.

        :param T: Tensor in dense format.
        :type T: jnp.array
        :rtype: Tucker
       """
        d = len(T.shape)
        modes = jnp.arange(0, d)
        factors = []
        UT = []
        tensor_letters = ascii_letters[:d]
        factor_letters = ""
        core_letters = ""
        for i, k in enumerate(range(d)):
            unfolding = jnp.transpose(T, [modes[:k + 1], modes[k + 1:]])
            unfolding = jnp.reshape(unfolding, [T.shape[k], -1], order="F")
            u, s, _ = jnp.linalg.svd(unfolding, full_matrices=False)
            # Search for preferable truncation
            eps_svd = eps / jnp.sqrt(d) * jnp.sqrt(s.T @ s)
            cumsum = jnp.cumsum(s[::-1])
            cumsum = (cumsum <= eps_svd)[::-1]
            rank = cumsum.argmin()
            print(cumsum, rank)
            u = u[:, :rank]
            factors.append(u)
            UT.append(u.T)
            factor_letters += f"{ascii_letters[d + i]}{ascii_letters[i]},"
            core_letters += ascii_letters[d + i]

        einsum_str = tensor_letters + factor_letters[:-1] + "->" + core_letters
        core = jnp.einsum(einsum_str, T, *UT)
        return cls(core, factors)

    @property
    def shape(self):
        """
        Get the tuple representing the shape of Tucker.

        :return: Tucker shape
        :rtype: tuple
        """
        return tuple([factor.shape[1] for factor in self.factors])

    @property
    def ml_rank(self):
        """
        Get ranks of the Tucker in amount of ``ndim``.

        :return: ranks
        :rtype: tuple
        """
        return self.core.shape

    @property
    def ndim(self):
        """
        Get the number of dimensions of the Tucker.

        :return: dimensions number
        :rtype: int
        """
        return len(self.core.shape)

    @property
    def dtype(self):
        """
        Represents the dtype of elements in Tucker.

        :return: dtype of elements
        :rtype: dtype
        """
        return self.core.dtype

    def __add__(self, other):
        """
        Add two `Tucker` tensors and double rank.

        :return: Sum of tensors
        :rtype: `Tucker`
        """
        factors = []
        core = jnp.zeros(jnp.array(self.ml_rank) + jnp.array(other.ml_rank), dtype=self.dtype)
        sub_core_slice1 = []
        sub_core_slice2 = []
        for i in range(self.ndim):
            sub_core_slice1.append(slice(None, self.ml_rank[i]))
            sub_core_slice2.append(slice(self.ml_rank[i], None))
            factors.append(jnp.concatenate((self.factors[i], other.factors[i]), axis=1))

        core[sub_core_slice1] = self.core
        core[sub_core_slice2] = other.core
        return Tucker(core, factors)

    def __mul__(self, other):
        """
        Elementwise multiplication of two `Tucker` tensors.

        :return: Elementwise multiplication of two tensors.
        :rtype: `Tucker`
        """
        core = jnp.kron(self.core, other.core)
        factors = []
        for i in range(self.ndim):
            factors.append(jnp.einsum('ia,ib->iab', self.factors[i], other.factors[i])\
                           .reshape(self.factors[i].shape[0], -1))

        return Tucker(core, factors)

    def __rmul__(self, a):
        """
        Elementwise multiplication of `Tucker` tensor by scalar.

        :return: Elementwise multiplication of tensor by scalar.
        :rtype: `Tucker`
        """
        new_tensor = copy(self)
        new_tensor.core = a * new_tensor.core
        return new_tensor

    def __neg__(self):
        new_tensor = copy(self)
        return (-1) * new_tensor

    def __sub__(self, other):
        other = -other
        return self + other

    def round(self, eps=1e-14):
        """
        HOSVD rounding procedure, returns a `Tucker` with smaller `ml-ranks`.

        :return: `Tucker` with reduced ranks.
        :rtype: `Tucker`
        """
        factors = [None] * self.ndim
        intermediate_factors = [None] * self.ndim
        for i in range(self.ndim):
            factors[i], intermediate_factors[i] = jnp.linalg.qr(self.factors[i])

        intermediate_tensor = Tucker(self.core, intermediate_factors)
        intermediate_tensor = Tucker.full2tuck(intermediate_tensor.full(), eps)
        core = intermediate_tensor.core
        for i in range(self.ndim):
            factors[i] = factors[i] @ intermediate_tensor.factors[i]
        return Tucker(core, factors)

    def flat_inner(self, other):
        """
        Calculate inner product of given `Tucker` tensors.

        :rerurn: the result of inner product
        :rtype: float
        """
        new_tensor = copy(self)
        for i in range(self.ndim):
            new_tensor.factors[i] = other.factors[i].T @ new_tensor.factors[i]

        inds = ascii_letters[:self.ndim]
        return jnp.squeeze(jnp.einsum(f"{inds},{inds}->", new_tensor.full(), other.core))

    def k_mode_product(self, k, mat):
        new_tensor = copy(self)
        new_tensor.factors[k] = mat @ new_tensor.factors[k]
        return new_tensor

    def norm(self):
        core_factors = []
        for i in range(self.ndim):
            core_factors.append(jnp.linalg.qr(self.factors[i])[1])

        new_tensor = Tucker(self.core, core_factors)
        new_tensor = new_tensor.full()
        return np.linalg.norm(new_tensor)

    def full(self):
        core_letters = ascii_letters[:self.ndim]
        factor_letters = ""
        tensor_letters = ""
        for i in range(self.ndim):
            factor_letters += f"{ascii_letters[self.ndim + i]}{ascii_letters[i]},"
            tensor_letters += ascii_letters[self.ndim + i]

        einsum_str = core_letters + factor_letters[:-1] + "->" + tensor_letters
        return jnp.einsum(einsum_str, self.core, *self.factors)
