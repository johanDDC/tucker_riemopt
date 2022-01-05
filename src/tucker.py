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
    def full2tuck(cls, T : jnp.array, eps=1e-14):
        """
        Convert full tensor T to Tucker by applying HOSVD algorithm.

        :param T: Tensor in dense format.
        :type T: jnp.array
        :rtype: Tucker
        :raises [ValueError]: if `eps` < 0
       """
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
            unfolding = jnp.transpose(T, [modes[k], *(modes[:k] + modes[k+1:])])
            unfolding = jnp.reshape(unfolding, (T.shape[k], -1), order="F")
            u, s, _ = jnp.linalg.svd(unfolding, full_matrices=False)
            # Search for preferable truncation
            eps_svd = eps / jnp.sqrt(d) * jnp.sqrt(s.T @ s)
            cumsum = jnp.cumsum(s[::-1])
            cumsum = (cumsum <= eps_svd)
            rank = len(s) - cumsum.argmin()
            u = u[:, :rank]
            factors.append(u)
            UT.append(u.T)
            factor_letters += f"{ascii_letters[d + i]}{ascii_letters[i]},"
            core_letters += ascii_letters[d + i]

        einsum_str = tensor_letters + "," + factor_letters[:-1] + "->" + core_letters
        core = jnp.einsum(einsum_str, T, *UT)
        return cls(core, factors)

    @property
    def shape(self):
        """
        Get the tuple representing the shape of Tucker.

        :return: Tucker shape
        :rtype: tuple
        """
        return tuple([factor.shape[0] for factor in self.factors])

    @property
    def rank(self):
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
        core = np.zeros(jnp.array(self.rank) + jnp.array(other.rank), dtype=self.dtype)
        sub_core_slice1 = []
        sub_core_slice2 = []
        for i in range(self.ndim):
            sub_core_slice1.append(slice(None, self.rank[i]))
            sub_core_slice2.append(slice(self.rank[i], None))
            factors.append(jnp.concatenate((self.factors[i], other.factors[i]), axis=1))

        core[tuple(sub_core_slice1)] = self.core
        core[tuple(sub_core_slice2)] = other.core
        core = jnp.array(core)
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
        return Tucker(a * new_tensor.core, new_tensor.factors)

    def __neg__(self):
        new_tensor = copy(self)
        return (-1) * new_tensor

    def __sub__(self, other):
        other = -other
        return self + other

    def round(self, eps=1e-14):
        """
        HOSVD rounding procedure, returns a `Tucker` with smaller `ranks`.

        :return: `Tucker` with reduced ranks.
        :rtype: `Tucker`
        :raises [ValueError]: if `eps` < 0
        """
        if eps < 0:
            raise ValueError("eps should be greater or equal than 0")
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
        """
        K-mode tensor-matrix product.

        :param k: mode id from 0 to ndim - 1
        :return: the result of k-mode tensor-matrix product
        :rtype: `Tucker`
        :raises [ValueError]: if `k` not from valid range
        """
        if k < 0 or k >= self.ndim:
            raise ValueError(f"k shoduld be from 0 to {self.ndim - 1}")
        new_tensor = copy(self)
        new_tensor.factors[k] = mat @ new_tensor.factors[k]
        return new_tensor

    def norm(self, qr_based=False):
        """
        Frobenius norm of `Tucker`.

        :param qr_based: whether to use stable QR-based implementation of norm, which is not differentiable,
        or unstable but differentiable implementation based on inner product. By default differentiable implementation
        is used
        :return: non-negative number which is
        the Frobenius norm of `Tucker` :rtype: `float`
        """
        if qr_based:
            core_factors = []
            for i in range(self.ndim):
                core_factors.append(jnp.linalg.qr(self.factors[i])[1])

            new_tensor = Tucker(self.core, core_factors)
            new_tensor = new_tensor.full()
            return np.linalg.norm(new_tensor)

        return jnp.sqrt(self.flat_inner(self))

    def full(self):
        """
        Dense representation of `Tucker`.

        :return: Dense tensor
        :rtype: `jnp.array`
        """
        core_letters = ascii_letters[:self.ndim]
        factor_letters = ""
        tensor_letters = ""
        for i in range(self.ndim):
            factor_letters += f"{ascii_letters[self.ndim + i]}{ascii_letters[i]},"
            tensor_letters += ascii_letters[self.ndim + i]

        einsum_str = core_letters + "," + factor_letters[:-1] + "->" + tensor_letters

        return jnp.einsum(einsum_str, self.core, *self.factors)
