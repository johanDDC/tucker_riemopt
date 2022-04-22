from typing import List, Union, Sequence, Dict
import numpy as np
from flax import struct
from string import ascii_letters
from copy import deepcopy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, svds

from tucker_riemopt import backend as back

class SparseTensor:
    """
        Contains sparse tensor in coo format. It can be constructed manually or converted from dense format.
    """

    def __init__(self, shape : Sequence[int], inds : Sequence[Sequence[int]], vals : Sequence[back.float64]):
        """
        Parameters
        ----------
            shape: tuple
                tensor shape
            inds: tuple of integer arrays of size nnz
                positions of nonzero elements
            vals: list of size nnz
                corresponding values
        """

        assert len(inds) == len(shape) >= 1
        assert len(vals) == len(inds[0])
        self.shape = back.to_numpy(shape)
        self.inds = back.to_numpy(inds)
        self.vals = back.to_numpy(vals)

    @classmethod
    def dense2sparse(cls, T : back.type()):
        inds = [np.arange(mode, dtype=int) for mode in T.shape]
        grid_inds = tuple(I.flatten(order="F") for I in np.meshgrid(*inds, indexing="ij"))
        return cls(T.shape, grid_inds, back.reshape(T, (-1,), order="F"))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nnz(self):
        return len(self.vals)

    def unfolding(self, k):
        def multiindex(p):
            """
                Calculates \overline{i_1 * ... * i_d} except i_k
            """
            joined_ind = 0
            for i in range(0, self.ndim):
                if i != p:
                    shape_prod = 1
                    if i != 0:
                        shape_prod = np.prod(self.shape[:i])
                    if i > p:
                        shape_prod //= self.shape[p]
                    joined_ind += self.inds[i] * shape_prod
            return joined_ind

        if k < 0 and k > self.ndim:
            raise ValueError(f"param k should be between 0 and {self.ndim}")
        row = self.inds[k]
        col = multiindex(k)
        unfolding = coo_matrix((self.vals, (row, col)),
                                    shape=(self.shape[k], np.prod(self.shape) // self.shape[k]))
        return unfolding.tocsr()

    @staticmethod
    def __unfolding_to_tensor(unfolding : csr_matrix, k : int, shape : Sequence[int]):
        """
            Converts k-unfolding matrix back to sparse tensor.
        """
        def detach_multiindex(p, multiindex):
            """
                Performs detaching of multiindex back to tuple of indices. Excepts p-mode.
            """
            inds = []
            dynamic = np.zeros_like(multiindex[1]) # use dynamic programming to prevent d**2 complexity
            shape_prod = np.prod(shape) // shape[p]
            # shape_prod //= shape[-2] if p == len(shape) - 1 else shape[-1]
            for i in range(len(shape) - 1, -1, -1):
                if i != p:
                    shape_prod //= shape[i]
                    inds.append((multiindex[1] - dynamic) // shape_prod)
                    dynamic += inds[-1] * shape_prod
                else:
                    inds.append(multiindex[0])
            return inds[::-1]

        unfolding = unfolding.tocoo()
        vals = unfolding.data
        inds = detach_multiindex(k, (unfolding.row, unfolding.col))
        return SparseTensor(shape, inds, vals)
    
    def reshape(self, new_shape : Sequence[Sequence[int]]):
        """
            Reshape sparse tensor.

            Parameters
            ----------
             new_shape : Sequence[Sequence[int]]
                Maps old tensor shape to a new one. For example, if tensor was of order 3 of shape (512, 512, 3), then
                 each of its element had 3 indices (i, j, k). Now we want to reshape it to tensor of order 5 of shape
                 (64, 8, 64, 8, 3). Then `new_shape` parameter will look like [[64, 8], [64, 8], [3]].
        """
        def detach_multiindex(multiindex, shape):
            """
                Performs detaching of multiindex back to tuple of indices. Excepts p-mode.
            """
            inds = []
            dynamic = np.zeros_like(multiindex)  # use dynamic programming to prevent d**2 complexity
            shape_prod = back.prod(back.tensor(shape))
            for i in range(len(shape) - 1, -1, -1):
                shape_prod //= shape[i]
                inds.append((multiindex - dynamic) // shape_prod)
                dynamic += inds[-1] * shape_prod
            return inds[::-1]

        new_inds = None
        shape_new = []
        for i, shape in enumerate(new_shape):
            # new_shape_inds = back.zeros((len(shape), self.nnz), dtype=back.int32)
            shape_new += shape
            new_shape_inds = back.tensor(detach_multiindex(self.inds[i], shape))
            new_inds = new_shape_inds if new_inds is None else back.concatenate((new_inds, new_shape_inds))

        return SparseTensor(shape_new, new_inds, back.copy(self.vals))


    def contract(self, contraction_dict : Dict[int, back.type()]):
        """
            Performs tensor contraction.

            Parameters
            ----------
            contraction_dict: dict[int, back.tensor]
                dictionary of pairs (i, M), where i is mode index, and M is matrix, we want to contract tensor with by i mode

            Returns
            -------
            SparseTensor : result of contranction
        """
        def contract_by_mode(T : SparseTensor, k: int, M : back.type()):
            """
                Performs tensor contraction by specified mode using the following property:
                {<A, M>_k}_(k) = M @ A_(k)
                where <., .>_k is k-mode contraction, A_(k) is k-mode unfolding
            """
            M = csr_matrix(M)
            unfolding = T.unfolding(k)
            new_unfolded_tensor = M @ unfolding
            new_shape = list(T.shape)
            new_shape[k] = M.shape[0]
            return SparseTensor.__unfolding_to_tensor(new_unfolded_tensor, k, new_shape)

        result = self
        for mode in contraction_dict.keys():
            result = contract_by_mode(result, mode, contraction_dict[mode])
        return result

    def to_dense(self):
        """
            Converts sparse tensor to dense format.
            Be sure, that tensor can be constructed in memory in dense format.
        """
        T = np.zeros(self.shape)
        T[tuple(self.inds[i] for i in range(self.ndim))] = self.vals
        return back.tensor(T)


ML_rank = Union[int, Sequence[int]]

@struct.dataclass
class Tucker:
    core: back.type()
    factors: List[back.type()]

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

    @classmethod
    def sparse2tuck(cls, sparse_tensor : SparseTensor, max_rank: ML_rank=None, maxiter: Union[int, None]=5):
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
        for i, k in enumerate(range(sparse_tensor.ndim)):
            unfolding = sparse_tensor.unfolding(k)
            unfolding_linop = LinearOperator(unfolding.shape, matvec=lambda x: unfolding @ x,
                                             rmatvec=lambda x: unfolding.T @ x)
            factors.append(back.tensor(svds(unfolding_linop, max_rank[k], return_singular_vectors="u")[0]))
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
        raise NotImplementedError()

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
        new_tensor = deepcopy(self)
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
        new_tensor = deepcopy(self)
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
