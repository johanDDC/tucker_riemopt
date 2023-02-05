import numpy as np

from typing import Sequence, Dict
from scipy.sparse import coo_matrix, csr_matrix

from tucker_riemopt import backend as back


class SparseTensor:
    """
        Contains sparse tensor in coo format. It can be constructed manually or converted from dense format.
    """

    def __init__(self, shape: Sequence[int], inds: Sequence[Sequence[int]], vals: Sequence[back.float64]):
        """
        Parameters
        ----------
            shape: tuple
                tensor shape
            inds: matrix of size ndim x nnz
                positions of nonzero elements
            vals: array of size nnz
                corresponding values
        """

        assert len(inds) == len(shape)
        assert len(vals) == len(inds[0])
        self.shape = shape
        self.inds = inds
        self.vals = vals
        self.shape_ = back.to_numpy(shape)
        self.inds_ = back.to_numpy(inds)
        self.vals_ = back.to_numpy(vals)

    @classmethod
    def dense2sparse(cls, T: back.type()):
        inds = [np.arange(mode, dtype=int) for mode in T.shape]
        grid_inds = tuple(I.flatten(order="F") for I in np.meshgrid(*inds, indexing="ij"))
        return cls(T.shape, grid_inds, back.reshape(T, (-1,), order="F"))

    @property
    def ndim(self):
        return len(self.shape_)

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
                        shape_prod = np.prod(self.shape_[:i])
                    if i > p:
                        shape_prod //= self.shape_[p]
                    joined_ind += self.inds[i] * shape_prod
            return joined_ind

        if k < 0 and k > self.ndim:
            raise ValueError(f"param k should be between 0 and {self.ndim}")
        row = self.inds[k]
        col = multiindex(k)
        unfolding = coo_matrix((self.vals, (row, col)),
                               shape=(self.shape_[k], np.prod(self.shape_) // self.shape_[k]))
        return unfolding.tocsr()

    @staticmethod
    def __unfolding_to_tensor(unfolding: csr_matrix, k: int, shape: Sequence[int]):
        """
            Converts k-unfolding matrix back to sparse tensor.
        """

        def detach_multiindex(p, multiindex):
            """
                Performs detaching of multiindex back to tuple of indices. Excepts p-mode.
            """
            inds = []
            dynamic = np.zeros_like(multiindex[1])  # use dynamic programming to prevent d**2 complexity
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

    def reshape(self, new_shape: Sequence[Sequence[int]]):
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

    def contract(self, contraction_dict: Dict[int, back.type()]):
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

        def contract_by_mode(T: SparseTensor, k: int, M: back.type()):
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
        T = np.zeros(self.shape_)
        T[tuple(self.inds_[i] for i in range(self.ndim))] = self.vals_
        return back.tensor(T)