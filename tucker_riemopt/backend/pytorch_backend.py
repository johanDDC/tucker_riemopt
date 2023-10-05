import warnings
import itertools
import numpy as np
import typing

try:
    import torch
except ImportError as error:
    message = ("Impossible to import PyTorch.\n"
               "To use tucker_riemopt with the PyTorch backend, "
               "you must first install PyTorch!")
    raise ImportError(message) from error

from opt_einsum import contract
from distutils.version import LooseVersion

from .backend import Backend

linalg_lstsq_avail = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class PyTorchBackend(Backend, backend_name="pytorch"):
    @staticmethod
    def type():
        return type(PyTorchBackend.tensor([]))

    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype,
                "device": tensor.device,
                "requires_grad": tensor.requires_grad}

    @staticmethod
    def tensor(data, dtype=torch.float32, device="cpu", requires_grad=False):
        if isinstance(data, np.ndarray):
            data = data.copy()
        return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

    @staticmethod
    def to_numpy(tensor):
        if torch.is_tensor(tensor):
            if tensor.requires_grad:
                tensor = tensor.detach()
            if tensor.cuda:
                tensor = tensor.cpu()
            return tensor.numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.asarray(tensor)

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.dim()

    @staticmethod
    def arange(start, stop=None, step=1.0, *args, **kwargs):
        if stop is None:
            return torch.arange(start=0., end=float(start), step=float(step), *args, **kwargs)
        else:
            return torch.arange(float(start), float(stop), float(step), *args, **kwargs)

    @staticmethod
    def reshape(tensor, newshape, order="C"):
        def reshape_fortran(x, shape):
            if len(x.shape) > 0:
                x = x.permute(*reversed(range(len(x.shape))))
            return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

        if order == "C":
            return torch.reshape(tensor, newshape)
        elif order == "F":
            return reshape_fortran(tensor, newshape)
        else:
            raise NotImplementedError("Only C-style and Fortran-style reshapes are supported")

    @staticmethod
    def clip(tensor, a_min=None, a_max=None, inplace=False):
        if a_max is None:
            a_max = torch.max(tensor)
        if a_min is None:
            a_min = torch.min(tensor)
        if inplace:
            return torch.clamp(tensor, a_min, a_max, out=tensor)
        else:
            return torch.clamp(tensor, a_min, a_max)

    @staticmethod
    def all(tensor):
        return torch.sum(tensor != 0)

        def transpose(self, tensor, axes=None):
            if axes is not None:
                axes = axes
            else:
                axes = list(range(self.ndim(tensor)))[::-1]
            return tensor.permute(*axes)

    @staticmethod
    def copy(tensor):
        return tensor.clone()

    @staticmethod
    def norm(tensor, ord=None, axis=None):
        return torch.linalg.norm(tensor, ord=ord, dim=axis)

    @staticmethod
    def dot(a, b):
        if a.ndim > 2 and b.ndim > 2:
            return torch.tensordot(a, b, dims=([-1], [-2]))
        if not a.ndim or not b.ndim:
            return a * b
        return torch.matmul(a, b)

    @staticmethod
    def mean(tensor, axis=None):
        if axis is None:
            return torch.mean(tensor)
        else:
            return torch.mean(tensor, dim=axis)

    @staticmethod
    def sum(tensor, axis=None):
        if axis is None:
            return torch.sum(tensor)
        else:
            return torch.sum(tensor, dim=axis)

    @staticmethod
    def max(tensor, axis=None):
        if axis is None:
            return torch.max(tensor)
        else:
            return torch.max(tensor, dim=axis)[0]

    @staticmethod
    def flip(tensor, axis=None):
        if isinstance(axis, int):
            axis = [axis]

        if axis is None:
            return torch.flip(tensor, dims=[i for i in range(tensor.ndim)])
        else:
            return torch.flip(tensor, dims=axis)

    @staticmethod
    def concatenate(tensors, axis=0):
        return torch.cat(tensors, dim=axis)

    @staticmethod
    def argmin(input, axis=None):
        return torch.argmin(input, dim=axis)

    @staticmethod
    def argsort(input, axis=None, descending=False):
        return torch.argsort(input, dim=axis, descending=descending)

    @staticmethod
    def argmax(input, axis=None):
        return torch.argmax(input, dim=axis)

    @staticmethod
    def stack(arrays, axis=0):
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def diag(tensor, k=0):
        return torch.diag(tensor, diagonal=k)

    @staticmethod
    def sort(tensor, axis, descending=False):
        if axis is None:
            tensor = tensor.flatten()
            axis = -1

        return torch.sort(tensor, dim=axis, descending=descending).values

    @staticmethod
    def update_index(tensor, index, values):
        tensor.index_put_(index, values)

    def solve(self, matrix1, matrix2):
        if self.ndim(matrix2) < 2:
            solution, _ = torch.solve(matrix2.unsqueeze(1), matrix1)
        else:
            solution, _ = torch.solve(matrix2, matrix1)
        return solution

    @staticmethod
    def lstsq(a, b):
        if linalg_lstsq_avail:
            x, residuals, _, _ = torch.linalg.lstsq(a, b, rcond=None, driver="gelsd")
            return x, residuals
        else:
            n = a.shape[1]
            sol = torch.lstsq(b, a)[0]
            x = sol[:n]
            residuals = torch.linalg.norm(sol[n:], ord=2, dim=0) ** 2
            return x, residuals if torch.matrix_rank(a) == n else torch.tensor([], device=x.device)

    @staticmethod
    def eigh(tensor):
        return torch.symeig(tensor, eigenvectors=True)

    @staticmethod
    def svd(matrix, full_matrices=True):
        some = not full_matrices
        u, s, v = torch.svd(matrix, some=some, compute_uv=True)
        return u, s, v.transpose(-2, -1).conj()

    @staticmethod
    def cumsum(tensor, axis=None):
        return torch.cumsum(tensor, dim=-1 if axis is None else axis)

    @staticmethod
    def grad(func: typing.Callable, argnums: typing.Union[int, typing.Sequence[int]] = 0, retain_graph=False):
        def grad_tensor(tensor):
            return tensor.grad

        def grad_list(lst):
            grads = []
            for el in lst:
                grads.append(grad_tensor(el))
            return grads

        def process_grad(elem):
            if type(elem) is list:
                if len(elem) == 0:
                    return []
                if not type(elem[0]) is PyTorchBackend.type():
                    raise TypeError("Expected list of torch.tensor, not list of {}".format(type(elem[0])))
                return grad_list(elem)
            elif type(elem) is PyTorchBackend.type():
                return grad_tensor(elem)
            else:
                raise TypeError("Unsupported argument type for grad method")

        def set_require_grad(args, argnums):
            for arg in argnums:
                if type(args[arg]) is PyTorchBackend.type():
                    if args[arg].is_leaf:
                        args[arg].requires_grad_(True)
                    else:
                        args[arg].retain_grad()
                elif type(args[arg]) is list:
                    set_require_grad(args[arg], np.arange(0, len(args[arg])))

        def detach(args, argnums):
            for arg in argnums:
                if type(args[arg]) is PyTorchBackend.type():
                    if args[arg].is_leaf:
                        args[arg].requires_grad_(False)
                        args[arg].grad = None
                elif type(args[arg]) is list:
                    detach(args[arg], np.arange(0, len(args[arg])))

        def aux_func(*args):
            args = list(args)
            set_require_grad(args, argnums if type(argnums) is list else [argnums])
            func(*args).backward(retain_graph=retain_graph)
            if type(argnums) is int:
                grads = process_grad(args[argnums])
                detach([grads], [0])
            else:
                grads = []
                for arg in argnums:
                    grads.append(process_grad(args[arg]))
                detach(grads, np.arange(0, len(grads)))
            detach(args, argnums if type(argnums) is list else [argnums])
            return grads

        return aux_func

    @staticmethod
    def pad(tensor, pad_width, mode, **kwargs):
        def get_mode_torch(mode_numpy):
            if mode_numpy in ['reflect', 'constant']:
                return mode_numpy
            elif mode_numpy == 'edge':
                return 'replicate'
            elif mode_numpy == 'wrap':
                return 'circular'
            else:
                assert False, f'NumPy mode "{mode_numpy}" has no PyTorch equivalent'

        def get_pad_width_torch(pad_width_numpy, mode_torch):
            pad_width_torch = tuple(itertools.chain.from_iterable(reversed(pad_width_numpy)))

            if mode_torch in ['reflect', 'replicate', 'circular']:
                assert all([p == 0 for p in pad_width_torch[
                                            -4:]]), f'Cannot pad first two dimensions in PyTorch with mode="{mode_torch}"'
                return pad_width_torch[:-4]

            return pad_width_torch

        mode_torch = get_mode_torch(mode)
        pad_width_torch = get_pad_width_torch(pad_width, mode_torch)
        value = kwargs.get("constant_values")
        res = torch.nn.functional.pad(tensor, pad_width_torch, mode=mode_torch, value=value)
        return res

    @staticmethod
    def einsum(subscripts, *operands):
        return contract(subscripts, *operands, backend="torch")

    @staticmethod
    def cho_factor(A, upper=False, **kwargs):
        return torch.linalg.cholesky_ex(A, upper=upper, **kwargs)

    @staticmethod
    def cho_solve(B, L, upper=False, **kwargs):
        return torch.cholesky_solve(B, L, upper=upper, **kwargs)

    @staticmethod
    def lu_factor(A, pivot=True):
        if A.device.type == "cuda":
            lu, pivot, _ = torch.linalg.lu_factor_ex(A)
        else:
            lu, pivot = torch.linalg.lu_factor(A)
        return lu, pivot

    @staticmethod
    def lu_solve(lu_pivots, B, left=True):
        return torch.linalg.lu_solve(lu_pivots[0], lu_pivots[1], B, left=left)


for name in ["float64", "float32", "int64", "int32", "complex128", "complex64",
             "is_tensor", "ones", "zeros", "any", "trace", "count_nonzero",
             "zeros_like", "eye", "min", "prod", "abs", "matmul",
             "sqrt", "sign", "where", "conj", "finfo", "log2", "sin", "cos", "squeeze"]:
    PyTorchBackend.register_method(name, getattr(torch, name))

if LooseVersion(torch.__version__) < LooseVersion("1.8.0"):
    warnings.warn(f"You are using an old version of PyTorch ({torch.__version__}). "
                  "We recommend upgrading to a newest one, e.g. >1.8.0.")
    PyTorchBackend.register_method("qr", getattr(torch, "qr"))

else:
    for name in ["kron"]:
        PyTorchBackend.register_method(name, getattr(torch, name))

    for name in ["solve", "qr", "svd", "eigh"]:
        PyTorchBackend.register_method(name, getattr(torch.linalg, name))
