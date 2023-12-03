import numpy as np

from typing import Union, List, Callable, Tuple
from string import ascii_letters

from tucker_riemopt import Tucker
from tucker_riemopt import backend as back
from tucker_riemopt.tucker.matrix import TuckerMatrix


class TangentVector:
    """Special representation for tangent vectors of some `point` from the manifold.

    :param point: Point from the manifold of tensors of fixed multilinear (`Tucker`) rank.
    :param delta_core: Tangent vector delta core.
    :param delta_factors: Tangent vector delta factors.
    """
    point: Tucker
    delta_core: back.type()
    delta_factors: List[back.type()]

    def __init__(self, manifold_point, delta_core: Union[None, back.type()] = None,
                 delta_factors: Union[None, back.type()] = None):
        self.point = manifold_point
        self.delta_core = delta_core if delta_core is not None else self.point.core
        self.delta_factors = delta_factors if delta_factors is not None else \
            [back.zeros_like(self.point.factors[i]) for i in range(self.point.ndim)]

    @staticmethod
    def group_cores(corner_core, padding_core):
        """Combine point's core and delta core into unite core of tangent vector.

        :param corner_core: Corner block of the grouped core.
        :param padding_core: Adjacent block of the grouped core.
        :return: Grouped core.
        """
        d = len(corner_core.shape)
        delta_shape = corner_core.shape

        new_core = back.copy(corner_core)
        to_concat = back.copy(padding_core)

        for i in range(d):
            to_concat = back.pad(to_concat, [(0, delta_shape[j]) if j == i - 1 else (0, 0) for j in range(d)],
                                 mode="constant",
                                 constant_values=0)
            new_core = back.concatenate([new_core, to_concat], axis=i)

        return new_core

    def construct(self):
        """Build `Tucker` tensor from `TangentVector` representation with at most twice bigger rank.

        :return: `Tucker` tensor.
        """
        grouped_core = self.group_cores(self.delta_core, self.point.core)
        factors = [back.concatenate([
            self.point.factors[i], self.delta_factors[i]
        ], axis=1) for i in range(self.point.ndim)]
        if isinstance(self.point, TuckerMatrix):
            return TuckerMatrix(grouped_core, factors, self.point.n, self.point.m)
        return Tucker(grouped_core, factors)

    def __rmul__(self, a: float):
        """Elementwise multiplication of `TangentVector` by scalar.

        :param a: Scalar value.
        :return: `TangentVector` tensor.
        """
        return TangentVector(self.point, a * self.delta_core, [a * factor for factor in self.delta_factors])
    
    def __neg__(self):
        return (-1) * self

    def __add__(self, other: "TangentVector"):
        """Addition of two `TangentVector`s. It is assumed that `other` is a vector from the same tangent space
        (the `self.point` and `other.point` fields are the same). Otherwise, the result may be incorrect.

        :param other: `TangentVector` from the same tangent space.
        :return: `TangentVector` from the same tangent space.
        """
        new_delta_core = self.delta_core + other.delta_core
        new_delta_factors = []
        for self_factor, other_factor in zip(self.delta_factors, other.delta_factors):
            new_delta_factors.append(self_factor + other_factor)
        return TangentVector(self.point, new_delta_core, new_delta_factors)

    def norm(self):
        """Norm of tangent vector. This method is not differentiable as applies series of QR decompositions.

        :return: Frobenius norm of tangent vector.
        """
        norms = back.norm(self.delta_core) ** 2
        core_letters = ascii_letters[:self.point.ndim]
        for i, factor in enumerate(self.delta_factors):
            R = back.qr(factor)[1]
            norms += back.norm(
                back.einsum(f"{core_letters},y{core_letters[i]}->{core_letters[:i]}y{core_letters[i + 1:]}",
                            self.delta_core, R)
            ) ** 2
        return back.sqrt(norms)


def grad(f: Callable[[Tucker], float], X: Tucker, retain_graph=False) -> Tuple[TangentVector, float]:
    """Compute the Riemannian gradient of the smooth scalar valued function `f` at point `X`. The result is the tangent
     vector of `X`. Also returns value `f(X)`.

    :param f: Smooth scalar valued function.
    :param X: Point from the manifold.
    :param retain_graph: Optional argument, which may be provided to autodiff framework (e.g. pytorch).
    :return: A tangent vector of `X` which is the Riemannian gradient of `f` and value `f(X)`.
    """
    fx = None
    def h(delta_core, delta_factors):
        nonlocal X, fx
        tangent_X = TangentVector(X, delta_core, delta_factors).construct()
        fx = f(tangent_X)  # computes value as a side effect
        return fx

    dh = back.grad(h, [0, 1], retain_graph=retain_graph)
    dS, dV = dh(X.core, [back.zeros_like(X.factors[i]) for i in range(X.ndim)])
    core_letters = ascii_letters[:X.ndim]
    for i in range(X.ndim):
        dV[i] = dV[i] - X.factors[i] @ (X.factors[i].T @ dV[i])
        gram_core = back.einsum(f"{core_letters[:i]}X{core_letters[i+1:]},{core_letters[:i]}Y{core_letters[i+1:]}->XY",
                              X.core, X.core)
        dV[i] = back.cho_solve(dV[i].T, back.cho_factor(gram_core)[0]).T

    return TangentVector(X, dS, dV), fx


def project(X: Tucker, xi: Tucker, retain_graph=False) -> TangentVector:
    """Project `xi` onto the tangent space of `X`.

    :param X: Point from the manifold.
    :param xi: Arbitrary tensor represented in Tucker format.
    :return: Result of projection of `xi` onto the tangent space of `X`.
    """
    f = lambda x: x.flat_inner(xi)
    return grad(f, X, retain_graph)[0]
