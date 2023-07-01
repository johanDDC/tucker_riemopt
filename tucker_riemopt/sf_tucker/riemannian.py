from string import ascii_letters

import numpy as np

from typing import Union, List, Callable, Tuple

from tucker_riemopt import SFTucker
from tucker_riemopt import backend as back


class TangentVector:
    point: SFTucker
    delta_core: back.type()
    delta_regular_factors: List[back.type()]
    delta_shared_factor: back.type()

    def __init__(self, manifold_point, delta_core: Union[None, back.type()] = None,
                 delta_regular_factors: Union[None, back.type()] = None,
                 delta_shared_factor: Union[None, back.type()] = None):
        self.point = manifold_point
        self.delta_core = delta_core if delta_core is not None else self.point.core
        self.delta_regular_factors = delta_regular_factors if delta_regular_factors is not None else \
            [back.zeros_like(self.point.regular_factors[i]) for i in range(self.point.ndim)]
        self.delta_shared_factor = delta_shared_factor if delta_shared_factor is not None else \
            back.zeros_like(self.point.shared_factor)

    @staticmethod
    def group_cores(corner_core, padding_core):
        d = len(corner_core.shape)
        r = corner_core.shape

        new_core = back.copy(corner_core)
        to_concat = back.copy(padding_core)

        for i in range(d):
            to_concat = back.pad(to_concat, [(0, r[j]) if j == i - 1 else (0, 0) for j in range(d)], mode="constant",
                                 constant_values=0)
            new_core = back.concatenate([new_core, to_concat], axis=i)

        return new_core

    def construct(self):
        grouped_core = self.group_cores(self.delta_core, self.point.core)
        regular_factors = [back.concatenate([
            self.point.regular_factors[i], self.delta_regular_factors[i]
        ], axis=1) for i in range(self.point.dt)]
        shared_factor = back.concatenate([self.point.shared_factor, self.delta_shared_factor], axis=1)
        return SFTucker(grouped_core, regular_factors, self.point.num_shared_factors, shared_factor)

    def linear_comb(self, a: float = 1, b: float = 1, xi: Union["TangentVector", None] = None):
        """
        Compute linear combination of this tangent vector of `X` (`self.point`) with either other tangent vector `xi` or
        `X`. Although, linear combination may be obtained by addition operation of SF-Tucker tensors, it is important to
        note, that such operation leads to rank increasing. For instance, if `X` rank is `r`, then ranks of its
        tangent vectors are `2r`, ant thus, the rank of the result of taking linear combination by naive addition is
        `4r`. On the other hand, this function obtains linear combination efficiently without increase of the rank. The
         rank of the result is always `2r`.

        :param a: parameter of the linear combination.
        :param b: parameter of the linear combination.
        :param xi: if `None`, that linear combination with manifold point `X` will be computed. With `xi` otherwise.
        :return: `a * self + b * xi` if `xi` is not `None` and `a * self + b * self.point` otherwise.
        """
        if xi is None:
            xi = TangentVector(self.point)

        regular_factors = [a * self.delta_regular_factors[i] + b * xi.delta_regular_factors[i] for i in
                           range(self.point.dt)]
        shared_factor = a * self.delta_shared_factor + b * xi.delta_shared_factor
        core = a * self.delta_core + b * xi.delta_core
        return TangentVector(self.point, core, regular_factors, shared_factor)


def grad(f: Callable[[SFTucker], float], X: SFTucker, retain_graph=False) -> Tuple[TangentVector, float]:
    """
    Compute the Riemannian gradient of the smooth scalar valued function `f` at point `X`. The result is the tangent
    vector of `X`. Also returns value `f(X)`.

    :param f: smooth scalar valued function.
    :param X: point from the manifold.
    :param retain_graph: optional argument, which may be provided to autodiff framework (e.g. pytorch).
    :return: a tangent vector of `X` which is the Riemannian gradient of `f` and value `f(X)`.
    """
    fx = None
    modes = list(np.arange(0, X.ndim))

    def h(delta_core, delta_regular_factors, delta_shared_factor):
        nonlocal X, fx
        tangent_X = TangentVector(X, delta_core, delta_regular_factors, delta_shared_factor).construct()
        fx = f(tangent_X)
        return fx

    dh = back.grad(h, [0, 1, 2], retain_graph=retain_graph)

    dS, dV, dU = dh(X.core, [back.zeros_like(X.regular_factors[i]) for i in range(X.dt)],
                    back.zeros_like(X.shared_factor))
    dV = [dV[i] - X.regular_factors[i] @ (X.regular_factors[i].T @ dV[i]) for i in range(X.dt)]
    dU = dU - X.shared_factor @ (X.shared_factor.T @ dU)
    # non-sf_tucker factors
    for i in range(X.dt):
        unfolding_core = back.transpose(X.core, [modes[i], *(modes[:i] + modes[i + 1:])])
        unfolding_core = back.reshape(unfolding_core, (X.core.shape[i], -1), order="F")
        gram_core = unfolding_core @ unfolding_core.T
        L = back.lu_factor(gram_core)
        dV[i] = back.lu_solve(L, dV[i].T).T
    # sf_tucker factors
    sum_core_unfoldings = None
    prefix = ascii_letters[:X.dt]
    postfix = ascii_letters[X.dt:X.ndim - 1]
    for i in range(X.ds):
        script = ",".join([
            "".join([prefix, "x", postfix]),
            "".join([prefix, "y", postfix]),
        ]) + "->xy"
        unfolding = back.einsum(script, X.core, X.core)
        if sum_core_unfoldings is None:
            sum_core_unfoldings = unfolding
        else:
            sum_core_unfoldings += unfolding
        if len(postfix) > 0:
            prefix = "".join([prefix, postfix[0]])
            postfix = postfix[1:]

    L = back.lu_factor(sum_core_unfoldings)
    dU = back.lu_solve(L, dU.T).T
    return TangentVector(X, dS, dV, dU), fx


def project(X: SFTucker, xi: SFTucker, retain_graph=False) -> TangentVector:
    """
    Project `xi` onto the tangent space of `X`.

    :param X: point from the manifold.
    :param xi: arbitrary tensor represented in SF-Tucker format.
    :return: result of projection of `xi` onto the tangent space of `X`.
    """
    f = lambda x: x.flat_inner(xi)
    return grad(f, X, retain_graph)[0]
