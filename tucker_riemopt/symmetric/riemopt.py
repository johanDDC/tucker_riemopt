from tucker_riemopt.symmetric.tucker import Tucker
from tucker_riemopt import Tucker as RegularTucker
from tucker_riemopt import backend as back
from numpy import arange


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


def compute_gradient_projection(func, T, retain_graph=False):
    """
    Computes riemann gradient of given function in given point of manifold

    Parameters
        func: Callable
        T: Tucker
            Tensor from manifold
        retain_graph: bool
            Whether you want go backwards through computational graph multiple times
            (may be significant for some backgrounds like PyTorch)

     Output
        proj: projections of gradient onto the tangent space
    """

    def g(core, factors):
        nonlocal T
        common_factors = [back.concatenate([T.common_factors[i], factors[i]], axis=1)
                          for i in range(len(T.common_factors))]
        new_core = group_cores(core, T.core)
        symmetric_factor = back.concatenate([T.symmetric_factor, factors[-1]], axis=1)
        X = Tucker(new_core, common_factors, T.symmetric_modes, symmetric_factor)
        return func(X)

    dg = back.grad(g, [0, 1], retain_graph=retain_graph)

    dS, dU = dg(T.core, [back.zeros_like(T.common_factors[i]) for i in range(len(T.common_factors))] +
                [back.zeros_like(T.symmetric_factor)])
    dU = [dU[i] - T.common_factors[i] @ (T.common_factors[i].T @ dU[i]) for i in range(len(T.common_factors))] + \
         [dU[-1] - T.symmetric_factor @ (T.symmetric_factor.T @ dU[-1])]
    modes = list(range(T.ndim))
    common_modes = list(set(modes) - set(T.symmetric_modes))
    # non-symmetric factors
    factors = []
    for idx, mode in enumerate(common_modes):
        unfolding_core = back.transpose(T.core, [modes[mode], *(modes[:mode] + modes[mode + 1:])])
        unfolding_core = back.reshape(unfolding_core, (T.core.shape[mode], -1), order="F")
        gram_core = unfolding_core @ unfolding_core.T
        L = back.cho_factor(gram_core)
        factor = back.cho_solve(dU[idx].T, L[0]).T
        factors.append(back.concatenate([T.common_factors[idx], factor], axis=1))
    # symmetric factors
    unfolding_core = back.transpose(T.core, [modes[T.symmetric_modes[0]],
                                             *(modes[:T.symmetric_modes[0]] + modes[T.symmetric_modes[0] + 1:])])
    unfolding_core = back.reshape(unfolding_core, (T.core.shape[T.symmetric_modes[0]], -1), order="F")
    gram_core = unfolding_core @ unfolding_core.T
    L = back.cho_factor(gram_core)
    factor = back.cho_solve(dU[-1].T, L[0]).T
    symmetric_factor = back.concatenate([T.symmetric_factor, factor], axis=1)
    return Tucker(group_cores(dS, T.core), factors,
                  T.symmetric_modes, symmetric_factor)


def vector_transport(x: Tucker, y: Tucker, xi: Tucker, retain_graph=False):
    """
        Performs vector transport of tangent vector `xi` from `T_xM` to T_yM.

        Parameters
        ----------
        x : Tucker
        y : Tucker
        xi : Tucker
            Vector which transports from T_xM to T_yM

        Returns
        -------
        xi_y: Tucker
            Result of vector transport `xi`.
    """
    f = lambda u: u.flat_inner(xi)
    return compute_gradient_projection(f, y, retain_graph)
