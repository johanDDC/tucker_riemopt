from tucker_riemopt.symmetric.tucker import Tucker
from tucker_riemopt import Tucker as RegularTucker
from tucker_riemopt import backend as back
from numpy import arange
from string import ascii_letters


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
        proj: projections of gradient onto the tangent space,
        fx: func(T)
    """
    fx = None

    def g(core, factors):
        nonlocal T, fx
        common_factors = [back.concatenate([T.common_factors[i], factors[i]], axis=1)
                          for i in range(len(T.common_factors))]
        new_core = group_cores(core, T.core)
        symmetric_factor = back.concatenate([T.symmetric_factor, factors[-1]], axis=1)
        X = Tucker(new_core, common_factors, T.num_symmetric_modes, symmetric_factor)
        fx = func(X)
        return fx

    dg = back.grad(g, [0, 1], retain_graph=retain_graph)

    dS, dU = dg(T.core, [back.zeros_like(T.common_factors[i]) for i in range(len(T.common_factors))] +
                [back.zeros_like(T.symmetric_factor)])
    dU = [dU[i] - T.common_factors[i] @ (T.common_factors[i].T @ dU[i]) for i in range(len(T.common_factors))] + \
         [dU[-1] - T.symmetric_factor @ (T.symmetric_factor.T @ dU[-1])]
    modes = list(range(T.ndim))
    # non-symmetric factors
    factors = []
    for i in range(T.ndim - T.num_symmetric_modes):
        unfolding_core = back.transpose(T.core, [modes[i], *(modes[:i] + modes[i + 1:])])
        unfolding_core = back.reshape(unfolding_core, (T.core.shape[i], -1), order="F")
        gram_core = unfolding_core @ unfolding_core.T
        L = back.lu_factor(gram_core)
        factor = back.lu_solve(L, dU[i].T).T
        factors.append(back.concatenate([T.common_factors[i], factor], axis=1))
    # symmetric factors
    sum_core_unfoldings = None
    prefix = ascii_letters[:T.ndim - T.num_symmetric_modes]
    postfix = ascii_letters[T.ndim - T.num_symmetric_modes:T.ndim - 1]
    for i in range(T.num_symmetric_modes):
        scrpit = ",".join([
            "".join([prefix, "x", postfix]),
            "".join([prefix, "y", postfix]),
        ]) + "->xy"
        unfolding = back.einsum(scrpit, T.core, T.core)
        if sum_core_unfoldings is None:
            sum_core_unfoldings = unfolding
        else:
            sum_core_unfoldings += unfolding
        if len(postfix) > 0:
            prefix = "".join([prefix, postfix[0]])
            postfix = postfix[1:]

    L = back.lu_factor(sum_core_unfoldings)
    factor = back.lu_solve(L, dU[-1].T).T
    symmetric_factor = back.concatenate([T.symmetric_factor, factor], axis=1)
    return Tucker(group_cores(dS, T.core), factors,
                  T.num_symmetric_modes, symmetric_factor), fx


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
