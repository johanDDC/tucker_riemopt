from tucker_riemopt import Tucker
from tucker_riemopt import backend as back
from numpy import arange

def group_cores(corner_core, padding_core):
    d = len(corner_core.shape)
    r = corner_core.shape

    new_core = back.copy(corner_core)
    to_concat = back.copy(padding_core)

    for i in range(d):
        to_concat = back.pad(to_concat, [(0, r[j]) if j == i - 1 else (0, 0) for j in range(d)], mode="constant", constant_values=0)
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
    fx = None

    def g(core, factors):
        nonlocal T, fx
        new_factors = [back.concatenate([T.factors[i], factors[i]], axis=1) for i in range(T.ndim)]
        new_core = group_cores(core, T.core)
        X = Tucker(new_core, new_factors)
        fx = func(X)
        return fx

    dg = back.grad(g, [0, 1], retain_graph=retain_graph)

    dS, dU = dg(T.core, [back.zeros_like(T.factors[i]) for i in range(T.ndim)])
    dU = [dU[i] - T.factors[i] @ (T.factors[i].T @ dU[i]) for i in range(T.ndim)]
    modes = list(arange(0, T.ndim))
    for i in range(T.ndim):
        unfolding_core = back.transpose(T.core, [modes[i], *(modes[:i] + modes[i + 1:])])
        unfolding_core = back.reshape(unfolding_core, (T.core.shape[i], -1), order="F")
        gram_core = unfolding_core @ unfolding_core.T
        L = back.lu_factor(gram_core)
        dU[i] = back.lu_solve(L, dU[i].T).T

    return Tucker(group_cores(dS, T.core), [back.concatenate([T.factors[i], dU[i]], axis=1) for i in range(T.ndim)]), fx

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
