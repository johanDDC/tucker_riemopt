from tucker_riemopt import Tucker

from tucker_riemopt import backend as back

def group_cores(core1, core2):
    d = len(core1.shape)
    r = core1.shape

    new_core = core1
    to_concat = core2

    for i in range(d):
        to_concat = back.pad(to_concat, [(0, r[j]) if j == i - 1 else (0, 0) for j in range(d)], constant_values=0)
        new_core = back.concatenate([new_core, to_concat], axis=i)

    return new_core


def compute_gradient_projection(func, T):
    """
    Input
        X: tensor from manifold
        
     Output
        proj: projections of gradient onto the tangent space
    """

    def g(T1, core, factors):
        new_factors = [back.concatenate([T1.factors[i], factors[i]], axis=1) for i in range(T1.ndim)]
        new_core = group_cores(core, T1.core)

        T = Tucker(new_core, new_factors)
        return func(T)

    dg = back.grad(g, [1, 2])

    dS, dU = dg(T, T.core, [back.zeros_like(T.factors[i]) for i in range(T.ndim)])
    dU = [dU[i] - T.factors[i] @ (T.factors[i].T @ dU[i]) for i in range(len(dU))]
    return Tucker(group_cores(dS, T.core), [back.concatenate([T.factors[i], dU[i]], axis=1) for i in range(T.ndim)])

def vector_transport(x: Tucker, y: Tucker, xi: Tucker):
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
    return compute_gradient_projection(f, y)
