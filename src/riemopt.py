from src.tucker import Tucker
import numpy as np

from src import backend as back

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


def optimize(f, g, X0, maxiter=10):
    """
    Input
        f: function to maximize
        X0: first approximation
        maxiter: number of iterations to perform

    Output
        Xk: approximation after maxiter iterations
        errs: values of functional on each step
    """
    X = X0
    max_rank = np.max(X.rank)

    errs = []
    errs.append(f(X))
    
    for i in range(maxiter):
        print(f'Doing iteration {i+1}/{maxiter}\t Calculating gradient...\t', end='\r')
        G = compute_gradient_projection(X, g)
        print(f'Doing itaration {i+1}/{maxiter}\t Calculating tau...\t\t', end='\r')
        tau = 1
        print(f'Doing iteration {i+1}/{maxiter}\t Calculating retraction...\t', end='\r')
        X = X + tau * G
        X = X.round(max_rank=max_rank) # retraction
        
        errs.append(f(X))
        print(f'Done iteration {i+1}/{maxiter}!\t Error: {errs[-1]}' + ' ' * 50, end='\r')
        
    return X, errs
