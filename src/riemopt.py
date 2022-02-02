from typing import Callable

import jax
import jax.config
jax.config.update("jax_enable_x64", True)

from matrix import TuckerMatrix
from tucker import Tucker
import numpy as np
import jax.numpy as jnp

A = np.diag([1, 2, 3, 4])
Q, _ = np.linalg.qr(np.random.random((4, 4)))
A = Q @ A @ Q.T
A = TuckerMatrix.full2tuck(A.reshape([2] * 4), [2] * 2, [2] * 2)

@jax.jit
def f(T):
    return (T.flat_inner(A @ T)) / T.flat_inner(T)

@jax.jit
def group_cores(core1, core2):
    d = len(core1.shape)
    r = core1.shape

    new_core = core1
    to_concat = core2

    for i in range(d):
        to_concat = jnp.pad(to_concat, [(0, r[j]) if j == i - 1 else (0, 0) for j in range(d)], mode='constant', constant_values=0)
        new_core = jnp.concatenate([new_core, to_concat], axis=i)

    return new_core

@jax.jit
def g(T1, core, factors):
    d = T1.ndim
    r = T1.rank

    new_factors = [jnp.concatenate([T1.factors[i], factors[i]], axis=1) for i in range(T1.ndim)]
    new_core = group_cores(core, T1.core)

    T = Tucker(new_core, new_factors)
    return f(T)


@jax.jit
def compute_gradient_projection(T):
    """
    Input
        inds: list of N lists representing indices of tensor A
        vals: np.array of size N: values of A in indices inds
        X: tensor from manifold
        
     Output
        proj: projections of tensor A onto the tangent space
    """
    dg_dS = jax.grad(g, argnums=1)
    dg_dU = jax.grad(g, argnums=2)

    dS = dg_dS(T, T.core, [jnp.zeros_like(T.factors[i]) for i in range(T.ndim)])
    dU = dg_dU(T, T.core, [jnp.zeros_like(T.factors[i]) for i in range(T.ndim)])
    dU = [dU[i] - T.factors[i] @ (T.factors[i].T @ dU[i]) for i in range(len(dU))]
    return Tucker(group_cores(dS, T.core), [jnp.concatenate([T.factors[i], dU[i]], axis=1) for i in range(T.ndim)])


def optimize(f, X0, maxiter=10):
    """
    Input
        inds: list of N lists representing indices of tensor A
        vals: np.array of size N: values of A in indices inds
        rank: target rank of approximation
        X0: initial approximation. If None, chosen randomly
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
        G = compute_gradient_projection(X)
        print(f'Doing itaration {i+1}/{maxiter}\t Calculating tau...\t\t', end='\r')
        tau = 1
        print(f'Doing iteration {i+1}/{maxiter}\t Calculating retraction...\t', end='\r')
        X = X + tau * G
        X = X.round(max_rank=max_rank) # retraction
        
        errs.append(f(X))
        print(f'Done iteration {i+1}/{maxiter}!\t Error: {errs[-1]}' + ' ' * 50, end='\r')
        
    return X, errs

np.random.seed(229)

x = np.random.random(4)
X = Tucker.full2tuck(x.reshape([2] * 2))
print(f(X))

X, _ = optimize(f, X, maxiter=10)
x = X.full().reshape(4)
A = A.full().reshape((4, 4))

print(x)
print(A @ x)
print((x.T @ A @ x) / (x.T @ x))
