import numpy as np
from typing import Callable
from collections import defaultdict
from copy import copy

from src.riemopt import compute_gradient_projection, vector_transport
from src.tucker import Tucker, TangentVector, ML_rank


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
            - 'Best' -- optimal step size inferred via analytical minimization.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            alpha_0 : The step size which is returned on every step.
    """
    def __init__(self, method='Armijo', **kwargs):
        self._method = method
        if self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        elif self._method == 'Best':
            pass
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    def __armijo(self, func: Callable[[TangentVector], Tucker], x_k: Tucker, d_k: TangentVector, rank: ML_rank, previous_alpha=None):
        alpha = previous_alpha if previous_alpha is not None else self.alpha_0
        armijo_threshold = self.c1 * alpha * d_k.norm(qr_based=True) ** 2
        fx = func(x_k)
        while fx - func((x_k + alpha * d_k).round(rank)) < armijo_threshold:
            alpha /= 2
            armijo_threshold /= 2
        return alpha

    def line_search(self, func: Callable[[TangentVector], Tucker], x_k: Tucker, d_k: TangentVector, rank: ML_rank, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        func : Callable[[TangentVector], Tucker]
            Cost function for minimizing.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == "Armijo":
            alpha = self.__armijo(func, x_k, d_k, rank, previous_alpha)
        if self._method == "Constant":
            alpha = self.c

        return alpha

def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def RGD(func: Callable, x_0: Tucker, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False):
    """
    Performs Riemannian steepest descent optimization method.

    Parameters
    ----------
    func : Callable
        Cost function for minimizing.
    x_0 : Tucker
        Starting point for optimization algorithm. Multilinear rank of x_0 defines rank of riemann manifold, on which
         optimization process is launched.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list)
    def write_history(func_val, grad_norm):
        history["func"].append(func_val)
        history["grad_norm"].append(grad_norm)

    line_search_tool = get_line_search_tool(line_search_options)
    x_k = copy(x_0)
    rank = x_k.rank
    alpha = line_search_tool.alpha_0
    riemann_grad = compute_gradient_projection(func, x_k)

    write_history(func(x_k), riemann_grad.norm())
    iters = 0
    while history["grad_norm"][-1] > tolerance:
        alpha = line_search_tool.line_search(func, x_k, -riemann_grad, rank, 2 * alpha)
        x_k -= alpha * riemann_grad
        x_k = x_k.round(rank)
        riemann_grad = compute_gradient_projection(func, x_k)

        write_history(func(x_k), riemann_grad.norm(qr_based=True))

        if np.isnan(history["func"][-1]) or np.isnan(history["grad_norm"][-1]) or \
                np.isinf(history["func"][-1]) or np.isinf(history["grad_norm"][-1]):
            raise ValueError("Function or riemann gradient norm values became NaN during optimization process")

        iters += 1
        if iters >= max_iter:
            break

    history = history if trace else None
    return x_k, history


def RCG(func: Callable, x_0: Tucker, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False):
    """
        Performs Riemannian conjugate descent optimization method.

        Parameters
        ----------
        func : Callable
            Cost function for minimizing.
        x_0 : Tucker
            Starting point for optimization algorithm. Multilinear rank of x_0 defines rank of riemann manifold, on which
             optimization process is launched.
        tolerance : float
            Epsilon value for stopping criterion.
        max_iter : int
            Maximum number of iterations.
        line_search_options : dict, LineSearchTool or None
            Dictionary with line search options. See LineSearchTool class for details.
        trace : bool
            If True, the progress information is appended into history dictionary during training.
            Otherwise None is returned instead of history.

        Returns
        -------
        x_star : np.array
            The point found by the optimization procedure
        history : dictionary of lists or None
            Dictionary containing the progress information or None if trace=False.
            Dictionary has to be organized as follows:
                - history['func'] : list of function values f(x_k) on every step of the algorithm
                - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm

        Example:
        --------
        >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
        >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
        >> print('Found optimal point: {}'.format(x_opt))
           Found optimal point: [ 0.  1.  2.  3.  4.]
        """
    history = defaultdict(list)

    def write_history(func_val, grad_norm):
        history["func"].append(func_val)
        history["grad_norm"].append(grad_norm)

    line_search_tool = get_line_search_tool(line_search_options)
    x_k = copy(x_0)
    rank = x_k.rank
    alpha = line_search_tool.alpha_0
    riemann_grad = compute_gradient_projection(func, x_k)
    conjugate_direction = -riemann_grad

    write_history(func(x_k), riemann_grad.norm())
    iters = 0
    while history["grad_norm"][-1] > tolerance:
        alpha = line_search_tool.line_search(func, x_k, conjugate_direction, rank, 2 * alpha)
        x_k += alpha * conjugate_direction
        x_k = x_k.round(rank)
        riemann_grad = compute_gradient_projection(func, x_k)

        write_history(func(x_k), riemann_grad.norm(qr_based=True))

        beta = history["grad_norm"][-1] / history["grad_norm"][-2]
        conjugate_direction = -riemann_grad + (beta ** 2) * vector_transport(None, x_k, conjugate_direction)

        if np.isnan(history["func"][-1]) or np.isnan(history["grad_norm"][-1]) or \
                np.isinf(history["func"][-1]) or np.isinf(history["grad_norm"][-1]):
            raise ValueError("Function or riemann gradient norm values became NaN during optimization process")

        iters += 1
        if iters >= max_iter:
            break

    history = history if trace else None
    return x_k, history