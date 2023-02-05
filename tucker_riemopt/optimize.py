import numpy as np

from typing import Callable
from collections import defaultdict
from copy import copy

from tucker_riemopt.riemopt import compute_gradient_projection, vector_transport
from tucker_riemopt.tucker import Tucker, TangentVector, ML_rank

class OptimizationConfig:
    """
    Class contains bunch of fields for setting up riemann optimization methods.
    By default optimization config has following fields and set ups:
        maxiter : 100
            Amount of iterations method can pass until it will be stopped
        line_search_options : None
            See class `LineSearchTool` for more information
        method_trace : False
            If true, method will return history, contains function values and
            riemann gradient norms per iter
        tolerance : 1e-5
            Desired tolerance of method
        CG_momentum_method : "FR"
            There is a momentum coeff in riemann CG method, that is responsible for
            contribution of previous conjugate direction in current search direction.
            There are two methods for choosing this coeff possible:
                "FR" (default) : Fletcher-Reeves update
                "PR+" : Polak-Ribiere+ update
        early_stopping_criteria : None
            Whether method should stop, if function value on current solution is little different
            from function value on previous solution. Formally:
            if func(x_{k-1}) - func(x_k) < early_stopping_criteria => stop.
            If None is provided, then early stopping procedure is not applied.
    """
    def __init__(self, **kwargs):
        self.maxiter = kwargs.get("maxiter", 100)
        self.line_search_options = kwargs.get("line_search_options", None)
        self.method_trace = kwargs.get("method_trace", False)
        self.tolerance = kwargs.get("tolerance", 1e-5)
        self.CG_momentum_method = kwargs.get("CG_momentum_method", "FR")
        self.early_stopping_criteria = kwargs.get("early_stopping_criteria", None)

class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Armijo', 'Constant' or 'Custom'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
            - 'Custom' -- You may provide your own line search method, which for instance
                            inferred via analytical minimization. In this case you should provide
                            function, which will calculate step size based on all known reasonable
                            arguments.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            alpha_0 : The step size which is returned on every step.
        If method == 'Custom':
            alpha_0 : if param `prev_alpha` for first iteration (see next param description). By default alpha_0 = 1
            f : function, which gains following arguments (in order)
                x_k : last calculated solution
                g_k : riemann gradient
                d_k : search direction (it might be riemann anti-gradient, or conjugate direction)
                rank : solution is being searched from riemann manifold of this rank
                prev_alpha : previous step size (alpha_0 on first iteration)
            and calculates next step size.
            adjust : if True, calculated step size will be initial guess for Armijo backtracking. Default is False
    """
    def __init__(self, method='Armijo', **kwargs):
        self._method = method
        self.alpha_0 = kwargs.get('alpha_0', 1.0)
        if self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.max_iter = kwargs.get('max_iter', 30)
        elif self._method == 'Custom':
            self.custom_func = kwargs.get('f', lambda *args: self.alpha_0)
            self.adjust = kwargs.get('adjust', False)
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
        iters = 0
        while fx - func((x_k + alpha * d_k).round(rank)) < armijo_threshold:
            alpha /= 2
            armijo_threshold /= 2
            iters += 1
            if iters > self.max_iter:
                return alpha
        return alpha

    def line_search(self, func: Callable[[TangentVector], Tucker], x_k: Tucker, g_k: Tucker, d_k: TangentVector, rank: ML_rank, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        func : Callable[[TangentVector], Tucker]
            Cost function for minimizing.
        x_k : Tucker
            Starting point
        g_k : Tucker
            Riemann gradient
        d_k : Tucker
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
            alpha = self.alpha_0
        if self._method == "Custom":
            alpha = self.custom_func(x_k, g_k, d_k, rank, previous_alpha)
            if self.adjust:
                alpha = self.__armijo(func, x_k, d_k, rank, alpha)

        return alpha

def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def RGD(func: Callable, x_0: Tucker, config = OptimizationConfig()):
    """
    Performs Riemannian steepest descent optimization method.

    Parameters
    ----------
    func : Callable
        Cost function for minimizing.
    x_0 : Tucker
        Starting point for optimization algorithm. Multilinear rank of x_0 defines rank of riemann manifold, on which
         optimization process is launched.
    config : OptimizationConfig
        Optimization config with method set ups. See `OptimizationConfig` for more information.

    Returns
    -------
    x_star : Tucker
        The point found by the optimization procedure
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values riemann gradient norms
    """
    history = defaultdict(list)
    def write_history(func_val, grad_norm):
        history["func"].append(func_val)
        history["grad_norm"].append(grad_norm)

    line_search_tool = get_line_search_tool(config.line_search_options)
    x_k = copy(x_0)
    rank = x_k.rank
    alpha = line_search_tool.alpha_0
    riemann_grad = compute_gradient_projection(func, x_k)

    write_history(func(x_k), riemann_grad.norm())
    iters = 0
    while history["grad_norm"][-1] > config.tolerance:
        alpha = line_search_tool.line_search(func, x_k, riemann_grad, -riemann_grad, rank, 2 * alpha)
        x_k -= alpha * riemann_grad
        x_k = x_k.round(rank)
        riemann_grad = compute_gradient_projection(func, x_k)

        write_history(func(x_k), riemann_grad.norm(qr_based=True))
        if config.early_stopping_criteria and history["func"][-2] - history["func"][-1] < config.early_stopping_criteria:
            break

        if np.isnan(history["func"][-1]) or np.isnan(history["grad_norm"][-1]) or \
                np.isinf(history["func"][-1]) or np.isinf(history["grad_norm"][-1]):
            raise ValueError("Function or riemann gradient norm values became NaN during optimization process")

        iters += 1
        if iters >= config.maxiter:
            break

    history = history if config.method_trace else None
    return x_k, history


def RCG(func: Callable, x_0: Tucker, config = OptimizationConfig()):
    """
        Performs Riemannian conjugate descent optimization method.

        Parameters
        ----------
        func : Callable
            Cost function for minimizing.
        x_0 : Tucker
            Starting point for optimization algorithm. Multilinear rank of x_0 defines rank of riemann manifold, on which
             optimization process is launched.
        config : OptimizationConfig
            Optimization config with method set ups. See `OptimizationConfig` for more information.

        Returns
        -------
        x_star Tucker
            The point found by the optimization procedure
        history : dictionary of lists or None
            Dictionary containing the progress information or None if trace=False.
            Dictionary has to be organized as follows:
                - history['func'] : list of function values f(x_k) on every step of the algorithm
                - history['grad_norm'] : list of values riemann gradient norms
        """
    history = defaultdict(list)

    def write_history(func_val, grad_norm):
        history["func"].append(func_val)
        history["grad_norm"].append(grad_norm)

    line_search_tool = get_line_search_tool(config.line_search_options)
    x_k = copy(x_0)
    rank = x_k.rank
    alpha = line_search_tool.alpha_0
    riemann_grad = compute_gradient_projection(func, x_k)
    conjugate_direction = -riemann_grad

    write_history(func(x_k), riemann_grad.norm())
    iters = 0
    while history["grad_norm"][-1] > config.tolerance:
        alpha = line_search_tool.line_search(func, x_k, riemann_grad, conjugate_direction, rank, 2 * alpha)
        x_k += alpha * conjugate_direction
        x_k = x_k.round(rank)
        riemann_grad = compute_gradient_projection(func, x_k)

        write_history(func(x_k), riemann_grad.norm(qr_based=True))
        if config.early_stopping_criteria and history["func"][-2] - history["func"][-1] < config.early_stopping_criteria:
            break

        beta = history["grad_norm"][-1] / history["grad_norm"][-2]
        conjugate_direction = -riemann_grad + (beta ** 2) * vector_transport(None, x_k, conjugate_direction)

        if np.isnan(history["func"][-1]) or np.isnan(history["grad_norm"][-1]) or \
                np.isinf(history["func"][-1]) or np.isinf(history["grad_norm"][-1]):
            raise ValueError("Function or riemann gradient norm values became NaN during optimization process")

        iters += 1
        if iters >= config.maxiter:
            break

    history = history if config.method_trace else None
    return x_k, history