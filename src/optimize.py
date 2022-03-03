import numpy as np
from typing import Callable
from collections import defaultdict
from copy import copy

from src.riemopt import compute_gradient_projection
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
        armijo_threshold = self.c1 * alpha * d_k.norm(qr_based=True)
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
