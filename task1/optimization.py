import numpy as np
from numpy.linalg import LinAlgError
import scipy
import scipy.optimize
from datetime import datetime
from collections import defaultdict
import time
import math


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
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

        if self._method == "Constant":
            return self.c
        if self._method == "Armijo":
            if previous_alpha is None:
                alpha = self.alpha_0
            else:
                alpha = previous_alpha #FIXME: previous_alpha * 2
            return line_search_armijo(oracle, x_k, d_k, self.c1, alpha)

        if self._method == "Wolfe":
            phi = lambda a : oracle.func_directional(x_k, d_k, a)
            phi_der = lambda a : oracle.grad_directional(x_k, d_k, a)
            alpha,_,_,_ = scipy.optimize.linesearch.scalar_search_wolfe2(phi, phi_der, c1 = self.c1, c2 = self.c2)
            if alpha is None:
                if previous_alpha is None:
                    alpha = self.alpha_0
                else:
                    alpha = previous_alpha * 2
                return line_search_armijo(oracle, x_k, d_k, self.c1, alpha)
            return alpha


def line_search_armijo(oracle, x_k, d_k, c, initial_alpha) :
    alpha = initial_alpha
    phi = lambda a : oracle.func_directional(x_k, d_k, a)
    initial_phi = phi(0)
    initial_phi_der = oracle.grad_directional(x_k, d_k, 0)
    while phi(alpha) > initial_phi + c * alpha * initial_phi_der:
        alpha /= 2
    return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options={}, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = LineSearchTool(**line_search_options)
    x = np.copy(x_0)
    initial_gradient_square = np.linalg.norm(oracle.grad(x_0))**2
    start_time = time.perf_counter()
    iters = 0
    alpha = None

    while True:
        if iters > max_iter:
            return x, "iterations_exceeded", history
        gradient = oracle.grad(x)
        if trace:
            history['time'].append(time.perf_counter() - start_time)
            history['func'].append(oracle.func(x))
            history['grad_norm'].append(np.linalg.norm(gradient))
            if x.size <= 2:
                history['x'].append(x)
        alpha = line_search_tool.line_search(oracle, x, -gradient, alpha)
        if not (valid_number(alpha)) :
            return x, "computational_error", None

        if display:
            print("iteration {}: f(x) is {}, grad f norm is {}, we choose alpha = {}"
                    .format(iters, oracle.func(x), np.linalg.norm(gradient), alpha))
        if np.inner(gradient, gradient) <= tolerance * initial_gradient_square:
            if display:
                print("Found singular point: {}, grad norm: {}".format(x, np.linalg.norm(gradient)))
            return x, "success", history

        x = x - alpha * gradient
        iters += 1


def valid_number(x):
    """
    Checks if x is valid number.
    """
    return not (x is None or math.isinf(x) or math.isnan(x))

def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options={}, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_options.pop('alpha_0', None)
    line_search_tool = LineSearchTool(**line_search_options, alpha_0 = 1.)
    x = np.copy(x_0)
    initial_gradient_square = np.inner(oracle.grad(x_0), oracle.grad(x_0))
    start_time = time.perf_counter()
    iters = 0
    alpha = None

    while True:
        gradient = oracle.grad(x)
        if trace:
            history['time'].append(time.perf_counter() - start_time)
            history['func'].append(oracle.func(x))
            history['grad_norm'].append(np.linalg.norm(gradient))
            if x.size <= 2:
                history['x'].append(x)
        if np.inner(gradient, gradient) <= tolerance * initial_gradient_square:
            if display:
                print("Found singular point: {}, grad norm: {}".format(x, np.linalg.norm(gradient)))
            return x, "success", history
        if iters > max_iter:
            return x, "iterations_exceeded", history
        hess = oracle.hess(x)
        try:
            decomposition = scipy.linalg.cho_factor(hess)
        except LinAlgError:
            return x, "computational_error", history
        d = scipy.linalg.cho_solve(decomposition, -gradient)

        alpha = line_search_tool.line_search(oracle, x, d)
        if not (valid_number(alpha)) :
            return x, "computational_error", history

        if display:
            print("iteration {}: f(x) is {}, grad f norm is {}, we choose alpha = {}"
                    .format(iters, oracle.func(x), np.linalg.norm(gradient), alpha))

        x = x + alpha * d
        iters += 1

    return x_k, 'success', history
