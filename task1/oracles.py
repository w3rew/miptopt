import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.b_sqr = b * b
        self.m = np.size(b)
        self.regcoef = regcoef

    def func(self, x):
        y = -self.b * self.matvec_Ax(x)
        return 1 / self.m * np.sum(np.logaddexp(np.zeros(self.m), y)) + self.regcoef / 2 * np.inner(x, x)

    def grad(self, x):
        y = -self.b * self.matvec_Ax(x)
        return -self.matvec_ATx(self.b * scipy.special.expit(y)) / self.m + self.regcoef * x

    def hess(self, x):
        n = np.size(x)
        y = -self.b * self.matvec_Ax(x)
        sp = scipy.special.expit(-y)
        vec = self.b_sqr * sp * sp * np.exp(y)
        return self.regcoef * np.eye(n) + 1 / self.m * self.matmat_ATsA(vec)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.x = None
        self.Ad = None
        self.use_forwarded = False
        self.Ax = None
        self.x_0 = None
        self.Ax_0 = None
        def cached_matvec_Ax(x):
            if self.use_forwarded:
                return self.Ax
            if np.array_equal(x, self.x):
                return self.Ax
            return matvec_Ax(x)

        super().__init__(cached_matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
    #    self.ATx = None
    #    self.

    def func_directional(self, x, d, alpha):
        print(self.x_0, x)
        if not np.array_equal(self.x_0, x):
            self.x_0 = x
            if not np.array_equal(x, self.x):
                self.Ax = self.matvec_Ax(x)
                self.x = x
            self.Ax_0 = self.Ax
            self.Ad = self.matvec_Ax(d)
        self.x = self.x_0 + alpha * d
        self.Ax = self.Ax_0 + alpha * self.Ad
        self.use_forwarded = True
        res = super().func_directional(x, d, alpha)
        self.use_forwarded = True
        return res


    def grad_directional(self, x, d, alpha):
        if not np.array_equal(self.x_0, x):
            self.x_0 = x
            if not np.array_equal(x, self.x):
                self.Ax = self.matvec_Ax(x)
                self.x = x
            self.Ax_0 = self.Ax
            self.Ad = self.matvec_Ax(d)
        self.x = self.x_0 + alpha * d
        self.Ax = self.Ax_0 + alpha * self.Ad
        self.use_forwarded = True
        res = super().grad_directional(x, d, alpha)
        self.use_forwarded = True
        return res



def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A @ x
    matvec_ATx = lambda x: A.transpose() @ x

    if type(A) == scipy.sparse.csr_matrix:
        def matmat_ATsA(s):
            return A.transpose() @ scipy.sparse.diags(s) @ A
    else:
        def matmat_ATsA(s):
            return A.transpose() @ np.diag(s) @ A


    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = np.size(x)
    grad = [(func(x + eps * np.array([1 if j == i else 0 for j in range(n)])) -
        func(x - eps * np.array([1 if j == i else 0 for j in range(n)]))) / (2 * eps) for i in range(n)]
    return np.array(grad)


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i)
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    def base_vec(elem, n):
        return np.array([1 if i == elem else 0 for i in range(n)])
    n = np.size(x)
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            base_i = base_vec(i, n)
            base_j = base_vec(j, n)
            res[i, j] = (func(x + eps * base_i + eps * base_j) -
                    func(x + eps * base_i) - func(x + eps * base_j) + func(x)) / eps / eps
    return res
