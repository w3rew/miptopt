import numpy as np
from scipy.sparse import csr_matrix as csr
import oracles
from sys import exit
from random import randint
def my_grad(A, b, x, l):
    m = np.size(b)
    ans = l * x
    for i in range(m):
        y = -b[i] * np.inner(A[i,], x)
        ans -= 1 / m * (np.exp(y)/(1 + np.exp(y)) * b[i] * A[i, ])
    return ans
def my_func(A, b, x, l):
    ans = l / 2 * np.inner(x, x)
    for i in range(m):
        y = -b[i] * np.inner(A[i,], x)
        ans += 1 / m * np.log(1 + np.exp(y))
    return ans



for i in range(10000):
    n = randint(1, 100)
    m = randint(1, 100)
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    x = np.random.rand(n)
    l = randint(1, 10)
    L = oracles.create_log_reg_oracle(A, b, l)
    oracle_grad = L.grad(x)
    oracle_hess = L.hess(x)
    finite_diff_grad = oracles.grad_finite_diff(L.func, x, eps = 1e-7)
    finite_diff_hess = oracles.hess_finite_diff(L.func, x, eps = 1e-5)
    if not np.allclose(oracle_grad, finite_diff_grad, rtol = 0, atol = 2e-2):
        print("Error: oracle grad: {}, finite diff grad: {}, my_grad: {}".format(oracle_grad, finite_diff_grad, my_grad(A, b, x, 2)))
        print(L.func(x), my_func(A, b, x, 2))
        exit(1)
    if not np.allclose(oracle_hess, finite_diff_hess, rtol = 0, atol = 1e-3):
        print("Error: oracle hess: {}, finite diff hess: {}".format(oracle_hess, finite_diff_hess))
        exit(1)

print("Test passed! Grad and hess are ok")
