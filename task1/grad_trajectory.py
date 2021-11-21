#!/usr/bin/bash
import numpy as np
import optimization
from oracles import QuadraticOracle as qoracle
from plot_trajectory_2d import plot_levels, plot_trajectory
import matplotlib.pyplot as plt
import sys
matrices = [
np.diag([1, 1.5]),
np.diag([1, 100]),
np.diag([10, 1]),
]
x_ans = np.array([1, 1])
x_0 = np.array([-1, -1])
def plot_matrix(num):
    A = matrices[num]
    b = A @ x_ans
    oracle = qoracle(A, b)
    ans = optimization.gradient_descent(oracle, x_0, tolerance = 1e-10, max_iter = 10000, trace = True, line_search_options =
            {'method': 'Armijo', 'c':0.9})
    plot_levels(oracle.func, xrange = [-1.5, 3], yrange = [-1.5, 3], levels = np.linspace(oracle.func(x_ans), oracle.func(x_0), 15))
    plot_trajectory(oracle.func, ans[2]['x'], fit_axis = False)
    plt.show()

if len(sys.argv) < 2:
    sys.exit(1)
num = int(sys.argv[1])
if num > 2:
    sys.exit(1)
plot_matrix(num)
