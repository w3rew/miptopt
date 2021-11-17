import numpy as np
from oracles import QuadraticOracle
from optimization import gradient_descent

oracle = QuadraticOracle(np.eye(5), np.arange(5))
x_opt, message, history = gradient_descent(oracle, np.zeros(5),
        line_search_options={'method': 'Wolfe', 'c1': 1e-2, 'c2' : 0.9}, max_iter = 100000, tolerance = 1e-3, display = True)

print(x_opt, message, history)

