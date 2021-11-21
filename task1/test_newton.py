import numpy as np
import optimization
from oracles import QuadraticOracle
oracle = QuadraticOracle(np.eye(5), np.arange(5))
x_opt, message, history = optimization.newton(oracle, np.zeros(5), display = True)#, line_search_options={'method': 'Constant', 'c': 1.0})
print('Found optimal point: {}'.format(x_opt))
