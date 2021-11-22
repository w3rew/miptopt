import sklearn.datasets
import numpy as np
import sys
from oracles import create_log_reg_oracle
import optimization

if len(sys.argv) < 3:
    print("Provide dataset name and [newton/grad]!")
    sys.exit(1)

name = sys.argv[1]
method = sys.argv[2]
X, y = sklearn.datasets.load_svmlight_file(name)
m = len(y)
n = X.shape[1]
oracle = create_log_reg_oracle(X, y, 1 / m)
tolerance = 1e-10
x_0 = np.zeros(n)
if method == 'newton':
    trained, message, history = optimization.newton(oracle, x_0, tolerance, trace = True, display = True, max_iter = 1000000)
elif method == 'grad':
    trained, message, history = optimization.gradient_descent(oracle, x_0, tolerance, trace = True, display = True)
else:
    print('Unknown method:', method)
    sys.exit(1)

if message != 'success':
    print('Error!:', message)
    sys.exit(1)

np.save("{}_{}_trained".format(name, method), trained)
np.save("{}_{}_history".format(name, method), history)

