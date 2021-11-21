import numpy as np
import optimization
from oracles import QuadraticOracle as quadratic_oracle
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,4)
import matplotlib.patches as mpatches
import sys
import scipy.sparse
def generate_task(n, k):
    d = np.random.randint(1, k + 1, n)
    d[0] = 1
    d[1] = k
    A = scipy.sparse.diags(d)
    b = np.random.rand(n) * k
    x_0 = np.random.rand(n) * k
    return A, b, x_0
n = 10
N = 5
lso = {'method' : 'Wolfe', 'c1' : 1e-4, 'c2' : 0.9}
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
handles = []
for log_n in range(N):
    label = "n = {}".format(n)
    for i in range(5):
        x = []
        y = []
        for k in range(1, 21):
            x.append(k)
            A, b, x_0 = generate_task(n, k)
            oracle = quadratic_oracle(A, b)
            _, message, history = optimization.gradient_descent(oracle, x_0, tolerance = 1e-7, line_search_options = lso, trace = True)
            print(history)
            if message != "success":
                print("Error while testing!")
                sys.exit(1)
            y.append(len(history['func']))
        plt.plot(x, y, '-', c = colors[log_n], label = log_n)
    patch = mpatches.Patch(color = colors[log_n], label = label)
    handles.append(patch)
    n *= 10
plt.xlabel("Число обусловленности k", fontsize = 10)
plt.ylabel("Количество итераций градиентного спуска", fontsize = 10)
plt.legend(handles = handles, fontsize = 10)
plt.savefig("plots/perf_gradients.png", dpi = 200, bbox_inches = 'tight')
#plt.show()
#print(x, y)


