import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,4)
#plt.rcParams["text.usetex"] = True
import matplotlib.patches as mpatches
import sys

if len(sys.argv) < 3:
    print("Arguments: dataset [func|grad]")

dataset = sys.argv[1]
grad_file = "{}_grad_history.npy".format(dataset)
newton_file = "{}_newton_history.npy".format(dataset)

plot_func = (sys.argv[2] == 'func')

grad_history = np.load(grad_file, allow_pickle = True).item()
newton_history = np.load(newton_file, allow_pickle = True).item()
if plot_func:
    grad_data = grad_history['func']
    newton_data = newton_history['func']
    plt.ylabel("Значение функции", fontsize = 10)
else:
    grad_data = 2 * np.log(grad_history['grad_norm'] / grad_history['grad_norm'][0])
    newton_data = 2 * np.log(newton_history['grad_norm'] / newton_history['grad_norm'][0])
    plt.ylabel(r'$\log\left(\dfrac{\left\|\nabla f(x_k)\right\|^2}{\left\|\nabla f(x_0)\right\|^2}\right)$', fontsize = 10)

plt.plot(grad_history['time'], grad_data, c = 'blue')
grad_patch = mpatches.Patch(color = 'blue', label = 'Gradient descent')
plt.plot(newton_history['time'], newton_data, c = 'red')
newton_patch = mpatches.Patch(color = 'red', label = 'Newton method')
plt.xlabel("Время работы, с", fontsize = 10)
plt.legend(handles = (grad_patch, newton_patch), fontsize = 10)
#plt.show()

plt.savefig("{}_plot_{}.png".format(dataset, sys.argv[2]), dpi = 200, bbox_inches = 'tight')


