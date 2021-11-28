import numpy as np
import optimization
import oracles
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,4)
import matplotlib.patches as mpatches
import sys
import scipy.sparse
np.random.seed(12345)

def plot_history(history_usual, history_optimized, plot_type,) :
    plt.clf()
    label_usual = label = 'Обычный оракул'
    label_optimized = 'Оптимизированный оракул'
    patch_usual = mpatches.Patch(color = 'red', label = label_usual)
    patch_optimized = mpatches.Patch(color = 'blue', label = label_optimized)
    plot_name = 'optimized_{}.png'.format(plot_type)
    if plot_type == 'func_number':
        plt.plot(history_usual['func'], c = 'red', label = label_usual)
        plt.plot(history_optimized['func'], c = 'blue', label = label_usual)
        plt.xlabel("Номер итерации", fontsize = 10)
        plt.ylabel("Значение функции", fontsize = 10)
    elif plot_type == 'func_time':
        plt.plot(history_usual['time'], history_usual['func'], c = 'red', label = label_usual)
        plt.plot(history_optimized['time'], history_optimized['func'], c = 'blue', label = label_usual)
        plt.xlabel("Время выполнения", fontsize = 10)
        plt.ylabel("Значение функции", fontsize = 10)
    elif plot_type == 'grad_time':
        log_norm_usual = 2 * np.log(history_usual['grad_norm'] / history_usual['grad_norm'][0])
        log_norm_optimized = 2 * np.log(history_optimized['grad_norm'] / history_optimized['grad_norm'][0])
        plt.plot(history_usual['time'], log_norm_usual, c = 'red', label = label_usual)
        plt.plot(history_optimized['time'], log_norm_optimized, c = 'blue', label = label_usual)
        plt.xlabel("Время выполнения", fontsize = 10)
        plt.ylabel(r'$\log\left(\dfrac{\left\|\nabla f(x_k)\right\|^2}{\left\|\nabla f(x_0)\right\|^2}\right)$', fontsize = 10)
    plt.legend(handles = (patch_usual, patch_optimized), fontsize = 10)
    plt.savefig("plots/{}".format(plot_name), dpi = 200, bbox_inches = 'tight')
    #plt.show()
#print(x, y)

def generate_task():
    m = 10000
    n = 8000
    A = np.random.randn(m, n)
    b = np.sign(np.random.randn(m))
    x_0 = np.zeros(n)
    return A, b, x_0

A, b, x_0 = generate_task()
oracle_usual = oracles.create_log_reg_oracle(A, b, 1 / b.size, oracle_type = 'usual')
oracle_optimized = oracles.create_log_reg_oracle(A, b, 1 / b.size, oracle_type = 'optimized')
_, message, history_usual = optimization.gradient_descent(oracle_usual, x_0, tolerance = 1e-7, trace = True)
if message != "success":
    print("Error while testing!")
    sys.exit(1)
_, message, history_optimized = optimization.gradient_descent(oracle_optimized, x_0, tolerance = 1e-7, trace = True)
if message != "success":
    print("Error while testing!")
    sys.exit(1)
plot_history(history_usual, history_optimized, 'func_number')
plot_history(history_usual, history_optimized, 'func_time')
plot_history(history_usual, history_optimized, 'grad_time')
