"""
This example shows a simple implementation of the gradient descent algorithm for minimizing
a simple function: f(x) = x^2 - 4x + 1.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gradient_descent(start, function, gradient, learn_rate, max_iter, tol=0.001):
    """
    Performs gradient descent to minimize a given function.

    Parameters:
    start (float): The starting point for the algorithm.
    function (callable): The function to minimize.
    gradient (callable): The gradient of the function.
    learn_rate (float): The learning rate (step size).
    max_iter (int): The maximum number of iterations.
    tol (float): The tolerance for stopping the algorithm.

    Returns:
    float: The point at which the function is minimized.
    """


    params = np.array(start)

    for it in range(max_iter):
        grads = np.array(gradient(*params))
        step = learn_rate * grads  # Calculate the step size

        if np.linalg.norm(step) < tol:  # Check if the step size is smaller than the tolerance
            print('End by tolerance')
            return params

        print("iteration =", it, ", a0 = {}, a1 = {}, a2 = {}, ".format(*params), "E(a0, a1, a2) = {}".format(function(*params)))

        params = params - step
    return params


def func(a0, a1, a2 = 0):
    """
    The function to minimize.

    Parameters:
    x (float): The input value.

    Returns:
    float: The function value at x.
    """
    E = 0
    for i in range(len(T)):
        E += (P[1][i] - (a0 + a1*T[i] + a2*T[i]**2))**2

    return E

def gradient_func(a0, a1, a2 = 0):
    """
    The gradient of the function.

    Parameters:
    a0 (float): The input value.

    Returns:
    float: The gradient value at x.
    """
    if a2 == 0:
        dE_da2 = 0
    else:
        dE_da2 = -2 * np.sum(T**2 * (P[0] - (a2 * T**2 + a1 * T + a0)))

    dE_da1 = -2 * np.sum(T * (P[0] - (a2 * T**2 + a1 * T + a0)))

    dE_da0 = -2 * np.sum(P[0] - (a2 * T**2 + a1 * T + a0))

    return dE_da0, dE_da1, dE_da2


#Gather cwd and parameters
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

jparams = json.load(open(dir_path +  '/params.json'))
P = jparams["time_series"]
T = np.array([1,2,3,4,5,6])

# Run the gradient descent algorithm starting from x = 9
a0, a1, a2 = gradient_descent([-1, -1, -1], func, gradient_func, 0.0001, 1000)



"""

time = np.linspace(1, 6, 100)
curve = []
for t in time:
    curve.append(a0 + a1*t + a2*t**2)



plt.plot(time, curve)
plt.plot(T, P[0])
plt.show()



# Generate values for alpha and beta
alpha_vals = np.linspace(-400, 400, 400)
beta_vals = np.linspace(-400, 400, 400)
alpha_grid, beta_grid = np.meshgrid(alpha_vals, beta_vals)

# Compute function values
E_vals = np.array([[func(a, b, 30) for a, b in zip(alpha_row, beta_row)]
                    for alpha_row, beta_row in zip(alpha_grid, beta_grid)])

# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(alpha_grid, beta_grid, E_vals, cmap='viridis')

# Labels
ax.set_xlabel("Alpha")
ax.set_ylabel("Beta")
ax.set_zlabel("E(alpha, beta)")
ax.set_title("Surface Plot of E(alpha, beta)")

plt.show()

"""