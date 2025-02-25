import os
import json
import numpy as np
import matplotlib.pyplot as plt


def func(dim, a0, a1, a2):
    """
    The function to minimize.

    Parameters:
    a0:
    a1:
    a2: Shape parameters
    dim: position values
    Returns:
    E: Objective function
    """
    E = 0
    for i in range(len(T)):
        E += ((dim[i] - (a0 + a1*T[i] + a2*T[i]**2))**2)

    return E

def gradient_func(dim, a0, a1, a2 = 0):
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
        dE_da2 = -2 * np.sum(T**2 * (dim - (a2 * T**2 + a1 * T + a0)))

    dE_da1 = -2 * np.sum(T * (dim - (a2 * T**2 + a1 * T + a0)))

    dE_da0 = -2 * np.sum(dim - (a2 * T**2 + a1 * T + a0))

    return dE_da0, dE_da1, dE_da2



def gradient_descent(start, dim, function, gradient, learn_rate, max_iter, tol=0.00001):
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
    param_trace = [] # Saving all parameters for visual representation

    params = np.zeros(3)
    for i in range(len(start)):
        params[i] = start[i]

    for it in range(max_iter):
        grads = np.array(gradient(dim, *params))
        steps = learn_rate * grads
        print("Iteration: {} - Shape parameters: {}, {}, {} - E = {}".format(it, *params, func(dim, *params)))


        if (np.linalg.norm(steps, len(start)) < tol):
            print("Terminating. Exceeding minimum tolerance after {} iterations".format(it))
            return params, param_trace


        #Collect parameters of iteration
        it_params = []
        it_params.extend(params)
        it_params.append(func(dim, *params))
        it_params.append(it)

        param_trace.append(it_params)
        params -= steps

    return params, param_trace

########################################

#Gather cwd and parameters
dir_path = os.path.dirname(os.path.realpath(__file__))

jparams = json.load(open(dir_path +  '/params.json'))
P = jparams["time_series"]
T = np.array([1,2,3,4,5,6])


parameters, trace = gradient_descent([0, 1, 1], P[1], func, gradient_func, 0.0002, 10000)

print(parameters)

time = np.linspace(1, 6, 100)
curve = []
for t in time:
    curve.append(parameters[0] + parameters[1]*t + parameters[2]*t**2)


plt.plot(time, curve)
plt.plot(T, P[1])
plt.show()

"""
### Regenerating grid for visual comparison
a0s = np.linspace(-10, 10, 21)
a1s = np.linspace(-10, 10, 21)

A0, A1 = np.meshgrid(a0s, a1s, indexing='ij')
Z = func(P[0], A0, A1, 0)

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(projection='3d')
ax1.plot_surface(A0, A1, Z, cmap='viridis', alpha = 0.6)
for point in trace:
    ax1.scatter(point[0], point[1], point[3], c = 'red')
ax1.set_xlabel('a0')
ax1.set_ylabel('a1')
ax1.set_zlabel('Function Value')
ax1.set_title('3D Surface Plot of func (a0, a1)')
plt.show()
"""