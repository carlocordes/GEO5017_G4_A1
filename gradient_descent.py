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

def gradient_descent(start, dim, function, gradient, learn_rate, max_iter, tol):
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
        #print("Iteration: {} - Shape parameters: {}, {}, {} - E = {}".format(it, *params, func(dim, *params)))


        if (np.linalg.norm(steps, len(start)) < tol):
            print("Exceeding tolerance after {} iterations, parameters: {},{},{}, E = {}".format(it, *params, func(dim, *params)))
            return params, param_trace


        #Collect parameters of iteration
        it_params = []
        it_params.extend(params)
        it_params.append(function(dim, *params))
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

# Perform gradient descent for separate dimensions
time = np.linspace(1, 7, 100)
reconstructed = []



for dimension in P:
    parameters, trace = gradient_descent([jparams["a0"], jparams["a1"], jparams["a2"]],
                                         dimension, func, gradient_func,
                                         jparams["learning_rate"],
                                         jparams["max_iter"],
                                         tol = jparams["tolerance"])

    # Reconstructing curve with obtained parameters

    curve = []
    for t in time:
        curve.append(float(parameters[0] + parameters[1]*t + parameters[2]*t**2))

    reconstructed.append(curve)


# Plot results
fig = plt.figure(figsize=(18, 4))
fig.suptitle("Polynomial Regression", fontsize=20)

axes = [fig.add_subplot(1, 4, i) for i in range(1, 4)]
axes.append(fig.add_subplot(1, 4, 4, projection='3d'))

for i in range(3):
    axes[i].plot(time, reconstructed[i], label='Regression')
    axes[i].plot(T, P[i], label='Data')
    axes[i].set_xlabel('Time / s')
    axes[i].set_ylabel('Spatial dimension {}'.format(i+1))
    axes[i].legend()

axes[3].plot(*P, label='Data')
axes[3].plot(*reconstructed, label='Regression')
axes[3].legend()

plt.show()
