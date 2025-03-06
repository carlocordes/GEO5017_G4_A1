import numpy as np
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt

# Initialization
data_pts = np.array([
    [2, 1.08, -0.83, -1.97, -1.31, 0.57],   # X coordinates
    [0, 1.68, 1.82, 0.28, -1.51, -1.91],    # Y coordinates
    [1, 2.38, 2.49, 2.15, 2.59, 4.32],      # Z coordinates
])

t = np.array([1,2,3,4,5,6])

# Extract coordinates and time
x = data_pts[0]     # x,y,z,t is an array
y = data_pts[1]
z = data_pts[2]

print("X:", x)
print("Y:", y)
print("Z:", z)
print("Time:", t)
print(type(x))

# def func1(dim, a0, a1):
#     """
#
#     Args:
#         dim: XYZ position values, an array
#         a0: starting point, int/float
#         a1: speed - first derivative, int/float
#
#     Returns:
#         E: residual error
#     """
#     E = 0                                     # initialize Objective function
#     for i in range(len(t)):                   #loop through time t
#         E += (dim[i] - (a0 +a1*t[i]))**2      # sum of squared errors/objective function
#     # print(f"E = {E}")
#     return E

def func(dim, a0, a1):
    """

    Args:
        dim: XYZ position values, an array
        a0: starting point, int/float
        a1: speed - first derivative, int/float

    Returns:
        E: residual error
    """
    E = np.sum((dim - (a0 + a1*t))**2)   # actual position (dim) - guessed position ((a0 + a1*t))**2)
    # print(f"E = {E}")
    return E

def gradient_func(dim, a0, a1):
    """

    Args:
        dim:
        a0:
        a1:

    Returns:
        dE_da0 and dE_da1: The gradient value or the value of starting point and speed
    """
    dE_da0 = -2 * np.sum(dim-(a0 + a1*t))           # dE_da0 = gradient of a0 -> starting point
    dE_da1 = -2 * np.sum((dim-(a0 + a1*t)) * t)     # dE_da1 = gradient of a1 -> speed

    # print(f"dE_da0 = {dE_da0}, dE_da1 = {dE_da0}")
    return dE_da0, dE_da1

def gradient_descent(start, dim, function, gradient, learn_rate, max_iter, tol):
    """
    Args:
        start: list of initial guess of starting point (a0) and speed (a1): [a0, a1]
        dim: array of position value: X, Y, Z, example X: [2, 1.08, -0.83, -1.97, -1.31, 0.57].
        function: function that calculates residual error (E)
        gradient: function that calculates how much to adjust the "initial" starting point and speed
        learn_rate: float, controls how big the step
        max_iter: int, number of iteration
        tol: float

    Returns:
        params: updated params [a0, a1]
        last_E: last residual error (for printing)
        last_it: last iteration (for printing)

    """
    params = np.array(start, dtype=float)       # [a0,a1]
    last_E = 0     # initialize E for each dimension
    for it in range(max_iter):
        # Calculate gradients, grads = array to hold gradients of a0 and a1
        grads = np.array(gradient_func(dim, *params))   # *params (*) is unpack operator to store a0 and a1 individually instead of array of params
        # Update params using gradient descent formula
        params = params - learn_rate*grads
        # Calculate residual error
        E = func(dim, *params)
        print(f"Iteration: {it} \t params: {params} \t E: {E} \t grads: {grads}")
        # Extract last residual error and last iteration for printing
        last_E = E
        last_it = it
        # Check for convergence
        if np.linalg.norm(grads) < tol:
            print("Yes, it converges")
            break
    return params, last_E, last_it


# Run Gradient Descent function for all 3 dimensions: X,Y,Z
learn_rate = 0.0002
max_iter = 100000
tol = 0.00001
x_params, x_error, x_it = gradient_descent([1,1],x,func, gradient_func,learn_rate,max_iter,tol)
# x2_params = gradient_descent([0,0],x,func, gradient_func,learn_rate,max_iter,tol)
y_params, y_error, y_it = gradient_descent([1,1],y,func, gradient_func,learn_rate,max_iter,tol)
z_params, z_error, z_it = gradient_descent([1,1],z,func, gradient_func,learn_rate,max_iter,tol)


# Extracting and printing final parameters
x_start, x_speed = x_params
# x2_start, x2_speed = x2_params
y_start, y_speed = y_params
z_start, z_speed = z_params

print("\nFinal Results:")
print(f"x_start: {x_start:.4f} and x_speed: {x_speed:.4f}, residual error = {x_error:.3f}, iteration={x_it}")
# print(f"x2_start: {x2_start:.4f} and x2_speed: {x2_speed:.4f}")
print(f"y_start: {y_start:.4f} and y_speed: {y_speed:.4f}, residual error = {y_error:.3f}, iteration={y_it}")
print(f"z_start: {z_start:.4f} and z_speed: {z_speed:.4f}, residual error = {z_error:.3f}, iteration={z_it}")

# Plot the result
x_pred = x_start + x_speed * t
y_pred = y_start + y_speed * t
z_pred = z_start + z_speed * t


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, color='red', label='Actual data')
ax.plot(x_pred, y_pred, z_pred, color='blue', label='Speed model prediction')
ax.set_title('Constant Speed Model')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.legend()
plt.show()

print("\n------------------- PRINT TO CHECK DELETE LATER --------------------------------\n")

# # Call the function
# # Px
# func1(x, 0,0)
# grads = gradient_func(x, 2, 3)
# print("grads: ", grads)
# print("grads: ", *grads)
# print(type(grads))
# #
# # # Py
# func2(y, 2,3)
# gradient_func(y, 2, 3)
# #
# # # Pz
# func1(z, 2,3)
# gradient_func(z, 2, 3)
#
# print("Checking with the type and dimension of dim")
# a=x-(0 + 1*t)
# print("a: ", a)

















