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

def func1(dim, a0, a1):
    """

    Args:
        dim: XYZ position values, an array
        a0: starting point, int/float
        a1: speed - first derivative, int/float

    Returns:
        E: residual error
    """
    E = 0                                     # initialize Objective function
    for i in range(len(t)):                   #loop through time t
        E += (dim[i] - (a0 +a1*t[i]))**2      # sum of squared errors/objective function
    # print(f"E = {E}")
    return E

def func2(dim, a0, a1):
    """

    Args:
        dim: XYZ position values, an array
        a0: starting point, int/float
        a1: speed - first derivative, int/float

    Returns:
        E: residual error
    """
    E = np.sum(dim - (a0 + a1*t))**2            # actual position (dim) - guessed position ((a0 + a1*t))**2)
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

    """
    params = np.array(start, dtype=float)
    for it in range(max_iter):
        # Calculate gradients, grads = array to hold gradients of a0 and a1
        grads = np.array(gradient_func(dim, *params))   # *params (*) is unpack operator to store a0 and a1 individually instead of array of params
        # Update params using gradient descent formula
        params = params - learn_rate*grads
        # Calculate residual error
        E = func1(dim, *params)
        print(f"Iteration: {it} \t params: {params} \t E: {E} \t grads: {grads}")
        # Check for convergence
        if np.linalg.norm(grads) < tol:
            break
    return params


# Run Gradient Descent function for all 3 dimensions: X,Y,Z
learn_rate = 0.0001
max_iter = 10000
tol = 0.0001
x_params = gradient_descent([0,0],x,func1, gradient_func,learn_rate,max_iter,tol)
x2_params = gradient_descent([0,0],x,func2, gradient_func,learn_rate,max_iter,tol)
y_params = gradient_descent([0,0],y,func1, gradient_func,learn_rate,max_iter,tol)
z_params = gradient_descent([0,0],z,func1, gradient_func,learn_rate,max_iter,tol)


# Extracting and printing final parameters
x_start, x_speed = x_params
x2_start, x2_speed = x2_params
y_start, y_speed = y_params
z_start, z_speed = z_params

print("\nFinal Results:")
print(f"x_start: {x_start:.4f} and x_speed: {x_speed:.4f}")
print(f"x2_start: {x2_start:.4f} and x2_speed: {x2_speed:.4f}")
print(f"y_start: {y_start:.4f} and y_speed: {y_speed:.4f}")
print(f"x_start: {z_start:.4f} and x_speed: {z_speed:.4f}")


print("\n------------------- PRINT TO CHECK DELETE LATER --------------------------------\n")

# Call the function
# Px
func1(x, 0,0)
grads = gradient_func(x, 2, 3)
print("grads: ", grads)
print("grads: ", *grads)
print(type(grads))
#
# # Py
func2(y, 2,3)
gradient_func(y, 2, 3)
#
# # Pz
func1(z, 2,3)
gradient_func(z, 2, 3)

print("Checking with the type and dimension of dim")
a=x-(0 + 1*t)
print("a: ", a)



















#--------------------------------------------------------------------------------------------------------------
# def gradient_descent(start, dim, function, gradient, learn_rate, max_iter, tol=0.00001):
#     params = np.array(start, dtype=float)  # [a0, a1]
#     for it in range(max_iter):
#         grads = np.array(gradient(dim, *params))
#         params -= learn_rate * grads
#         E = function(dim, *params)
#         if np.linalg.norm(grads) < tol:
#             break
#     return params
#
# x_params = gradient_descent([0, 0], data_pts[0], func, gradient_func, 0.01, 100)
# y_params = gradient_descent([0, 0], data_pts[1], func, gradient_func, 0.01, 100)
# z_params = gradient_descent([0, 0], data_pts[2], func, gradient_func, 0.01, 100)
#
# x_start, x_speed = x_params
# y_start, y_speed = y_params
# z_start, z_speed = z_params
#
# print("\nFinal Results:")
# print(f"X: Starting Point = {x_start:.2f}, Speed = {x_speed:.2f}")
# print(f"Y: Starting Point = {y_start:.2f}, Speed = {y_speed:.2f}")
# print(f"Z: Starting Point = {z_start:.2f}, Speed = {z_speed:.2f}")
#
# # 📌 8. Plot the results.
# time = np.linspace(1, 6, 100)
# x_pred = x_start + x_speed * time
# y_pred = y_start + y_speed * time
# z_pred = z_start + z_speed * time
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data_pts[0], data_pts[1], data_pts[2], color='red', label='Actual Data')
# ax.plot(x_pred, y_pred, z_pred, color='blue', label='Speed Model Prediction')
# ax.set_title('Constant Speed Model for Drone Trajectory')
# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')
# ax.set_zlabel('Z Position')
# ax.legend()
# plt.show()