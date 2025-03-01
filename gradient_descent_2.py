import numpy as np
import matplotlib.pyplot as plt

# Initialisation
data_pts = np.array([
    [2, 1.08, -0.83, -1.97, -1.31, 0.57],
    [0, 1.68, 1.82, 0.28, -1.51, -1.91],
    [1, 2.38, 2.49, 2.15, 2.59, 4.32],
    [1, 2, 3, 4, 5, 6]
])
x = data_pts[0]
y = data_pts[1]
z = data_pts[2]
t = data_pts[3]

# Tune parameters here
params = {
    "time_dim":t,
    "learn_rate":0.001,
    "max_iter":100000,
    "tol":0.0001
}

# 1) Plotting trajectory of data points
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, cmap='ocean', c=t, marker='o', label='Points')
ax.plot(x, y, z, color='blue')
ax.legend()

def gradient_descent_speed(pts_dim, params):
    time_dim = params["time_dim"]
    learn_rate = params["learn_rate"]
    max_iter = params["max_iter"]
    tol = params["tol"]
    # Formula of constant speed: Position = Start + Speed * Time
    # Initialise random parameters
    speed = np.random.randn()
    start = np.random.randn()
    for i in range(max_iter):
        estimate = time_dim * speed + start
        error = estimate - pts_dim
        E = np.mean(error**2)

        grad_start = 2 * np.mean(error)
        grad_speed = 2 * np.mean(error * time_dim)
        if np.abs(grad_start * learn_rate) < tol and np.abs(grad_speed * learn_rate) < tol:
            print(f'Converged after {i} iterations.')
            print(f'Residual error: {E}')
            break

        start = start - learn_rate * grad_start
        speed = speed - learn_rate * grad_speed

        # if i % 100 == 0:
        #     print(f'{i}th iteration, loss = {E}, start = {start}, speed = {speed}')
    print(f'Best estimates for start: {start}, speed : {speed}.')
    return start, speed

def gradient_descent_acceleration(pts_dim, params):
    time_dim = params["time_dim"]
    learn_rate = params["learn_rate"]
    max_iter = params["max_iter"]
    tol = params["tol"]

    # Formula for constant acceleration: start + speed * time + 0.5 * acceleration * time**2
    # Initialise random parameters
    # Possible to also initialise at 0 - ie: take t=1 as starting points with 0 speed/acceleration
    speed = np.random.randn()
    start = np.random.randn()
    acc = np.random.randn()

    for i in range(max_iter):
        estimate = start + speed * time_dim + 0.5 * acc * time_dim**2
        error = estimate - pts_dim
        E = np.mean(error**2)

        grad_start = 2 * np.mean(error)
        grad_speed = 2 * np.mean(error * time_dim)
        grad_acc = 2 * np.mean(error * 0.5 * time_dim**2)

        if (np.abs(grad_start * learn_rate) < tol and
                np.abs(grad_speed * learn_rate) < tol and
                np.abs(grad_acc * learn_rate) < tol):
            print(f'Converged after {i} iterations.')
            print(f'Residual error: {E}')
            break
        start = start - learn_rate * grad_start
        speed = speed - learn_rate * grad_speed
        acc = acc - learn_rate * grad_acc
    print(f'Best estimates for start: {start}, speed: {speed}, acceleration: {acc}.')
    return start, speed, acc

# 2(a) Constant Speed
print("Constant Speed estimates (X,Y,Z):")
x_start_a, x_speed_a = gradient_descent_speed(x, params)
y_start_a, y_speed_a = gradient_descent_speed(y,params)
z_start_a, z_speed_a = gradient_descent_speed(z, params)

# 2(b) Constant Acceleration
print("Constant Acceleration estimates (X,Y,Z):")
x_start_b, x_speed_b, x_acc_b = gradient_descent_acceleration(x, params)
y_start_b, y_speed_b, y_acc_b = gradient_descent_acceleration(y, params)
z_start_b, z_speed_b, z_acc_b = gradient_descent_acceleration(z, params)

# 2(c) Position at t=7
x7 = x_start_b + x_speed_b * 7 + 0.5 * x_acc_b * 7**2
y7 = y_start_b + y_speed_b * 7 + 0.5 * y_acc_b * 7**2
z7 = z_start_b + z_speed_b * 7 + 0.5 * z_acc_b * 7**2
print("Estimated position at t=7, given constant acceleration: \n"
      f"X = {x7}, Y = {y7}, Z = {z7}")

x1 = np.append(x, x7)
y1 = np.append(y, y7)
z1 = np.append(z, z7)
t1 = np.append(t, 7)

ax1 = fig.add_subplot(projection='3d')
ax1.scatter(x1, y1, z1, cmap='ocean', c=t1, marker='o', label='Points')
ax1.plot(x1, y1, z1, color='blue')
ax1.legend()

plt.show()

