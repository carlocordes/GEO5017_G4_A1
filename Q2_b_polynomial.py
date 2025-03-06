import os
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu

#Gather cwd and parameters
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

jparams = json.load(open(dir_path +  '/params.json'))
P = jparams["time_series"]

#Time frame
T = [1,2,3,4,5,6]

# Gather entries
time = np.linspace(1, 6, 100)
reconstructed = []
for dim in P:
    a = np.zeros((3, 4))

    a[0][0] = len(T)
    a[0][1] = sum(T)
    a[0][2] = sum(t**2 for t in T)
    a[0][3] =sum(dim)

    a[1][0] = a[0][1]
    a[1][1] = a[0][2]
    a[1][2] = sum(t**3 for t in T)
    a[1][3] = sum(s * t for s, t in zip(dim, T))

    a[2][0] = a[0][2]
    a[2][1] = a[1][2]
    a[2][2] = sum(t**4 for t in T)
    a[2][3] = sum(t**2 * s for s, t in zip(dim, T))

    pl, u = lu(a, permute_l=True)

    # Backwards substitution
    a3 = u[2][3] / u[2][2]
    a2 = (u[1][3] - u[1][2]*a3) / u[1][1]
    a1 = (u[0][3] - u[0][2]*a3 - u[0][1]*a2) / u[0][0]
    print(a1, a2, a3)
    curve = []
    for t in time:
        curve.append(a1 + a2*t + a3*t**2)

    reconstructed.append(curve)

# Plot results
fig = plt.figure(figsize=(18, 4))
fig.suptitle("Closed-form Polynomial", fontsize=20)

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
