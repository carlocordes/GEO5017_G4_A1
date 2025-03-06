import json
import os

import matplotlib.pyplot as plt
import numpy as np

#Gather cwd and parameters
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

jparams = json.load(open(dir_path +  '/params.json'))
P = jparams["time_series"]

#Time frame
T = [1,2,3,4,5,6]
P = [[2, 1.08, -0.83, -1.97, -1.31, 0.57],
     [0, 1.68, 1.82, 0.28, -1.51 ,-1.91],
     [1, 2.38, 2.49, 2.15, 2.59, 4.32]]


time = np.linspace(1, 6, 1000)
reconstructed = []
for dim in P:
     s_mean = np.mean(dim)
     t_mean = np.mean(T)

     s_var = np.var(dim)

     st_cov = 0
     for i in range(len(P[0])):
          st_cov += (dim[i]- s_mean)*(T[i]- t_mean)
     st_cov /= (len(P[0])-1)

     beta = st_cov / s_var
     alpha = s_mean - beta*t_mean
     print(alpha, beta)
     #Reconstruct curve
     curve = []
     for t in time:
          curve.append(alpha + beta * t)


     reconstructed.append(curve)


# Plot results
fig = plt.figure(figsize=(18, 4))
fig.suptitle("Closed-form linear", fontsize=20)

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