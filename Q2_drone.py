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


time = np.linspace(1, 5, 100)
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

     #Reconstruct curve
     curve = []
     for t in time:
          curve.append(alpha + beta * t)


     reconstructed.append(curve)


# Plot results
fig = plt.figure(figsize=(18, 4))
fig.suptitle("Linear Regression", fontsize=20)
axes = [fig.add_subplot(1, 4, 1, projection='3d')]
axes += [fig.add_subplot(1, 4, i+1) for i in range(1, 4)]

axes[0].plot(*P, label = 'Data')
axes[0].plot(*reconstructed, label = 'Regression')
axes[0].legend()

for i in range(3):
     axes[i+1].plot(time, reconstructed[i], label = 'Regression')
     axes[i+1].plot(T, P[i], label = 'Data')
     axes[i+1].set_xlabel('Time / s')
     axes[i+1].set_ylabel('Spatial dimension {}'.format(i+1))
     axes[i+1].legend()


plt.show()