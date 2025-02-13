import matplotlib.pyplot as plt
import numpy as np

#Tracked positions
P = np.array([[2, 0, 1],
     [1.08, 1.68, 2.38],
     [-0.83, 1.82, 2.49],
     [-1.97, 0.28, 2.15],
     [-1.31, -1.51, 2.59],
     [0.57, -1.91, 4.32]])

#Time frame
T = [1,2,3,4,5,6]

print(*P.T)

#Plot trajectory
ax = plt.figure().add_subplot(projection='3d')
ax.plot(*P.T)

plt.show()