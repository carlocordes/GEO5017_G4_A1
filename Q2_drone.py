import numpy as np
import matplotlib.pyplot as plt

# Store positions P and times T as numpy array
positions = np.array([
    [2, 0, 1],              # t=1
    [1.08, 1.68, 2.38],
    [-0.83, 1.82, 2.49],
    [-1.97, 0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [0.57, -1.91, 4.32]     # t=6
])

time = np.array([1,2,3,4,5,6])

# Print to check
# print(positions)
# print(type(positions))
# print(time)
# print(type(time))

# Extract xyz coordinates
x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]

# Print to check
# print(y)
# print(y[1])
# print(type(y))

# Create a figure with 4 subplots
fig = plt.figure(figsize=(12,10))

# 1st plot -> 3d plot
ax1 = fig.add_subplot(2,2,1, projection='3d')
ax1.plot(x,y,z, marker='o', label='Drone Trajectory') # Plot the trajectory
ax1.set_xlabel('X-axis') # Labels and titles
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('Z-axis')
ax1.set_title(('Drone Positions (Px, Py, Pz) over Time (t)'))
for i in range (len(positions)): # Add annotations for each point
    ax1.text(x[i], y[i], z[i], f"t={i+1}")

# 2nd plot -> 2D XY
ax2 = fig.add_subplot(2,2,2)
ax2.plot(x,y, marker='o', label='Drone Trajectory') # Plot the trajectory
ax2.set_xlabel('X-axis') # Labels and titles
ax2.set_ylabel('Y-axis')
ax2.set_title(('2D XY Trajectory'))
ax2.grid(True)
for i in range (len(positions)): # Add annotations for each point
    ax2.text(x[i], y[i], f"t={i+1}")

# 3rd plot -> 2D YZ
ax3 = fig.add_subplot(2,2,3)
ax3.plot(y,z, marker='o', label='Drone Trajectory') # Plot the trajectory
ax3.set_xlabel('Y-axis') # Labels and titles
ax3.set_ylabel('Z-axis')
ax3.set_title(('2D YZ Trajectory'))
ax3.grid(True)
for i in range (len(positions)): # Add annotations for each point
    ax3.text(y[i], z[i],f"t={i+1}")

# 4th plot -> 2D XZ
ax4 = fig.add_subplot(2,2,4)
ax4.plot(x,z, marker='o', label='Drone Trajectory') # Plot the trajectory
ax4.set_xlabel('X-axis') # Labels and titles
ax4.set_ylabel('Z-axis')
ax4.set_title(('2D XZ Trajectory'))
ax4.grid(True)
for i in range (len(positions)): # Add annotations for each point
    ax4.text(x[i], z[i],f"t={i+1}")

# Show the 4 plots
plt.tight_layout()
plt.legend()
plt.show()
