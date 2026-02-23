import numpy as np
import matplotlib.pyplot as plt

# 1. Dataset
X0 = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
X1 = np.array([[7, 8], [8, 9], [9, 8], [8, 7], [9, 9]])

mean0 = np.mean(X0, axis=0)
mean1 = np.mean(X1, axis=0)

Sw = np.dot((X0 - mean0).T, (X0 - mean0)) + np.dot((X1 - mean1).T, (X1 - mean1))
mean_diff = (mean1 - mean0).reshape(2, 1)
Sb = np.dot(mean_diff, mean_diff.T)

matrix = np.dot(np.linalg.inv(Sw), Sb)
eigenvalues, eigenvectors = np.linalg.eig(matrix)
# eigenvalues --> how much separation : eigenvectors --> which direction
# we want that direction w that gives us the maximum separation
w = eigenvectors[:, np.argmax(eigenvalues)].real

#normalize for better plotting
w = w / np.linalg.norm(w)

#projection
total_mean = np.mean(np.vstack((X0, X1)), axis=0)

# project points to the line defined by direction w and passing through total_mean
def projection_point(p, direction, origin):
    v = p - origin
    proj_len = np.dot(v, direction)
    return origin + direction * proj_len

projected_X0 = np.array([projection_point(p, w, total_mean) for p in X0])
projected_X1 = np.array([projection_point(p, w, total_mean) for p in X1])


# Visualization
# 4. Plotting
plt.figure(figsize=(8, 6))

# Original Data
plt.scatter(X0[:, 0], X0[:, 1], color='red', label='Class 0 (Original)', alpha=0.5)
plt.scatter(X1[:, 0], X1[:, 1], color='blue', label='Class 1 (Original)', alpha=0.5)

# The LDA Axis (Line)
line_x = np.linspace(-2, 12, 100)
# Equation of line: point = origin + t * w
# We need to solve for y given x to plot easily, or just generate points
line_points = np.array([total_mean + t * w for t in np.linspace(-10, 10, 100)])
plt.plot(line_points[:, 0], line_points[:, 1], color='green', linewidth=2, label='LDA Axis (Eigenvector)')

# Projected Points
plt.scatter(projected_X0[:, 0], projected_X0[:, 1], color='red', marker='x', s=100, label='Class 0 (Projected)')
plt.scatter(projected_X1[:, 0], projected_X1[:, 1], color='blue', marker='x', s=100, label='Class 1 (Projected)')

# Connecting lines (visualize the projection)
for p, p_proj in zip(X0, projected_X0):
    plt.plot([p[0], p_proj[0]], [p[1], p_proj[1]], 'r--', alpha=0.3)
for p, p_proj in zip(X1, projected_X1):
    plt.plot([p[0], p_proj[0]], [p[1], p_proj[1]], 'b--', alpha=0.3)

plt.title('LDA: Projection onto the Best Eigenvector')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.axis('equal') # Crucial to see perpendicular projection correctly
plt.show()
