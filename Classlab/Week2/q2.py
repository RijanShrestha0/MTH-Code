import numpy as np
import matplotlib.pyplot as plt

v = np.array([1, 3])

scalars = np.random.uniform(-4, 4, 100)
points = np.array([s * v for s in scalars])

plt.scatter(points[:,0], points[:,1])
plt.title("Points on a Line (Subspace)")
plt.grid()
plt.show()