import numpy as np
import matplotlib.pyplot as plt

# random vectors
t = np.random.randn(2)
r = np.random.randn(2)

# projection of t onto r
t_parallel = (np.dot(t, r) / np.dot(r, r)) * r

# perpendicular component
t_perp = t - t_parallel

# check
print("t =", t)
print("t_parallel + t_perp =", t_parallel + t_perp)
print("Dot product (should be 0):", np.dot(t_parallel, t_perp))

# plot
plt.quiver(0,0,t[0],t[1], angles='xy', scale_units='xy', scale=1, label='t')
plt.quiver(0,0,r[0],r[1], angles='xy', scale_units='xy', scale=1, label='r')
plt.quiver(0,0,t_parallel[0],t_parallel[1], linestyle='dashed', label='t_parallel')
plt.quiver(0,0,t_perp[0],t_perp[1], linestyle='dotted', label='t_perp')

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.legend()
plt.grid()
plt.show()