import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

# Parameters
n = 3
m = 100

# Create random vector b of length n
b = np.random.randn(n)

# Create random vector x of length m
x = np.random.randn(m)

# Construct Toeplitz matrix X from signal x with dimensions m by n
X = toeplitz(x, np.zeros(n))

# Ensure X has dimensions m by n
X = X[:m, :n]

# Compute y = X * b
y = X @ b

# Plot x and y
plt.figure(figsize=(10, 4))
plt.plot(x, label='x')
plt.plot(y, label='y')
plt.title('Plot of x and y')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()