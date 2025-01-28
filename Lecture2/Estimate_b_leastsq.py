import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

# Parameters
n = 100
m = 3
p = 100
duration = 2  # seconds
frequency = 3  # Hz

# Generate time vector
t = np.linspace(0, duration, n, endpoint=False)

# Generate 3Hz sine wave
# Generate EKG-like signal
x = np.sin(2 * np.pi * frequency * t)
plt.close('all')
# Plot the sine wave

plt.plot(t, x)
plt.title('3Hz Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)


# Create vector b of length m
b = np.random.randn(m)

# Construct Toeplitz matrix X from signal x with dimensions n by m
X = toeplitz(x, np.zeros(m))

y = X @ b

plt.plot(t, y)
plt.legend(['3Hz Sine Wave', 'Random Noise'])
plt.show()
