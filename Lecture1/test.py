import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
duration = 3  # seconds
sampling_rate = 1000  # Hz
num_samples = duration * sampling_rate

# Generate time vector
t = np.linspace(0, duration, num_samples, endpoint=False)

# Generate random noise
noise = np.random.normal(0, 1, num_samples)

# Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(t, noise)
plt.title('Gaussian Noise')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Plot the histogram
plt.figure(figsize=(10, 4))
count, bins, ignored = plt.hist(noise, bins=30, density=True, alpha=0.6, color='g')

# Fit a Gaussian curve to the histogram data
mu, std = norm.fit(noise)
p = norm.pdf(bins, mu, std)
plt.plot(bins, p, 'k', linewidth=2)
plt.title('Histogram of Gaussian Noise with Fitted Gaussian Curve')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()
