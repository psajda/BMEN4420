import numpy as np
import matplotlib.pyplot as plt

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
plt.title('Time Series of Random Noise')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

