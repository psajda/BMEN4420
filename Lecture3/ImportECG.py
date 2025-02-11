import numpy as np
import matplotlib.pyplot as plt
import pooch
from scipy.datasets import electrocardiogram


ecg = electrocardiogram()

fs = 360  # Sampling frequency in Hz


time = np.arange(ecg.size) / fs
plt.plot(time, ecg)
plt.xlabel("Time (s)")
plt.ylabel("ECG (mV)")
plt.title("ECG Signal")
plt.show()