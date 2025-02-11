import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from scipy.datasets import electrocardiogram
from scipy.optimize._tstutils import f1, f3, f2

ecg_signal = electrocardiogram()

# Generate a synthetic ECG-like signal
fs_high = 1000  # High sampling rate (assumed continuous)
t_high = np.linspace(0, 2, 2 * fs_high, endpoint=False)  # 2 seconds duration

# Function to sample and visualize aliasing effects
def sampling_demo(fs_sample):
    plt.figure(figsize=(12, 6))

    # Sample the ECG at the given sampling rate
    t_sampled = np.arange(0, 2, 1 / fs_sample)  # Sampled time points
    ecg_sampled = np.sin(2 * np.pi * f1 * t_sampled) + 0.3 * np.sin(2 * np.pi * f2 * t_sampled) + 0.1 * np.sin(
        2 * np.pi * f3 * t_sampled)

    # Plot original ECG
    plt.plot(t_high, ecg_signal, 'k-', alpha=0.5, label="Original ECG (High-Resolution)")

    # Plot sampled ECG
    plt.scatter(t_sampled, ecg_sampled, color='red', label=f"Sampled at {fs_sample} Hz", zorder=3)

    # Highlight Nyquist frequency
    nyquist_freq = 2 * f3
    plt.axvline(1 / nyquist_freq, color='blue', linestyle='--', label=f"Nyquist Rate = {nyquist_freq} Hz")

    # Labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Aliasing Effects on an ECG Signal")
    plt.legend()
    plt.grid(True)
    plt.show()

# Interactive widget
interact(sampling_demo, fs_sample=IntSlider(min=5, max=100, step=5, value=50));