import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from scipy.signal import tf2zpk

def arma_filter(a, b, x):
    P = len(a)  # AR order
    Q = len(b)  # MA order
    N = len(x)  # Length of the input signal

    y = np.zeros(N)  # Initialize output array

    for n in range(N):
        # Compute the AR contribution: -sum_{k=1..P} a[k] * y[n-k]
        ar_sum = 0.0
        for k in range(1, P + 1):
            if n - k >= 0:
                ar_sum += a[k - 1] * y[n - k]

        # Compute the MA contribution: sum_{k=0..Q} b[k] * x[n-k]
        ma_sum = 0.0
        for k in range(Q):
            if n - k >= 0:
                ma_sum += b[k] * x[n - k]

        # Combine them (notice the minus sign on the AR part)
        y[n] = -ar_sum + ma_sum

    return y

def plot_arma(a, b, x):
    y = arma_filter(a, b, x)

    # Plot the original and filtered signals
    plt.figure(figsize=(12, 6))
    plt.plot(x, label='Original Signal x')
    plt.plot(y, label='Filtered Signal y')
    plt.title('Original and Filtered Signals')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Print the difference equation
    ar_terms = ' - '.join([f'{a[i]}*y[n-{i+1}]' for i in range(len(a))])
    ma_terms = ' + '.join([f'{b[i]}*x[n-{i}]' for i in range(len(b))])
    difference_equation = f'y[n] = -({ar_terms}) + ({ma_terms})'

    # Add the difference equation to the plot
    plt.text(0.05, 0.95, difference_equation, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    plt.show()

    print("Difference Equation:")
    print(difference_equation)

    # Plot poles and zeros
    #z, p, k = tf2zpk(np.hstack(([1], np.array(b))), np.hstack(([1], -np.array(a))))
    z, p, k = tf2zpk(b, np.hstack(([1], -np.array(a))))
    # Create transfer function
    print("Poles and Zeros")
    print("Zeros:", z)
    print("Poles:", p)

    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
    plt.scatter(np.real(p), np.imag(p), marker='x', label='Poles')
    plt.axvline(0, color='k', lw=1)
    plt.axhline(0, color='k', lw=1)
    plt.title('Poles and Zeros')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot the unit circle
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    plt.gca().add_artist(unit_circle)

    # Set plot limits
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    # Check stability
    if np.all(np.abs(p) < 1):
        stability = "Stable"
    else:
        stability = "Unstable"
    plt.text(0.05, 0.95, f'System is {stability}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    plt.show()


# Example usage
x = np.sin(2 * np.pi * 3 * np.linspace(0, 2, 100))  # 3Hz sine wave for 2 seconds
#a = [0.5, -0.2]  # Example AR coefficients
#b = [1.0, 0.5, 0.3]  # Example MA coefficients
a = [-0.9]  # Example AR coefficients
b = [-1, 1]  # Example MA coefficients

plot_arma(a, b, x)