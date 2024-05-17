import sys
import numpy as np
import matplotlib.pyplot as plt

from data import *


# Function to calculate the cosine and sine components of a given signal
def cosineSineComponents(signal, w0):
    N = len(signal)
    t = np.arange(N)  # Time indices
    Ic = np.sum(signal * np.cos(w0 * t)) / N
    Is = np.sum(signal * np.sin(w0 * t)) / N
    return Ic, Is


# Function to calculate the correlation matrix for a given signal
def RNMu(signal, M):
    N = len(signal)
    R = np.zeros((M, M))

    for i in range(M):
        for j in range(i, M):
            R[i, j] = 1 / N * np.sum(signal[j - i : N] * signal[: N - (j - i)])
            if i != j: # R(tou) == R(-tuo)
                R[j, i] = R[i, j]

    return R


# Function to calculate the correlation vector for a given signal
def RNMy(signal, M):
    N = len(signal)
    R = np.zeros(M)
    for i in range(M):
        R[i] = 1 / N * np.sum(signal[i:N] * signal[0 : N - i])
    return R


# function to calculate the impulse response
def assessmentImpulseResponse(yk, uk, M):
    RNMuVar = RNMu(uk, M)
    RNMyVar = RNMy(yk, M)

    return np.matmul(np.linalg.inv(RNMuVar), RNMyVar)


# function to calculate the Fourier transform
def fourierTransform(input_signal, show_periodogram=False):
    # Calculation of the Fourier transform
    u1_fft =np.fft.fft(
        input_signal
    )  # U = sum(1...N){x(n)*exp(-j*2*pi*n*k/N)}, use w=2pi*k/n

    if show_periodogram:
        # Calculating the periodogram
        periodogram = np.abs(u1_fft) ** 2

        # Frequencies corresponding to the values of the periodogram
        freqs = (
            np.fft.fftfreq(input_signal.size) * 2 * np.pi
        )  # Scale frequencies to [-pi, pi]

        # Adjust the periodogram and frequencies for visualization
        periodogram = np.fft.fftshift(periodogram)
        freqs = np.fft.fftshift(freqs)

        # Plotting the periodogram
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, periodogram)
        plt.title("Periodogram of the Signal")
        plt.xlabel("Frequency (radians per sample)")
        plt.ylabel("Power")
        plt.grid(True)
        plt.show()

    return u1_fft


def main():
    # uk = fourierTransform(u1, show_periodogram=True)
    # yk = fourierTransform(y1, show_periodogram=True)
    # print(f"Fourier transform of a signal uk:\n{uk}")
    # print(f"Fourier transform of a signal yk:\n{yk}")

    # assessment = assessmentImpulseResponse(y1, u1, M)
    # print(f"Vector of the impulse response of the system:\n{assessment}")
    
    Ic, Is = cosineSineComponents(y0, w0)
    print(f"Cosine of the Fourier transform of y0: Ic = {Ic}")
    print(f"Sine of the Fourier transform of y0: Is = {Is}")





if __name__ == "__main__":
    sys.exit(main())
