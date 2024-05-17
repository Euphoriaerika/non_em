import sys
import numpy as np
import matplotlib.pyplot as plt

from test_data import *


# Function to calculate the correlation matrix for a given signal
def RNMu(signal, M):
    N = len(signal)
    R = np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            if j >= i:
                R[i, j] = 1 / N * np.sum(signal[j - i : N] * signal[0 : N - j + i])
                

    for i in range(M):
        for j in range(M):
            if i > j:
                R[i, j] = R[j, i]

    return R

def RNMy(signal, M):
    N = len(signal)
    R = np.zeros(M)
    for i in range(M):
        R[i] = 1 / N * np.sum(signal[i : N] * signal[0 : N - i])
    return R


# function to calculate the impulse response
def assessmentImpulseResponse(yk, uk, M):
    RNMu = RNMu(uk, M)
    RNMy = RNMu(yk, M)

    return np.matmul(np.linalg.inv(RNMu), RNMy)


# function to calculate the Fourier transform
def fourierTransform(input_signal, show_periodogram=False):
    # Calculation of the Fourier transform
    u1_fft = np.fft.fft(
        input_signal
    )  # U = sum(1...N){x(n)*exp(-j*2*pi*n*k/N)}, use w=2pi*k/n

    if show_periodogram:
        # Calculating the periodogram
        periodogram = np.abs(u1_fft) ** 2

        # Frequencies corresponding to the values of the periodogram
        freqs = np.fft.fftfreq(input_signal.size) # N=100

        # Plotting the periodogram
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, periodogram)
        plt.title("Periodogram of the Signal")
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.grid(True)
        plt.show()

    return u1_fft


def main():
    #uk = fourierTransform(u1, show_periodogram=True)
    #yk = fourierTransform(y1, show_periodogram=True)

    #print(assessmentImpulseResponse(yk, uk, M))
    print(RNMu(u1, M))
    # print(np.sum(u1[0 - 2 : N] * u1[0 : N - 0 + 2])

if __name__ == "__main__":
    sys.exit(main())
