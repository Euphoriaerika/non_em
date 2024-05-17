import sys
import numpy as np
import matplotlib.pyplot as plt

from test_data import *

############################# EVALUATION OF THE FREQUENCY RESPONSE #############################

# Function to calculate the cosine and sine components of a given signal
def cosineSineComponents(signal, w0):
    N = len(signal)
    t = np.arange(N)  # Time indices
    Ic = np.sum(signal * np.cos(w0 * t)) / N
    Is = np.sum(signal * np.sin(w0 * t)) / N
    return Ic, Is

# Function to calculate the magnitude and phase shift of the frequency response
def frequencyResponse(signal, w0, show_cossin=False):
    Ic, Is = cosineSineComponents(signal, w0)

    if show_cossin:
        print(f"Cosine of the Fourier transform of y0: Ic = {Ic}")
        print(f"Sine of the Fourier transform of y0: Is = {Is}")

    magnitude = 2 * np.sqrt(Ic**2 + Is**2)
    phase_shift = -np.arctan2(Is, Ic)

    return magnitude, phase_shift

#################### ESTIMATES OF THE IMPULSE RESPONSE VECTOR OF THE SYSTEM ####################

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

################################ FOURIER TRANSFORM OF A SIGNAL #################################

# function to calculate the impulse response
def assessmentImpulseResponse(output_signal, input_signal, M):
    RNMuVar = RNMu(input_signal, M)
    RNMyVar = RNMy(output_signal, M)

    return np.matmul(np.linalg.inv(RNMuVar), RNMyVar)


# function to calculate the Fourier transform
def fourierTransform(signal, show_periodogram=False):
    # Calculation of the Fourier transform
    u1_fft =np.fft.fft(
        signal, norm="ortho"
    )  # U = sum(1...N){x(n)*exp(-j*2*pi*n*k/N)}, use w=2pi*k/n

    if show_periodogram:
        # Calculating the periodogram
        periodogram = np.abs(u1_fft) ** 2

        # Frequencies corresponding to the values of the periodogram
        freqs = (
            np.fft.fftfreq(signal.size) * 2 * np.pi
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

# function to check if the parseval equality is satisfied
def checkParsevalEquality(input_signal, fourier_signal):
    return round(np.sum(input_signal**2), 3) == round(np.sum(np.abs(fourier_signal)**2), 3)

# function to calculate the empirical evaluation of the transfer
def empiricalEvaluationTransfer(output_signal, input_signal):
    return np.divide(output_signal, input_signal)


def main():
    uk = fourierTransform(u1, show_periodogram=False)
    yk = fourierTransform(y1, show_periodogram=False)
    # print(f"Fourier transform of a signal uk:\n{uk}")
    # print(f"Fourier transform of a signal yk:\n{yk}")

    # assessment = assessmentImpulseResponse(y1, u1, M)
    # print(f"Vector of the impulse response of the system:\n{assessment}")
    
    # magnitude_yw0, phase_shift_yw0 = frequencyResponse(y0, w0, show_cossin=True)
    # print(f"Vector of the impulse response of the system:\nmagnitude = {magnitude_yw0}\nphase shift = {phase_shift_yw0}")

    upk = fourierTransform(up, show_periodogram=False)
    ypk = fourierTransform(yp, show_periodogram=False)
    # print(f"Fourier transform of a signal up:\n{upk}")
    # print(f"Fourier transform of a signal yk:\n{ypk}")

    # print("Parseval equality is satisfied" if checkParsevalEquality(up, upk) else "Parseval equality is not satisfied")
    
    ## TODO: NEED FIX SECOND PART

    # Estimation of the frequency response for empirical evaluation of the transfer function for up and yp
    magnitude_yw1, phase_shift_yw1 = frequencyResponse(empiricalEvaluationTransfer(yp, up), w0)
    print(f"Vector of the impulse response of the system:\nmagnitude = {magnitude_yw1}\nphase shift = {phase_shift_yw1}")

    # Estimation of the frequency response for empirical evaluation of the transfer function for u1 and y1
    magnitude_ywp, phase_shift_ywp = frequencyResponse(empiricalEvaluationTransfer(ypk, upk), w0)
    print(f"Vector of the impulse response of the system:\nmagnitude = {magnitude_ywp}\nphase shift = {phase_shift_ywp}")



if __name__ == "__main__":
    sys.exit(main())
