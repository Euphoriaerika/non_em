import numpy as np
import matplotlib.pyplot as plt

from .base import calculateAutocorrelationMatrix

# ===============================================
# EVALUATION OF THE FREQUENCY RESPONSE
# ===============================================


def cosineSineComponents(signal, w0):
    """
    Calculate the cosine and sine components of a given signal at a specific frequency.

    Parameters
    ----------
    signal : array_like
        The input signal whose components are to be calculated.
    w0 : float
        The angular frequency (in radians per sample) at which to calculate the components.

    Returns
    ----------
    Ic : float
        The cosine component of the signal at frequency w0.
    Is : float
        The sine component of the signal at frequency w0.

    Notes
    ------
    This function computes the cosine and sine components of the input signal
    at the specified frequency w0. These components indicate the amplitude and
    phase of the signal's response at that frequency.

    The cosine component (Ic) represents the in-phase component, while the sine
    component (Is) represents the quadrature component.
    """
    N = len(signal)
    t = np.arange(N)  # Time indices

    Ic = np.sum(signal * np.cos(w0 * t)) / N
    Is = np.sum(signal * np.sin(w0 * t)) / N
    return Ic, Is


def frequencyResponse(signal, w0, show_cossin=False):
    """
    Calculate the magnitude and phase shift of a given signal at a specific frequency.

    Parameters
    ----------
    signal : array_like
        The input signal whose frequency response is to be calculated.
    w0 : float
        The angular frequency (in radians per sample) at which to calculate the response.
    show_cossin : bool, optional
        If True, prints the cosine and sine components of the signal.

    Returns
    ----------
    magnitude : float
        The magnitude of the signal's frequency component at w0.
    phase_shift : float
        The phase shift of the signal's frequency component at w0.

    Notes
    ------
    This function calculates the magnitude and phase shift of a given signal
    at a specific frequency. The magnitude is calculated as the square root of
    the sum of the squares of the cosine and sine components. The phase shift
    is calculated as the arctan of the sine component divided by the cosine
    component.

    Optionally, it prints the cosine and sine components of the signal.
    """
    # Calculate the cosine and sine components
    Ic, Is = cosineSineComponents(signal, w0)

    # Optionally print the cosine and sine components
    if show_cossin:
        print(f"Cosine of the Fourier transform of y0: Ic = {Ic}")
        print(f"Sine of the Fourier transform of y0: Is = {Is}")

    # Calculate the magnitude of the frequency component
    magnitude = 2 * np.sqrt(Ic**2 + Is**2)

    # Calculate the phase shift of the frequency component
    phase_shift = -np.arctan2(Is, Ic)

    return magnitude, phase_shift


# ===============================================
# ESTIMATES OF THE IMPULSE RESPONSE VECTOR OF THE SYSTEM
# ===============================================


def calculateAutocorrelationFunction(signal, M):
    """
    Calculate the autocorrelation function of a given signal for lags from 0 to M-1.

    Parameters
    ----------
    signal : array_like
        The input signal whose autocorrelation function is to be calculated.
    M : int
        The number of lags for which to calculate the autocorrelation.

    Returns
    ----------
    R : ndarray
        The autocorrelation function of the input signal for lags from 0 to M-1.

    Notes
    ------
    The function calculates the autocorrelation function for lags ranging from 0 to M-1.
    The autocorrelation at lag i is computed as the average product of the signal with
    itself, shifted by i samples. This provides a measure of similarity between the signal
    and a delayed version of itself, which is useful for identifying repeating patterns or
    periodicity within the signal.

    The calculation assumes the signal length N is greater than or equal to M.
    """
    N = len(signal)
    R = np.zeros(M)
    for i in range(M):
        # Calculate the autocorrelation for lag i
        R[i] = 1 / N * np.sum(signal[i:N] * signal[: N - i])
    return R


def assessmentImpulseResponse(output_signal, input_signal, M):
    """
    Estimate the impulse response vector of a system based on input and output signals.

    Parameters
    ----------
    output_signal : array_like
        The output signal of the system.
    input_signal : array_like
        The input signal to the system.
    M : int
        The dimensionality of the autocorrelation matrices.

    Returns
    ----------
    impulse_response : ndarray
        The estimated impulse response vector of the system.

    Notes
    ------
    This function estimates the impulse response vector of a system using the method
    of autocorrelation. It computes the autocorrelation matrices of the input and
    output signals, and then calculates the inverse of the input signal's autocorrelation
    matrix. This inverse matrix is then multiplied by the output signal's autocorrelation
    vector to estimate the impulse response vector of the system.

    The accuracy of the estimation may depend on the size of the autocorrelation matrices
    and the quality of the input and output signals.
    """
    # Calculate autocorrelation matrices
    RNMuVar = calculateAutocorrelationMatrix(input_signal, M)
    RNMyVar = calculateAutocorrelationFunction(output_signal, M)

    # Estimate impulse response vector
    impulse_response = np.matmul(np.linalg.inv(RNMuVar), RNMyVar)

    return impulse_response


# ===============================================
# FOURIER TRANSFORM OF A SIGNAL
# ===============================================


def fourierTransform(signal, show_periodogram=False):
    """
    Calculate the Fourier transform of a given signal.

    Parameters
    ----------
    signal : array_like
        The input signal.
    show_periodogram : bool, optional
        Whether to display the periodogram of the signal (default is False).

    Returns
    -------
    u1_fft : ndarray
        The Fourier transform of the input signal.

    Notes
    ------
    The function calculates the Fourier transform of the input signal using the fast Fourier
    transform algorithm. If 'show_periodogram' is set to True, it also computes and displays
    the periodogram of the signal, which shows the frequency components present in the signal.

    """
    # Calculation of the Fourier transform
    u1_fft = np.fft.fft(
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


def checkParsevalEquality(input_signal, fourier_signal):
    """
    Check Parseval's equality for the given input signal and its Fourier transform.

    Parameters
    ----------
    input_signal : array_like
        The input signal.
    fourier_signal : array_like
        The Fourier transform of the input signal.

    Returns
    -------
    bool
        True if Parseval's equality holds, False otherwise.

    Notes
    ------
    Parseval's equality states that the sum of squares of signal values
    in time domain is equal to the sum of squares of Fourier transform
    coefficients in frequency domain, when properly normalized.

    """
    # Calculate the sum of squares of signal values in time domain
    time_domain_sum = round(np.sum(input_signal**2), 3)

    # Calculate the sum of squares of Fourier transform coefficients in frequency domain
    frequency_domain_sum = round(np.sum(np.abs(fourier_signal) ** 2), 3)

    # Check if the sums are equal
    return time_domain_sum == frequency_domain_sum


def empiricalEvaluationTransfer(output_signal, input_signal):
    """
    Perform empirical evaluation of the transfer function of a system.

    Parameters
    ----------
    output_signal : array_like
        The output signal of the system.
    input_signal : array_like
        The input signal to the system.

    Returns
    -------
    ndarray
        The empirical transfer function of the system.

    Notes
    ------
    Empirical evaluation of the transfer function involves dividing the output signal
    by the input signal. This approach is based on the assumption that the system acts
    as a linear time-invariant (LTI) system, where the output is proportional to the
    input according to the system's transfer function.

    """
    # Perform empirical evaluation of the transfer function
    empirical_transfer_function = np.divide(output_signal, input_signal)

    return empirical_transfer_function
