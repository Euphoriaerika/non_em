import numpy as np
import matplotlib.pyplot as plt

from .base import calculateAutocorrelationMatrix
from .base import calculateСrossСorrelationFunction

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
    """
    N = len(signal)
    t = np.arange(1, N + 1)  # Time indices
    print(t)

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
    """
    # Calculate autocorrelation matrices
    RNMuVar = calculateAutocorrelationMatrix(input_signal, M)
    RNMyVar = calculateСrossСorrelationFunction(input_signal, output_signal, M)

    # Estimate impulse response vector
    impulse_response = np.matmul(np.linalg.inv(RNMuVar), RNMyVar.T)

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


# My realization fourierTransform
def U_N(u):
    N = len(u)
    result = np.zeros(N, dtype=np.complex128)
    for t in range(N):
        result[t] = np.sum(u * np.exp(-1j * t * omega(np.arange(N), N))) / np.sqrt(N)
    return result


def omega(t, N):
    return 2 * np.pi * t / N


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
    """
    # Calculate the sum of squares of signal values in time domain
    time_domain_sum = round(np.sum(input_signal**2), 3)

    # Calculate the sum of squares of Fourier transform coefficients in frequency domain
    frequency_domain_sum = round(np.sum(np.abs(fourier_signal) ** 2), 3)

    # Check if the sums are equal
    return time_domain_sum == frequency_domain_sum


def searchZeroValue(fourier_signal):
    """
    Find the indices of Fourier transform coefficients that are equal to zero.

    Parameters
    ----------
    fourier_signal : array_like
        The Fourier transform of a signal.

    Returns
    -------
    list
        A list of indices of Fourier transform coefficients that are equal to zero.
    """
    res = []
    # Iterate over the elements of the Fourier transform
    for i in range(len(fourier_signal)):
        # Check if the magnitude of the current element is zero
        if abs(fourier_signal[i]) ** 2 == 0:
            # Append the index to the result list
            res.append(i)
    return res


def empiricalEvaluationTransfer(
    output_signal, input_signal, omega=None, magnitude_phase_show=False
):
    """
    Perform an empirical evaluation of the transfer function of a system.

    Parameters
    ----------
    output_signal : array_like
        The output signal of the system.
    input_signal : array_like
        The input signal to the system.
    omega : array_like, optional
        The frequency values (radians per sample) corresponding to the transfer function.
        If None, the function will generate frequencies from 1 to the sample rate of the output signal.
    magnitude_phase_show : bool, optional
        Whether to plot the magnitude and phase of the transfer function (default is False).

    Returns
    -------
    tuple or ndarray
        A tuple containing the empirical transfer function, its magnitude and phase if
        magnitude_phase_show is True, otherwise returns only the empirical transfer function.
    """
    if omega is None:
        # Generate frequencies from 1 to the sample rate of the output signal
        omega = [
            (2 * np.pi * i) / output_signal.size
            for i in range(1, output_signal.size + 1)
        ]
    print(omega)
    # Check if input signals have the same size
    if len(output_signal) != len(input_signal):
        raise ValueError("Output signal and input signal must have the same size.")

    # Perform an empirical evaluation of the transfer function
    empirical_transfer_function = np.divide(
        output_signal,
        input_signal,
        out=np.zeros_like(output_signal),
        where=input_signal != 0,
    )

    if magnitude_phase_show:
        # Plot the magnitude and phase of the transfer function
        magnitude, phase = plot_transfer_function(empirical_transfer_function, omega)
        return empirical_transfer_function, magnitude, phase

    return empirical_transfer_function


def computePsdBartlett(u, y, omega, gamma, show_plot=False):
    """
    Compute the Bartlett estimate of the power spectral density of a signal and a reference signal.

    Parameters
    ----------
    u : array_like
        The signal to which the power spectral density is computed.
    y : array_like
        The reference signal.
    omega : array_like
        The frequency values (radians per sample) corresponding to the power spectral density.
    gamma : int
        The length of the Bartlett window. It is used to compute the autocorrelation and cross-correlation matrices.
    show_plot : bool, optional
        Whether to plot the estimated power spectral density. If True, the function returns the estimated transfer function, its magnitude and phase. Otherwise, it returns only the estimated transfer function.

    Returns
    -------
    ndarray or tuple
        The estimated transfer function of the system. If show_plot is True, it returns a tuple containing the estimated transfer function, its magnitude and phase. Otherwise, it returns only the estimated transfer function.
    """
    tau = np.arange(-gamma, gamma + 1)  # Ngamma = gamme, so tau = -gamma:gamma
    h = np.maximum(0, 1 - np.abs(tau) / gamma)  # if tau out of range, h = 0

    # Compute the autocorrelation and cross-correlation matrices
    R_u = calculateAutocorrelationMatrix(u, gamma + 1)
    R_u_norm = np.concatenate((R_u[0], R_u[0][:-1][::-1]))

    R_yu = calculateСrossСorrelationFunction(u, y, gamma + 1)
    R_yu_norm = np.concatenate((R_yu, R_yu[:-1][::-1]))

    # Compute the Fourier transform of the Bartlett windowed autocorrelation and cross-correlation matrices
    F_u = np.zeros(len(omega), dtype=np.complex128)
    for i in range(len(omega)):
        F_u[i] = np.sum(h * R_u_norm * np.exp(-1j * omega[i] * tau), axis=0)

    F_yu = np.zeros(len(omega), dtype=np.complex128)
    for i in range(len(omega)):
        F_yu[i] = np.sum(h * R_yu_norm * np.exp(-1j * omega[i] * tau), axis=0)

    # Compute the estimated transfer function
    Ghat = np.divide(F_yu, F_u)

    # If show_plot is True, plot the estimated transfer function and return it, its magnitude and phase. Otherwise, return only the estimated transfer function.
    if show_plot:
        magnitude, phase = plot_transfer_function(Ghat, omega)
        return Ghat, magnitude, phase

    return Ghat


def plot_transfer_function(transfer_vector, omega):
    """
    Plot the magnitude and phase of the estimated transfer function of a system.

    Parameters
    ----------
    transfer_vector : ndarray
        The estimated transfer function of the system.
    omega : ndarray
        The frequency values (radians per sample) corresponding to the estimated transfer function.

    Returns
    -------
    tuple
        A tuple containing the magnitude and phase of the estimated transfer function.
    """
    magnitude = np.abs(transfer_vector)
    phase = np.angle(transfer_vector)

    # Create a figure with two subplots
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the magnitude of the estimated transfer function
    ax.subplot(2, 1, 1)
    ax.plot(omega, magnitude)
    ax.set_title("Magnitude of Estimated Transfer Function")
    ax.set_xlabel("Frequency (rad/sample)")
    ax.set_ylabel("|Ĝ(e^jω)|")

    # Plot the phase of the estimated transfer function
    ax.subplot(2, 1, 2)
    ax.plot(omega, phase)
    ax.set_title("Phase of Estimated Transfer Function")
    ax.set_xlabel("Frequency (rad/sample)")
    ax.set_ylabel("Phase (radians)")

    # Adjust the layout of the subplots
    fig.tight_layout()

    # Show the plot
    plt.show()

    # Return the magnitude and phase of the estimated transfer function
    return magnitude, phase
