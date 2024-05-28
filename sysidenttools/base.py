import numpy as np


def calculateAutocorrelationMatrix(array, M):
    """
    Calculate the autocorrelation matrix of a given array.

    Parameters
    ----------
    array : array_like
        The input array whose autocorrelation matrix is to be calculated.
    M : int
        The size of the autocorrelation matrix.

    Returns
    -------
    ndarray
        The M x M autocorrelation matrix of the input array.

    """

    # Get the length of the array
    N = len(array)

    # Initialize the autocorrelation matrix
    R = np.zeros((M, M))

    # Calculate the autocorrelation matrix
    for i in range(M):
        for j in range(i, M):
            # Calculate the autocorrelation element R[i, j]
            R[i, j] = (1 / N) * np.sum(array[j - i : N] * array[: N - (j - i)]).real

            # Ensure the matrix is symmetric
            if i != j:
                R[j, i] = R[i, j]

    return R


def calculateСrossСorrelationFunction(out_array, in_array, M):
    """
    Calculate the autocorrelation function of a given array for lags from 0 to M-1.

    Parameters
    ----------
    array : array_like
        The input array whose autocorrelation function is to be calculated.
    M : int
        The number of lags for which to calculate the autocorrelation.

    Returns
    ----------
    R : ndarray
        The autocorrelation function of the input array for lags from 0 to M-1.

    """
    N = len(out_array)
    R = np.zeros(M)
    for i in range(M):
        # Calculate the autocorrelation for lag i
        R[i] = 1 / N * np.sum(out_array[i:N] * in_array[: N - i]).real
    return R


def mathExpectation(arr):
    """
    Calculate the mean or expectation of an array.

    Parameters
    ----------
    arr : array_like
        The input array whose mean is to be calculated.

    Returns
    -------
    float
        The mean of the input array.

    """
    return 1 / len(arr) * np.sum(arr)


def mathDispersion(arr):
    """
    Calculate the dispersion or variance of an array.

    Parameters
    ----------
    arr : array_like
        The input array whose dispersion is to be calculated.

    Returns
    -------
    float
        The dispersion (variance) of the input array.

    """
    # Calculate the squared differences of array elements from the mean
    return 1 / len(arr) * np.sum(arr**2) - mathExpectation(arr) ** 2
