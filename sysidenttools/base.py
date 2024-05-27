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

    Notes
    -----
    The function calculates the autocorrelation matrix of the input array,
    which represents the correlation between different time-shifted versions
    of the array. The size of the autocorrelation matrix determines the maximum
    time shift considered. Each element R[i, j] of the matrix represents the
    correlation between the array shifted by i samples and the array shifted
    by j samples.

    The calculation assumes the array length N is greater than or equal to M.
    """

    # Get the length of the array
    N = len(array)

    # Initialize the autocorrelation matrix
    R = np.zeros((M, M))

    # Calculate the autocorrelation matrix
    for i in range(M):
        for j in range(i, M):
            # Calculate the autocorrelation element R[i, j]
            R[i, j] = (1 / N) * np.sum(array[j - i : N] * array[: N - (j - i)])

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

    Notes
    ------
    The function calculates the autocorrelation function for lags ranging from 0 to M-1.
    The autocorrelation at lag i is computed as the average product of the array with
    itself, shifted by i samples. This provides a measure of similarity between the array
    and a delayed version of itself, which is useful for identifying repeating patterns or
    periodicity within the array.

    The calculation assumes the array length N is greater than or equal to M.
    """
    N = len(out_array)
    R = np.zeros(M)
    for i in range(M):
        # Calculate the autocorrelation for lag i
        R[i] = 1 / N * np.sum(out_array[i:N] * in_array[: N - i])
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

    Notes
    -----
    The function calculates the mean of the input array. The mean is the sum
    of all the elements in the array divided by the total number of elements.
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

    Notes
    -----
    The function calculates the dispersion of the input array. The dispersion
    represents the average squared deviation of the array elements from the mean.
    """
    # Calculate the squared differences of array elements from the mean
    return 1 / len(arr) * np.sum(arr**2) - mathExpectation(arr) ** 2
