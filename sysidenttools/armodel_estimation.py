import numpy as np

from .base import calculateAutocorrelationMatrix


def estimateParametersVector(y, N, n):
    """
    The function estimateParametersVector estimates the parameter vector using a recursive procedure.

    Arguments:
    y -- input data vector
    N -- length of the input vector
    n -- number of parameters to be estimated

    Returns:
    an -- vector of estimated parameters

    """
    Nk = N - 1
    # calculation of regression matrix elements (4)
    rec_matrix = calculateAutocorrelationMatrix(y, N)

    # initial data a1 V1 (5)
    an = np.zeros(1)
    an[0] = -rec_matrix[0][1] / rec_matrix[0][0]  # a1
    V = rec_matrix[0][0] - rec_matrix[0][1] ** 2 / rec_matrix[0][0]  # V1

    for i in range(1, n):
        an1 = np.zeros(i + 1)  # vector next generation

        alpha = rec_matrix[0][i + 1] + np.sum(
            [an[k] * rec_matrix[0][i - k] for k in range(0, i)]
        )
        # print(f"alpha = {alpha}")
        p = -alpha / V
        # print(f"p = {p}")
        V += p * alpha
        # print(f"V = {V}")

        for k in range(0, i):
            an1[k] = an[k] + an[i - k - 1] * p  # add elements new generation
        an1[i] = p  # last element of new generation equal to p
        # print(f"an1 = {an1}")
        an = an1  # copy new generation to old generation

    return an


def evaluateOutputData(input_data, params, N, n):
    """
    This function evaluates the output of the AR model using the estimated parameters.

    Arguments:
    input_data -- input data vector
    params -- vector of estimated parameters
    N -- length of the input vector
    n -- number of parameters to be estimated

    Returns:
    yk -- vector of the output of the AR model

    """
    # Initialize output vector yk
    yk = np.zeros(N)
    yk[0] = input_data[0]

    # Initialize regression vector fT
    fT = np.zeros(n)

    # Iterate over the yk values
    for i in range(1, N):
        # Calculate fT
        for j in range(n):
            if i > j:
                fT[j] = input_data[i - j - 1]  # if i > j, calculate fT from the y array
            else:
                fT[j] = 0  # if out of bounds, fT = 0

        # Evaluate the output of the AR model
        yk[i] = np.dot(fT, params)

    return yk


def noiseEstimation(input_data, output_data, N):
    """
    Calculate the noise vector of a given input and output data vectors.

    Parameters
    ----------
    input_data : array_like
        The input data vector whose noise is to be calculated.
    output_data : array_like
        The output data vector of the AR model.
    N : int
        The length of the input and output vectors.

    Returns
    -------
    ndarray
        The noise vector of the input and output data.
    """
    noise = np.zeros(N)
    for i in range(N):
        noise[i] = input_data[i] - output_data[i]
    return noise
