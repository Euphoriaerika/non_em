import numpy as np

from .base import calculateAutocorrelationMatrix


def estimateParametersVector(y, N, n):
    Nk = N - 1
    # calculation of regression matrix elements (4)
    rec_matrix = calculateAutocorrelationMatrix(y, N)
    # print(rec_matrix)

    # initial data a1 V1 (5)
    an = np.zeros(N)
    for i in range(Nk):
        an[i] = -rec_matrix[0][i + 1] / rec_matrix[0][i]
    # print(an)

    V = rec_matrix[0][0] - rec_matrix[0][1] ** 2 / rec_matrix[0][0]
    # print(f"V = {V}")

    for i in range(1, n):
        alpha = rec_matrix[0][Nk] + np.sum(
            [an[i] * rec_matrix[0][Nk - i] for i in range(0, Nk)]
        )
        # print(f"alpha = {alpha}")
        p = -alpha / V
        # print(f"p = {p}")
        an[Nk] = p
        for k in range(0, Nk):
            an[k] = an[k] + an[Nk - k] * p

    return an
