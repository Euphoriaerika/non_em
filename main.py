import sys
import sysidenttools.nonem as nonem
import sysidenttools.armodel_estimation as ar

from sysidenttools.test_data import *


def testARModelEstimation():
    # calculation of regression matrix elements (4)
    r1 = nonem.calculateAutocorrelationMatrix(y, N)

    # initial data a1 V1 (5)
    a1 = -r1[1] / r1[0]
    V1 = r1[0] - r1[1] ** 2 / r1[0]
    print(a1, V1)


def testNonEM():
    # uk = nonem.fourierTransform(u1, show_periodogram=False)
    # yk = nonem.fourierTransform(y1, show_periodogram=False)
    # print(f"Fourier transform of a signal uk:\n{uk}")
    # print(f"Fourier transform of a signal yk:\n{yk}")

    # assessment = nonem.assessmentImpulseResponse(y1, u1, M)
    # print(f"Vector of the impulse response of the system:\n{assessment}")

    # magnitude_yw0, phase_shift_yw0 = nonem.frequencyResponse(y0, w0, show_cossin=True)
    # print(f"Vector of the impulse response of the system:\nmagnitude = {magnitude_yw0}\nphase shift = {phase_shift_yw0}")

    upk = nonem.fourierTransform(up, show_periodogram=False)
    ypk = nonem.fourierTransform(yp, show_periodogram=False)
    # print(f"Fourier transform of a signal up:\n{upk}")
    # print(f"Fourier transform of a signal yk:\n{ypk}")

    # print("Parseval equality is satisfied" if nonem.checkParsevalEquality(up, upk) else "Parseval equality is not satisfied")

    # TODO: NEED FIX SECOND PART

    # Estimation of the frequency response for empirical evaluation of the transfer function for up and yp
    magnitude_yw1, phase_shift_yw1 = nonem.frequencyResponse(
        nonem.empiricalEvaluationTransfer(yp, up), w0
    )
    print(
        f"Vector of the impulse response of the system:\nmagnitude = {magnitude_yw1}\nphase shift = {phase_shift_yw1}"
    )

    # Estimation of the frequency response for empirical evaluation of the transfer function for u1 and y1
    magnitude_ywp, phase_shift_ywp = nonem.frequencyResponse(
        nonem.empiricalEvaluationTransfer(ypk, upk), w0
    )
    print(
        f"Vector of the impulse response of the system:\nmagnitude = {magnitude_ywp}\nphase shift = {phase_shift_ywp}"
    )


def recursive_least_squares(y, n):
    N = len(y)
    # Матриця регресії
    X = np.zeros((N-n, n))
    for i in range(N-n):
        X[i, :] = y[i:i+n]
    
    # Вектор вихідних значень
    Y = y[n:]
    
    # Початкові значення
    theta = np.zeros(n)
    P = np.eye(n) * 1000  # Початкове значення коваріаційної матриці
    
    for t in range(N-n):
        x_t = X[t, :].reshape(-1, 1)
        y_t = Y[t]
        
        # Кроки 6-7: Рекурентна процедура для обчислення параметрів
        K_t = P @ x_t / (1 + x_t.T @ P @ x_t)
        theta = theta + (K_t * (y_t - x_t.T @ theta)).flatten()
        P = P - K_t @ x_t.T @ P
        
    # Крок 8: Оцінюємо вихідні дані
    y_hat = X @ theta
    
    # Крок 9: Оцінюємо шум системи
    e = Y - y_hat
    
    # Крок 10: Обчислюємо математичне сподівання оцінки шуму
    e_mean = np.mean(e)
    
    # Крок 11: Обчислюємо дисперсію оцінки шуму
    e_var = np.var(e)
    
    return theta, y_hat, e, e_mean, e_var


def main():
    # testNonEM()
    # testARModelEstimation()

    # test recursive_procedure()
    # Приклад використання
    best_n = 0
    min_e_var = float('inf')

    for n in range(3,9):
        _, _, _, _, e_var = recursive_least_squares(y, n)
        if e_var < min_e_var:
            min_e_var = e_var
            best_n = n

    print(f'Найкраще значення n: {best_n} з мінімальною дисперсією шуму: {min_e_var}')


if __name__ == "__main__":
    sys.exit(main())
