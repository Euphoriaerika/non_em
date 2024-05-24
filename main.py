import sys
import sysidenttools.nonem as nonem
import sysidenttools.armodel_estimation as ar
import sysidenttools.base as base

from sysidenttools.test_data import *


def testARModelEstimation():
    O = ar.estimateParametersVector(y, N, n)
    # print(O)

    yk = np.zeros(N)
    yk[0] = y[0]
    yT = np.zeros(N)
    for i in range(1, N):
        for j in range(N):
            if i - j > 0:
                yT[j] = y[i - j]
            else:
                yT[j] = 0
        yk[i] = np.dot(yT, O)

    # print(yk)

    noise = np.zeros(N)
    for i in range(N):
        noise[i] = y[i] - yk[i]

    # print(base.mathExpectation(noise))
    # print(base.mathDispersion(noise))
    # print(noise)


def testNonEM():
    # uk = nonem.fourierTransform(u1, show_periodogram=True)
    # yk = nonem.fourierTransform(y1, show_periodogram=True)
    # print(f"Fourier transform of a signal uk:\n{uk}")
    # print(f"Fourier transform of a signal yk:\n{yk}")

    # assessment = nonem.assessmentImpulseResponse(y1, u1, M)
    # print(f"Vector of the impulse response of the system:\n{assessment}")

    magnitude_yw0, phase_shift_yw0 = nonem.frequencyResponse(y0, w0, show_cossin=True)
    print(
        f"Vector of the impulse response of the system:\nmagnitude = {magnitude_yw0}\nphase shift = {phase_shift_yw0}"
    )

    # upk = nonem.fourierTransform(up, show_periodogram=False)
    # ypk = nonem.fourierTransform(yp, show_periodogram=False)
    # print(f"Fourier transform of a signal up:\n{upk}")
    # print(f"Fourier transform of a signal yk:\n{ypk}")

    # print("Parseval equality is satisfied" if nonem.checkParsevalEquality(up, upk) else "Parseval equality is not satisfied")

    # TODO: NEED FIX SECOND PART

    # # Estimation of the frequency response for empirical evaluation of the transfer function for up and yp
    # magnitude_yw1, phase_shift_yw1 = nonem.frequencyResponse(
    #     nonem.empiricalEvaluationTransfer(yp, up), w0
    # )
    # print(
    #     f"Vector of the impulse response of the system:\nmagnitude = {magnitude_yw1}\nphase shift = {phase_shift_yw1}"
    # )

    # Estimation of the frequency response for empirical evaluation of the transfer function for u1 and y1
    # magnitude_ywp, phase_shift_ywp = nonem.frequencyResponse(
    #     nonem.empiricalEvaluationTransfer(ypk, upk), w0
    # )
    # print(
    #     f"Vector of the impulse response of the system:\nmagnitude = {magnitude_ywp}\nphase shift = {phase_shift_ywp}"
    # )


def main():
    # testNonEM()
    testARModelEstimation()


if __name__ == "__main__":
    sys.exit(main())
