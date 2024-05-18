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


def main():
    # testNonEM()
    testARModelEstimation()


if __name__ == "__main__":
    sys.exit(main())
