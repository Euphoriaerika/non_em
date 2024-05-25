import sys

import sysidenttools.nonem as nonem
import sysidenttools.armodel_estimation as ar
import sysidenttools.base as base

from data import *


def testARModelEstimation():
    n_count = 8
    for n_iter in range(n, n_count):
        # Estimate the parameters vector
        O = ar.estimateParametersVector(y, N, n_iter)
        # print(f"Estimated parameters vector: {O}")
        # Evaluate the output data
        yk = ar.evaluateOutputData(y, O, N, n_iter)
        # print(f"Evaluation of the output data: \n{yk}")

        # Estimate the noise
        noise = ar.noiseEstimation(y, yk, N)
        # print(f"Noise estimation: \n{noise}")

        # Calculate the math expectation
        math_expectation = base.mathExpectation(noise)
        # print(f"Math expectation :{math_expectation}")

        # Calculate the math dispersion
        math_dispersion = base.mathDispersion(noise)
        # print(f"Math dispersion {math_dispersion}")

        # Initialize the minimum dispersion and optimal model order
        # The minimum dispersion and optimal model order are calculated
        # by comparing the dispersion of the current model order with
        # the minimum dispersion found so far
        if n_iter == 3:
            # Set the minimum dispersion to the current dispersion
            min_dis = math_dispersion
            min_me = math_expectation
            min_n = n_iter
            yk_res = yk
            O_res = O
            noise_res = noise

        if math_dispersion < min_dis:
            # Update the minimum dispersion to the current dispersion
            # and update the optimal model order to the current model order
            min_dis = math_dispersion
            min_me = math_expectation
            min_n = n_iter
            yk_res = yk
            O_res = O
            noise_res = noise

    # Print the optimal model order and minimum dispersion
    print(f"Optimal model order: {min_n} with min dispersion: {min_dis}")
    
    # Write y_res to output_data/BRRudenko.txt
    with open('output_data/BRRudenko.txt', 'w') as file:
        file.write(f"\nN: {N}\n")
        file.write(f"\nn: {min_n}\n")
        file.write(f"\nTheta values:\n")
        for value in O_res:
            file.write(f"{value}\n")
        file.write(f"\nyk values:\n")
        for value in yk_res:
            file.write(f"{value}\n")
        file.write(f"\nnoise values:\n")
        for value in noise_res:
            file.write(f"{value}\n")
        file.write(f"\nMath expectation: {min_me}\n")
        file.write(f"\nMath dispersion: {min_dis}\n")



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
