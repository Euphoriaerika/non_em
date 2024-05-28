# sysidenttools

`sysidenttools` is a Python library for identification and analysis of dynamic systems.

## Module: base.py

`base.py` is a core module of the `sysidenttools` library, providing fundamental functions for signal processing and statistical analysis.

### Functions:

1. `calculateAutocorrelationMatrix(array, M)`: Calculates the autocorrelation matrix of a given array.
2. `calculateСrossСorrelationFunction(out_array, in_array, M)`: Calculates the cross-correlation function between two arrays for lags from 0 to M-1.
3. `mathExpectation(arr)`: Calculates the mean or expectation of an array.
4. `mathDispersion(arr)`: Calculates the dispersion or variance of an array.

## Module: nonem.py

`nonem.py` is a module of the `sysidenttools` library, providing tools for nonlinear system identification.

### Functions:

1. `cosineSineComponents(signal, w0)`: Computes the cosine and sine components of a given signal at a specific frequency.
2. `frequencyResponse(signal, w0, show_cossin=False)`: Calculates the magnitude and phase shift of a given signal at a specific frequency.
3. `assessmentImpulseResponse(output_signal, input_signal, M)`: Estimates the impulse response vector of a system based on input and output signals.
4. `fourierTransform(signal, show_periodogram=False)`: Computes the Fourier transform of an input signal and optionally plots its periodogram.
5. `checkParsevalEquality(input_signal, fourier_signal)`: Checks Parseval's equality for a signal and its Fourier transform.
6. `searchZeroValue(fourier_signal)`: Finds the indices of Fourier transform coefficients that are equal to zero.
7. `empiricalEvaluationTransfer(output_signal, input_signal, omega=None, plot_show=False)`: Performs empirical evaluation of the transfer function of a system based on input and output signals.
8. `computePsdBartlett(u, y, omega, gamma, show_plot=False)`: Computes the Bartlett estimate of the power spectral density of a signal and a reference signal.
9. `plotTransferFunction(transfer_vector, omega)`: Plots the magnitude and phase of the estimated transfer function of a system.

## Module: armodel_estimation.py

`armodel_estimation.py` is a module of the `sysidenttools` library, providing tools for AR model parameter estimation and evaluation.

### Functions:

1. `estimateParametersVector(y, N, n)`: Estimates the parameter vector using a recursive procedure.
2. `evaluateOutputData(input_data, params, N, n)`: Evaluates the output of the AR model using the estimated parameters.
3. `noiseEstimation(input_data, output_data, N)`: Calculates the noise vector of a given input and output data vectors.

## Contribution and Support:

For any questions or suggestions regarding this project, feel free to contact me via [LinkedIn](https://www.linkedin.com/in/b-r-rudenko/).

## License:

This library is distributed under the MIT License. See the `LICENSE` file for more information.
