# sysidenttools

`sysidenttools` is a Python library for identification and analysis of dynamic systems.

## Module: nonem.py

`nonem.py` is a module of the `sysidenttools` library, providing tools for nonlinear system identification.

### Functions:

1. `cosineSineComponents(signal, w0)`: Computes the cosine and sine components of a given signal at a specific frequency.
2. `frequencyResponse(signal, w0, show_cossin=False)`: Calculates the magnitude and phase shift of a given signal at a specific frequency.
3. `checkParsevalEquality(input_signal, fourier_signal)`: Checks Parseval's equality for a signal and its Fourier transform.
4. `empiricalEvaluationTransfer(output_signal, input_signal)`: Performs empirical evaluation of the transfer function of a system based on input and output signals.
5. `fourierTransform(signal, show_periodogram=False)`: Computes the Fourier transform of an input signal and optionally plots its periodogram.

### Usage:

```python
import sysidenttools.nonem as nonem

# Use functions from the nonem module for nonlinear system identification.
```

## Contribution and Support:

If you would like to contribute to improving this library or have any questions, please create an issue or submit a pull request.

## License:

This library is distributed under the MIT License. See the `LICENSE` file for more information.