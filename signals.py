"""Digital signal processing functions for filter design and application."""

from complex import Complex
import math_utilities as mu


def calculate_filter_coefficients(fc, fs, mode):
    """Design a second-order Butterworth filter.

    Args:
        fc (float): The cutoff frequency.
        fs (float): The sampling frequency.
        mode (str): The filter mode ('high' or 'low').

    Returns:
        tuple: The filter coefficients (b0, b1, b2, a1, a2).

    Raises:
        ValueError: If the mode is not 'high' or 'low'.
    """
    if mode == "high":
        # The bilinear transform squeezes the analog frequency axis, which runs to infinity
        # into the digital one, which stops at Nyquist (half the sampling rate, and the 
        # highest frequency such a sampling rate can record). That  compression bends 
        # every frequency except DC (0), meaning that a filter designed  directly at fc 
        # would land the real cutoff below it. Prewarping picks the analog frequency that 
        # the transform will bend onto fc, cancelling the distortion. Inverting the bilinear 
        # transform, plug in the known digital frequency to find the analog one to design at.
        wc = 2 * fs * mu.tan(mu.pi * fc / fs)
        # Analog poles are placed evenly around a circle of radius wc, in the left half-plane 
        # to keep the filter stable (prevent runaway growth from a positive exponent). For 
        # a highpass, the substitution s -> wc^2 / s reflects each lowpass pole through 
        # the circle, which is division rather than multiplication below. Since the poles 
        # sit *on* the circle of radius wc, reflection maps the pair onto itself: both 
        # modes end up with the same a1 and a2 at the same cutoff, and only the numerator 
        # distinguishes them.
        hp0 = wc / Complex(0, mu.pi * 3 / 4).exp()
        hp1 = wc / Complex(0, mu.pi * 5 / 4).exp()
        # The bilinear transform converts continuous, analog frequencies to discrete, 
        # digital ones and maps the poles from the stable half of the s-plane to the inside 
        # of the unit circle, ensuring poles remain stable.
        dz0 = (1 + hp0 / (2 * fs)) / (1 - hp0 / (2 * fs))
        dz1 = (1 + hp1 / (2 * fs)) / (1 - hp1 / (2 * fs))
        # Multiply dz0 and dz1, the denominator's roots, back out into a polynomial. A 
        # quadratic with those roots is (z - dz0)(z - dz1), which expands to 
        # z^2 - (dz0 + dz1)*z + dz0*dz1. Written in powers of z^-1 that is 1 + a1*z^-1 + a2*z^-2, 
        # giving the two lines below. dz0 and dz1 are complex conjugates, with a real 
        # sum and product by construction. The remaining imaginary part is thus a rounding 
        # error, and discarded.
        a1 = -(dz0 + dz1).real
        a2 = (dz0 * dz1).real
        # Butterworth zeros sit at the far end of the spectrum from the passband - a 
        # highpass puts both at DC (z = 1), giving (1 - z^-1)^2, so the middle term flips 
        # sign: 1, -2, 1. The numerator is then scaled by gain so the passband is exactly 
        # one, and exits the filter untouched. For a highpass, gain is equal to 4G / (1 - a1 + a2). 
        # The following is the G that makes that one.
        G = (1 - a1 + a2) / 4
        b0 = G
        b1 = -2 * G
        b2 = G
        return b0, b1, b2, a1, a2
    elif mode == "low":
        # Prewarp, as before
        wc = 2 * fs * mu.tan(mu.pi * fc / fs)
        # Place analog, lowpass poles at 3pi/4 and 5pi/4, scaled to radius wc
        lp0 = wc * Complex(0, mu.pi * 3 / 4).exp()
        lp1 = wc * Complex(0, mu.pi * 5 / 4).exp()
        # Apply bilinear transform
        dz0 = (1 + lp0 / (2 * fs)) / (1 - lp0 / (2 * fs))
        dz1 = (1 + lp1 / (2 * fs)) / (1 - lp1 / (2 * fs))
        # Find denominator
        a1 = -(dz0 + dz1).real
        a2 = (dz0 * dz1).real
        # Find gain and numerator
        G = (1 + a1 + a2) / 4
        b0 = G
        b1 = 2 * G
        b2 = G
        return b0, b1, b2, a1, a2
    else:
        raise ValueError("mode must be 'high' or 'low'")


def apply_filter(signal, b0, b1, b2, a1, a2):
    """Apply a digital filter to a signal.

    Args:
        signal (list): The input signal.
        b0, b1, b2 (float): The coefficients of the numerator.
        a1, a2 (float): The coefficients of the denominator.

    Returns:
        list: The filtered signal.
    """
    output = []
    # The filter's response can be expressed as H(z): the distance from the zeros,
    # Y(z), over the distance from the poles, X(z), where z is a point on the unit
    # circle. Placed in the complex plane, the point represents a frequency. As the
    # roots of the numerator are approached, the output decays. As the roots of the
    # denominator are approached, it grows. The output thus emphasizes and attenuates
    # the desired frequencies. Y(z) is equal to (b0)(z) + (b1)(z^-1) + (b2)(z^-2), and X(z)
    # is equal to (a0) + (a1)(z^-1) + (a2)(z^-2). The a and b coefficients are those
    # of a polynomial whose roots are the poles/zeros - except for b0 and a0. b0 is
    # the gain, the weight on the current input, and a0 is the weight on the current
    # output. As such, it's moved to the right to isolate the output, and is already
    # accounted for by the other coefficients.
    for n in range(len(signal)):
        # The first two samples have an incomplete history. Missing inputs and outputs
        # are treated as zero, causing the filter to ring briefly at the start. Flitflit
        # solves this by applying the filter in both directions.
        if n == 0:
            output.append(b0 * signal[n])
        elif n == 1:
            output.append(b0 * signal[n] + b1 * signal[n - 1] - a1 * output[n - 1])
        else:
            output.append(
                b0 * signal[n]
                + b1 * signal[n - 1]
                + b2 * signal[n - 2]
                - a1 * output[n - 1]
                - a2 * output[n - 2]
            )
    return output


def filtfilt(signal, b0, b1, b2, a1, a2):
    """Apply a zero-phase digital filter to a signal.

    Args:
        signal (list): The input signal.
        b0, b1, b2 (float): The coefficients of the numerator.
        a1, a2 (float): The coefficients of the denominator.

    Returns:
        list: The filtered signal.
    """
    forward = apply_filter(signal, b0, b1, b2, a1, a2)
    backward = apply_filter(forward[::-1], b0, b1, b2, a1, a2)
    return backward[::-1]
