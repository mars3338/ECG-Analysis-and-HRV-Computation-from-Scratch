"""Minimal math library plus Pan-Tompkins QRS detection primitives."""

pi = 3.141592653589793
tau = 2 * pi
e = 2.718281828459045


def sin(x):
    """Compute the y coordinate on the unit circle for a given angle.

    Args:
        x (float): The angle in radians.

    Returns:
        float: The y coordinate on the unit circle.
    """
    # Taylor series converges quickly near 0
    # Fold x into (-pi, pi] using sine's 2*pi periodicity.
    x = x - tau * (x // tau)  # Reduce into [0, tau)
    # Reduce into (-pi, pi]
    if x > pi:
        x -= tau
    term = x
    total = 0.0
    # sin(x) = x - x^3/3! + x^5/5! - ...
    # Compute by building terms incrementally
    for n in range(15):
        total += term
        term *= -x * x / ((2 * n + 2) * (2 * n + 3))
    return total


def cos(x):
    """Compute the x coordinate on the unit circle for a given angle.

    Args:
        x (float): The angle in radians.

    Returns:
        float: The x coordinate on the unit circle.
    """
    # Cosine is sine phase-shifted, and can thus be defined by it
    return sin(x + pi / 2)


def tan(x):
    """Compute the ratio of sine and cosine of a given angle.

    Args:
        x (float): The angle in radians.

    Returns:
        float: The ratio of sine and cosine.
    """
    c = cos(x)
    if abs(c) < 1e-12:
        raise ValueError("tan undefined for this input")
    return sin(x) / c


def arctan(x):
    """Compute the angle of which tangent is x within (-pi/2, pi/2).

    Args:
        x (float): The ratio of sine to cosine.

    Returns:
        float: The angle in radians.
    """
    # Relevant Taylor series converges slowly at domain edges (|x| = 1)
    # Tangent describes the ratio between sine and cosine
    # As such, arctan(a/b) + arctan(b/a) must equal pi/2
    if x > 1:
        return pi / 2 - arctan(1 / x)
    # As arctan(−x) = −arctan(x), negate both sides of the previous identity to account 
    # for sign change
    if x < -1:
        return -pi / 2 - arctan(1 / x)
    # Shift by pi/4 to reduce further
    # Angle reduction: arctan(a) − arctan(b) = arctan((a−b)/(1+ab))
    # Arctan(1) = pi/4
    if x > 0.5:
        return pi / 4 + arctan((x - 1) / (x + 1))
    # Mirror for negative values
    if x < -0.5:
        return -pi / 4 + arctan((x + 1) / (1 - x))
    # Compute Taylor series by building terms incrementally
    term = x
    total = 0.0
    for n in range(30):
        total += term / (2 * n + 1)
        term *= -x * x
    return total


def arcsin(x):
    """Compute the angle of which sine is x within (-pi/2, pi/2).

    Args:
        x (float): The ratio of opposite to hypotenuse.

    Returns:
        float: The angle in radians.
    """
    # Outside [-1, 1] the square root below goes complex, causing a type error
    # from inside arctan rather than the clear error below.
    if not -1 <= x <= 1:
        raise ValueError("arcsin domain is [-1, 1]")
    # arcsin(x) = arctan(x / sqrt(1 - x^2)), but that denominator approaches zero
    # as |x| approaches 1, costing precision. The half-angle form is stable there:
    # arcsin(x) = 2 * arctan(x / (1 + sqrt(1 - x^2)))
    # (1 - x)(1 + x) rather than 1 - x*x avoids cancellation when |x| is near 1.
    return 2 * arctan(x / (1 + ((1 - x) * (1 + x)) ** 0.5))


def arccos(x):
    """Compute the angle of which cosine is x within (0, pi).

    Args:
        x (float): The ratio of adjacent to hypotenuse.

    Returns:
        float: The angle in radians.
    """
    # Check domain
    if not -1 <= x <= 1:
        raise ValueError("arccos domain is [-1, 1]")
    # Phase-shift arcsin
    return pi / 2 - arcsin(x)


def exp(x):
    """Compute e^x.

    Args:
        x (float): The exponent.

    Returns:
        float: The value of e raised to the power of x.
    """
    # Check for irreducible value
    if x == float("inf"):
        return x
    if x == float("-inf"):
        return 0.0
    # Range reduction via repeated halving
    # Count halvings as a power of 2
    exponent = 1
    while abs(x) >= 0.5:
        x /= 2
        exponent *= 2
    # Compute Taylor series incrementally
    term = 1
    total = 0.0
    for n in range(30):
        total += term
        term *= x / (n + 1)
    # Undo halvings
    return total**exponent


def differentiate(x):
    """Compute 5-point derivative for a given signal.

    Args:
        x (list): The input signal.

    Returns:
        list: The computed derivative.
    """
    n = len(x)

    def at(k):
        """Clamps indexes at edges."""
        return x[min(max(k, 0), n - 1)]

    # Compute rate of change across 5 points with PT-specified weights and normalize
    return [
        0.125 * (-at(i - 2) - 2 * at(i - 1) + 2 * at(i + 1) + at(i + 2))
        for i in range(n)
    ]


def integrate(squared, width):
    """Integrate/sum the signal across a window of the specified width. Expects squared input.

    Args:
        squared (list): The input signal (squared values).
        width (int): The width of the integration window.

    Returns:
        list: The integrated signal.
    """
    if width < 1:
        raise ValueError("width must be at least 1")
    out = []
    running = 0.0
    # Sum values within specified width
    for i, v in enumerate(squared):
        running += v  # Add new values
        # Subtract previous values
        if i >= width:
            running -= squared[i - width]
        out.append(running)
    return out


def median(values):
    """Find the median of a given set of values.

    Args:
        values (list): The input values.

    Returns:
        float: The median value.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median of empty sequence")
    n = len(ordered)
    mid = n // 2
    # Return single central value if possible
    if n % 2:
        return ordered[mid]
    # Return average of central values otherwise
    return (ordered[mid - 1] + ordered[mid]) / 2
