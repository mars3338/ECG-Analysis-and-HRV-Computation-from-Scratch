"""Complex numbers and arithmetic."""

import math_utilities as mu


class Complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def _format(self, value):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def __repr__(self):
        if self.imag >= 0:
            return f"({self._format(self.real)} + {self._format(self.imag)}i)"
        else:
            return f"({self._format(self.real)} - {self._format(-self.imag)}i)"

    def _ensure_complex(self, other):
        """Formats int/float operands as complex numbers to avoid special cases in arithmetic methods."""
        if isinstance(other, (int, float)):
            return Complex(other, 0)
        return other

    def __add__(self, other):
        other = self._ensure_complex(other)
        return Complex(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = self._ensure_complex(other)
        return Complex(self.real - other.real, self.imag - other.imag)

    def __rsub__(self, other):
        other = self._ensure_complex(other)
        return other - self

    def __mul__(self, other):
        other = self._ensure_complex(other)
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return Complex(real_part, imag_part)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self._ensure_complex(other)
        if other.real == 0 and other.imag == 0:
            raise ZeroDivisionError("division by complex zero")
        conjugate = Complex(other.real, -other.imag)
        # Multiply top and bottom by the conjugate
        numerator = self * conjugate
        denominator = (
            other.real**2 + other.imag**2
        )  # Shortcut: (c+di)(c-di) = c^2 + d^2
        # Real scalar allows for plain division:
        return Complex(numerator.real / denominator, numerator.imag / denominator)

    def __rtruediv__(self, other):
        other = self._ensure_complex(other)
        return other / self

    def magnitude(self):
        """Compute the distance from the origin (magnitude)."""
        return (self.real**2 + self.imag**2) ** 0.5

    def phase(self):
        """Compute the angle from the positive real axis (phase)."""
        # Avoid division by 0
        if self.real == 0:
            if self.imag > 0:
                return mu.pi / 2
            elif self.imag < 0:
                return -mu.pi / 2
            else:
                return 0
        angle = mu.arctan(self.imag / self.real)
        # As arctan sees only the ratio of imag/real, it can't tell quadrant 2 from 4 or 3 from 1
        # Add +/- pi when real < 0 to recover the true quadrant
        if self.real < 0:
            angle += mu.pi if self.imag >= 0 else -mu.pi
        return angle

    def exp(self):
        """Format the complex number as an exponential."""
        # Euler: e^(a+bi) = e^a * (cos b + i sin b)
        r = mu.exp(self.real)
        return Complex(r * mu.cos(self.imag), r * mu.sin(self.imag))
