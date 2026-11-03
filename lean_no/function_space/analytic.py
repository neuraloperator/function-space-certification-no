import cmath
import inspect
import math
from dataclasses import dataclass

import torch

from lean_no.function_space.input import InputBounds, coord, cos_pi, sin_pi

@dataclass(frozen=True)
class Series:
    terms: tuple
    slopes: tuple

    @classmethod
    def from_samples(cls, samples):
        count = len(samples)
        if samples.ndim != 1 or count < 4 or count % 2:
            raise ValueError("an even periodic sample vector is required")
        spectrum = torch.fft.rfft(samples.double(), norm="forward")
        terms = [((0.0,), complex(spectrum[0]))]
        for mode, value in enumerate(spectrum[1:], 1):
            value = complex(value) * (0.5 if mode == count // 2 else 1)
            terms.extend(
                (((float(mode),), value), ((float(-mode),), value.conjugate()))
            )
        return cls(tuple(terms), (0.0,))

    def coerce(self, value):
        if isinstance(value, Series):
            return value
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TypeError(
                "analytic inputs support scalar arithmetic and torch.sin/cos"
            )
        return Series(
            ((tuple(0.0 for _ in self.slopes), complex(value)),),
            tuple(0.0 for _ in self.slopes),
        )

    def __add__(self, other):
        other = self.coerce(other)
        return Series(
            self.terms + other.terms,
            tuple(a + b for a, b in zip(self.slopes, other.slopes)),
        )

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) + -self

    def __mul__(self, other):
        other = self.coerce(other)
        if any(self.slopes) or any(other.slopes):
            source, factor = (self, other) if any(self.slopes) else (other, self)
            if any(factor.slopes) or any(any(k) for k, _ in factor.terms):
                raise TypeError(
                    "coordinate products are unsupported; use sine/cosine functions"
                )
            scalar = sum(c for _, c in factor.terms)
            if scalar.imag:
                raise TypeError("coordinates require real scalar coefficients")
            return Series(
                tuple((k, c * scalar) for k, c in source.terms),
                tuple(a * scalar.real for a in source.slopes),
            )
        terms = tuple(
            (tuple(x + y for x, y in zip(k, l)), a * b)
            for k, a in self.terms
            for l, b in other.terms
        )
        return Series(terms, self.slopes)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, (int, float)) or other == 0:
            raise TypeError(
                "analytic inputs support division by nonzero real constants"
            )
        return self * (1 / other)

    @classmethod
    def __torch_function__(cls, function, types, args=(), kwargs=None):
        if kwargs:
            raise TypeError("analytic input operations do not accept torch options")
        source = args[0]
        if function in (torch.ones_like, torch.zeros_like):
            return source.coerce(1 if function is torch.ones_like else 0)
        if function not in (torch.sin, torch.cos) or any(
            any(k) for k, _ in source.terms
        ):
            raise TypeError(
                "analytic inputs support torch.sin/cos of affine coordinates"
            )
        phase = sum(c for _, c in source.terms)
        if phase.imag:
            raise ValueError("input phases must be real")
        frequency = tuple(a / (2 * math.pi) for a in source.slopes)
        positive = cmath.exp(1j * phase.real) * (
            0.5 if function is torch.cos else -0.5j
        )
        return Series(
            (
                (frequency, positive),
                (tuple(-a for a in frequency), positive.conjugate()),
            ),
            tuple(0.0 for _ in frequency),
        )

    def magnitude(self, orders):
        return sum(
            abs(c)
            * math.prod(abs(2 * math.pi * k) ** n for k, n in zip(frequency, orders))
            for frequency, c in self.terms
        )

    def expression(self):
        if len(self.slopes) != 1:
            raise ValueError(
                "symbolic interval inputs currently require one coordinate"
            )
        result = coord() * self.slopes[0]
        for frequency, value in self.terms:
            phase = 2 * frequency[0] * coord()
            result += value.real * cos_pi(phase) - value.imag * sin_pi(phase)
        return result.simplify() + 0 * coord()

    def derivative_bounds(self, points, length, order):
        step = length / points.numel()
        node = torch.zeros_like(points)
        lower = torch.zeros_like(points)
        upper = torch.zeros_like(points)
        for frequency, coefficient in self.terms:
            omega = 2 * math.pi * frequency[0]
            coefficient *= (1j * omega) ** order
            if coefficient == 0:
                continue
            amplitude, phase = abs(coefficient), cmath.phase(coefficient)
            left = omega * points + phase
            right = left + omega * step
            a, b = torch.minimum(left, right), torch.maximum(left, right)
            start, end = torch.cos(left), torch.cos(right)
            low, high = torch.minimum(start, end), torch.maximum(start, end)
            period = 2 * math.pi
            has_max = torch.ceil(a / period) <= torch.floor(b / period)
            has_min = torch.ceil((a - math.pi) / period) <= torch.floor(
                (b - math.pi) / period
            )
            lower += amplitude * torch.where(has_min, -1.0, low)
            upper += amplitude * torch.where(has_max, 1.0, high)
            node += amplitude * start
        return node[None], node[None], lower[None], upper[None]

    def bounds(self, derivative_backend="trigonometric"):
        if derivative_backend not in ("trigonometric", "arb"):
            raise ValueError("input derivative backend must be trigonometric or arb")
        dimension = len(self.slopes)
        if dimension != 1 or any(self.slopes):
            raise ValueError(
                "continuum input bounds require a one-dimensional trigonometric function"
            )
        value = self.expression()
        return InputBounds.exact(
            value=(value,),
            supremum=[self.magnitude((0,))],
            lipschitz=[self.magnitude((1,))],
            curvature=[self.magnitude((2,))],
            derivative_enclosure=self.derivative_bounds
            if derivative_backend == "trigonometric"
            else None,
        )

    def field(self, lengths, device="cpu", dtype=torch.float64):
        from lean_no.graph import fourier

        if any(self.slopes):
            raise ValueError("exact Fourier inputs cannot contain raw coordinate terms")
        dimension = len(lengths)
        combined = {}
        for frequency, value in self.terms:
            key = tuple(k * length for k, length in zip(frequency, lengths))
            combined[key] = combined.get(key, 0j) + value
        combined = {k: c for k, c in combined.items() if c != 0}
        if not combined:
            combined = {tuple(0.0 for _ in lengths): 0j}
        combined = {k: c for k, c in combined.items() if k[-1] >= 0}
        axes = [sorted({k[a] for k in combined}) for a in range(dimension)]
        if dimension == 1:
            axes[0] = sorted(set(axes[0]) | {0.0})
        shape = tuple(map(len, axes))
        coefficients = torch.zeros((1, *shape), dtype=torch.complex128, device=device)
        positions = [{k: i for i, k in enumerate(axis)} for axis in axes]
        for key, value in combined.items():
            coefficients[(0, *(positions[a][k] for a, k in enumerate(key)))] = value
        modes = [torch.tensor(axis, device=device, dtype=dtype) for axis in axes]
        result = fourier(
            coefficients, modes, modes[0].new_zeros(1), {"lengths": tuple(lengths)}
        )
        result.data["coefficient_error"] = torch.zeros_like(coefficients.real)
        return result

def describe(function, channels=False):
    parameters = [
        p
        for p in inspect.signature(function).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    ]
    if len(parameters) not in (1, 2, 3):
        raise TypeError(
            "input functions require one, two, or three coordinate arguments"
        )
    dimension = len(parameters)
    variables = [
        Series((), tuple(float(a == b) for b in range(dimension)))
        for a in range(dimension)
    ]
    try:
        result = function(*variables)
        if channels and isinstance(result, (tuple, list)):
            result = tuple(variables[0].coerce(value) for value in result)
        else:
            result = variables[0].coerce(result)
    except (TypeError, ValueError, AttributeError) as error:
        raise TypeError(
            "unsupported analytic input; use scalar arithmetic and torch.sin/cos, or provide explicit input bounds"
        ) from error
    return result

def evaluate(function, coordinates):
    values = function(*coordinates)
    reference = coordinates[0]
    return torch.as_tensor(
        values, device=reference.device, dtype=reference.dtype
    ).expand_as(reference)
