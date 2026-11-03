from collections.abc import Callable
from dataclasses import dataclass

import torch

def interval_linear(
    weight: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    multiply: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = weight.clamp(min=0)
    negative = weight.clamp(max=0)
    output_lower = multiply(positive, lower) + multiply(negative, upper)
    output_upper = multiply(positive, upper) + multiply(negative, lower)
    return output_lower, output_upper

def product_bounds(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.stack(
        (
            left_lower * right_lower,
            left_lower * right_upper,
            left_upper * right_lower,
            left_upper * right_upper,
        )
    )
    return values.amin(dim=0), values.amax(dim=0)

def square_bounds(lower, upper):
    low = torch.where(
        (lower <= 0) & (upper >= 0), 0.0, torch.minimum(lower.square(), upper.square())
    )
    return low, torch.maximum(lower.square(), upper.square())

@dataclass
class Interval:
    lower: torch.Tensor
    upper: torch.Tensor

    @property
    def actual(self):
        return self

    @property
    def center(self):
        return (self.lower + self.upper) / 2

    @classmethod
    def around(cls, center, radius=0):
        return cls(center - radius, center + radius)

    def __add__(self, other):
        if not isinstance(other, Interval):
            other = Interval.around(other)
        return Interval(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, Interval):
            other = Interval.around(other)
        products = torch.stack(
            [
                a * b
                for a in (self.lower, self.upper)
                for b in (other.lower, other.upper)
            ]
        )
        return Interval(products.amin(0), products.amax(0))

    __rmul__ = __mul__

    def square(self):
        return Interval(
            torch.where(
                (self.lower <= 0) & (self.upper >= 0),
                0.0,
                torch.minimum(self.lower.square(), self.upper.square()),
            ),
            self.magnitude().square(),
        )

    def magnitude(self):
        return torch.maximum(self.lower.abs(), self.upper.abs())

    def linear(self, weight):
        return Interval(
            *interval_linear(weight, self.lower, self.upper, lambda w, v: v @ w.T)
        )

    def intersect(self, other):
        return Interval(
            torch.maximum(self.lower, other.lower),
            torch.minimum(self.upper, other.upper),
        )
