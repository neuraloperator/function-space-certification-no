import math
from dataclasses import dataclass

import torch

from lean_no.bounds.activations import gelu_derivative, gelu_high_caps, gelu_range
from lean_no.bounds.interval import Interval

def derivative(x, order):
    if order <= 4:
        return gelu_derivative(x, order)
    density = (-x.square() / 2).exp() / math.sqrt(2 * math.pi)
    if order == 5:
        return (x**5 - 11 * x**3 + 18 * x) * density
    if order == 6:
        return (-(x**6) + 16 * x**4 - 51 * x.square() + 18) * density
    if order == 7:
        return (x**7 - 22 * x**5 + 115 * x**3 - 120 * x) * density
    raise ValueError("derivative order exceeds seven")

def derivative_magnitude(interval, order):
    if order <= 4:
        return gelu_range(interval, order).magnitude()
    try:
        global_bound = gelu_high_caps[order]
    except KeyError:
        raise ValueError("derivative order exceeds seven") from None
    near = torch.where(
        (interval.lower <= 0) & (interval.upper >= 0),
        0.0,
        torch.minimum(interval.lower.abs(), interval.upper.abs()),
    )
    y = interval.square()
    if order == 5:
        polynomial = (interval * (y * (y - 11) + 18)).magnitude()
    elif order == 6:
        polynomial = (y * (y * (-y + 16) - 51) + 18).magnitude()
    else:
        polynomial = (interval * (y * (y * (y - 22) + 115) - 120)).magnitude()
    local = polynomial * (-near.square() / 2).exp() / math.sqrt(2 * math.pi)
    local = torch.nan_to_num(local, nan=global_bound, posinf=global_bound)
    return local.clamp(max=global_bound)

@dataclass
class DualForm:

    c: torch.Tensor
    g: torch.Tensor
    r: torch.Tensor  # [cells, channels, nominal/actual]
    h: torch.Tensor

    @property
    def actual(self):
        return self.lane(1).interval()

    @property
    def center(self):
        return self.c

    def lane(self, index):
        return Lane(self, index)

    @classmethod
    def constant(cls, c, h):
        return cls(
            c,
            c.new_zeros(*c.shape, h.shape[-1]),
            c.new_zeros(*c.shape, 2),
            h,
        )

    def linear_radius(self):
        return (self.g.abs() * self.h[:, None]).sum(-1)

    def radius(self):
        return self.linear_radius()[..., None] + self.r

    def interval(self):
        return Interval.around(self.c[..., None], self.radius())

    def __add__(self, other):
        if not isinstance(other, DualForm):
            return DualForm(self.c + other, self.g, self.r, self.h)
        return DualForm(
            self.c + other.c,
            self.g + other.g,
            self.r + other.r,
            self.h,
        )

    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, DualForm):
            return DualForm(
                self.c * other,
                self.g * other,
                self.r * abs(other),
                self.h,
            )
        l1, l2 = self.linear_radius()[..., None], other.linear_radius()[..., None]
        # Products of linear terms are bounded immediately, retaining the
        # cancellation between the two mixed-coordinate contributions.
        product = torch.zeros_like(self.c)
        for left in range(self.h.shape[-1]):
            product += (self.g[..., left] * other.g[..., left]).abs() * self.h[
                :, left, None
            ].square()
            for right in range(left + 1, self.h.shape[-1]):
                product += (
                    self.g[..., left] * other.g[..., right]
                    + self.g[..., right] * other.g[..., left]
                ).abs() * (self.h[:, left, None] * self.h[:, right, None])
        product = product[..., None]
        return DualForm(
            self.c * other.c,
            self.c[..., None] * other.g + other.c[..., None] * self.g,
            self.c.abs()[..., None] * other.r
            + other.c.abs()[..., None] * self.r
            + l1 * other.r
            + l2 * self.r
            + self.r * other.r
            + product,
            self.h,
        )

    __rmul__ = __mul__

    def spatial_square_polynomial(self, spatial_axis=0):
        time_axis = 1 - spatial_axis
        c = self.c[:, 0]
        gx, gt = self.g[:, 0, spatial_axis], self.g[:, 0, time_axis]
        hx, ht = self.h[:, spatial_axis], self.h[:, time_axis]
        radius = self.r[:, 0, 1]
        magnitude = c.abs() + gx.abs() * hx + gt.abs() * ht
        error = 2 * magnitude * radius + radius.square()
        return torch.stack(
            (
                c.square() + gx.square() * hx.square() / 3,
                2 * c * gt,
                gt.square(),
                error,
            ),
            -1,
        )

    def square(self):
        return self * self

    def linear(self, weight, bias=0):
        return DualForm(
            self.c @ weight.T + bias,
            torch.einsum("ncd,oc->nod", self.g, weight),
            torch.einsum("nci,oc->noi", self.r, weight.abs()),
            self.h,
        )

def compose(value, order, derivative_fn=derivative, magnitude_fn=derivative_magnitude):
    center = derivative_fn(value.c, order)
    first, second = derivative_fn(value.c, order + 1), derivative_fn(value.c, order + 2)
    third = magnitude_fn(value.interval(), order + 3)
    return DualForm(
        center,
        first[..., None] * value.g,
        first.abs()[..., None] * value.r
        + second.abs()[..., None] * value.radius().square() / 2
        + third * value.radius() ** 3 / 6,
        value.h,
    )

@dataclass(frozen=True)
class Lane:
    form: DualForm
    index: int

    @property
    def r(self):
        return self.form.r[..., self.index]

    def interval(self):
        return Interval.around(self.form.c, self.form.radius()[..., self.index])

def tanh_polynomial(order):
    result = [0, 1]
    for _ in range(order):
        following = [0] * (len(result) + 1)
        for power, coefficient in enumerate(result[1:], 1):
            following[power - 1] += power * coefficient
            following[power + 1] -= power * coefficient
        result = following
    return result

def tanh_derivative(value, order):
    result = torch.zeros_like(value)
    t = value.tanh()
    for coefficient in reversed(tanh_polynomial(order)):
        result = result * t + coefficient
    return result

def tanh_derivative_magnitude(interval, order):
    t = Interval(interval.lower.tanh(), interval.upper.tanh())
    result = Interval.around(torch.zeros_like(interval.lower))
    polynomial = tanh_polynomial(order)
    for coefficient in reversed(polynomial):
        result = result * t + coefficient
    # |tanh z| <= 1 on the real axis; both bounds are analytic.
    return result.magnitude().clamp(max=sum(abs(c) for c in polynomial))

def tanh_compose(form, order):
    return compose(form, order, tanh_derivative, tanh_derivative_magnitude)
