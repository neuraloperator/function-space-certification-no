import math
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class PeriodicBox:
    lengths: tuple[float, ...]
    axes: tuple[str, ...] = ()
    periodic: tuple[bool, ...] = ()
    active: tuple[float, ...] = ()

    def __post_init__(self):
        dimension = len(self.lengths)
        if not dimension or any(
            not math.isfinite(length) or length <= 0 for length in self.lengths
        ):
            raise ValueError("positive finite domain lengths required")
        if self.axes and (
            len(self.axes) != dimension or len(set(self.axes)) != dimension
        ):
            raise ValueError(
                "coordinate names must be distinct and match the dimension"
            )
        if self.periodic and len(self.periodic) != dimension:
            raise ValueError("periodic axes must match the dimension")
        if self.active and (
            len(self.active) != dimension
            or any(not 0 < a <= b for a, b in zip(self.active, self.lengths))
        ):
            raise ValueError("active domain must lie inside its periods")

    def normalize(self, points):
        if (
            points.ndim != 2
            or points.shape[1] != len(self.lengths)
            or points.is_complex()
            or not torch.isfinite(points).all()
        ):
            raise ValueError("finite real [count,dimension] coordinates required")
        edge = points.new_tensor(self.active or self.lengths)
        if ((points < 0) | (points > edge)).any():
            raise ValueError("query lies outside the domain")
        return points / points.new_tensor(self.lengths)

def periodic_grid(count, length=1.0, dtype=torch.float32, device=None):
    return torch.linspace(0, length, count + 1, dtype=dtype, device=device)[:-1]

@dataclass(frozen=True)
class FourierGrid:

    resolution: tuple[int, ...]
    lengths: tuple[float, ...] = (1.0,)
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        if len(self.resolution) != len(self.lengths) or any(
            type(n) is not int or n < 1 for n in self.resolution
        ):
            raise ValueError("grid resolution must match its domain")
        PeriodicBox(self.lengths)

    @property
    def domain(self):
        return PeriodicBox(self.lengths)

@dataclass(frozen=True)
class DiscreteGrid:

    resolution: tuple[int, ...]

    def __post_init__(self):
        if (
            not isinstance(self.resolution, tuple)
            or not self.resolution
            or any(type(size) is not int or size < 1 for size in self.resolution)
        ):
            raise ValueError("discrete resolution must contain positive integers")

def grid_dimension(grid):
    if isinstance(grid, (FourierGrid, DiscreteGrid)):
        return len(grid.resolution)
    return 2
