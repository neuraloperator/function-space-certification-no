import itertools
import math
from dataclasses import dataclass

import torch

@dataclass
class BilinearBounds:

    lower: torch.Tensor
    upper: torch.Tensor
    lengths: tuple = ()

    @property
    def cell_bounds(self):
        return self.lower.amin(dim=0), self.upper.amax(dim=0)

    @property
    def functional_width(self):
        return (self.upper - self.lower).amax()

    @property
    def channel_width(self):
        return (self.upper - self.lower).flatten(2).amax(dim=(0, 2))

    def query(self, points):
        points = points.to(self.lower)
        if self.lengths:
            points = points / points.new_tensor(self.lengths)
        dimension = self.lower.ndim - 2
        if (
            points.ndim != 2
            or points.shape[1] != dimension
            or not torch.isfinite(points).all()
        ):
            raise ValueError(
                "queries must be finite [N, dimension] normalized coordinates"
            )
        if ((points < 0) | (points > 1)).any():
            raise ValueError("query lies outside the domain")
        shape = points.new_tensor(self.lower.shape[2:])
        scaled = points * shape
        indices = torch.minimum(scaled.floor(), shape - 1).long()
        fractions = scaled - indices
        weights = torch.stack(
            [
                math.prod(
                    fractions[:, a] if side else 1 - fractions[:, a]
                    for a, side in enumerate(sides)
                )
                for sides in itertools.product((0, 1), repeat=dimension)
            ],
            1,
        )

        def interpolate(values):
            corners = values[(slice(None), slice(None), *indices.T)].permute(2, 0, 1)
            return (corners * weights[:, :, None]).sum(1)

        return interpolate(self.lower), interpolate(self.upper)

@dataclass(frozen=True)
class BilinearBand:

    nominal: BilinearBounds
    epsilon: float | torch.Tensor
    constant_in_time: bool = False
    fixed_input: bool = False
