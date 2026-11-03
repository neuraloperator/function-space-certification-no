import math
from dataclasses import dataclass

import torch

@dataclass
class CellEnclosure:
    lower: torch.Tensor
    upper: torch.Tensor
    lengths: tuple[float, ...]

    @property
    def cell_bounds(self):
        return self.lower, self.upper

    @property
    def width(self):
        return (self.upper - self.lower).amax()

    @property
    def channel_width(self):
        return (self.upper - self.lower).flatten(1).amax(1)

    def query(self, points):
        points = points.to(self.lower)
        if points.ndim != 2 or points.shape[1] != len(self.lengths):
            raise ValueError("queries must have shape [N, dimension]")
        indices = []
        for axis, (length, count) in enumerate(zip(self.lengths, self.lower.shape[1:])):
            if not math.isfinite(length) or length <= 0:
                raise ValueError("domain lengths must be positive and finite")
            coordinate = points[:, axis]
            if ((coordinate < 0) | (coordinate > length)).any():
                raise ValueError("query lies outside the physical domain")
            indices.append(
                (coordinate * count / length).floor().long().clamp(max=count - 1)
            )
        key = (slice(None), *indices)
        return self.lower[key].T, self.upper[key].T
