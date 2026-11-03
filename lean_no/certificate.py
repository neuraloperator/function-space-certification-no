from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lean_no.domain import DiscreteGrid, FourierGrid, grid_dimension

if TYPE_CHECKING:
    from lean_no.function_space.spherical import SphereGrid
    from lean_no.inputs import InputSet
    from lean_no.methods import Method

@dataclass(frozen=True)
class QueryBounds:

    center: torch.Tensor
    radius: torch.Tensor

    @classmethod
    def from_bounds(cls, lower, upper):
        return cls(lower / 2 + upper / 2, upper / 2 - lower / 2)

    @property
    def lower(self):
        return self.center - self.radius

    @property
    def upper(self):
        return self.center + self.radius

    @property
    def width(self):
        return 2 * self.radius

    def __iter__(self):
        yield self.lower
        yield self.upper

@dataclass(frozen=True)
class Certificate:

    enclosure: object
    grid: FourierGrid | DiscreteGrid | SphereGrid
    input_bounds: InputSet
    method: Method
    diagnostics: dict

    quantity: str = "output"
    roundoff_enclosed: bool = False
    derivatives: tuple = ()
    pointwise_width: torch.Tensor | None = None
    cell_range_width: torch.Tensor | None = None
    node_width: torch.Tensor | None = None
    query_width: torch.Tensor | None = None

    @property
    def domain(self):
        return self.grid.domain if hasattr(self.grid, "domain") else self.grid

    @torch.no_grad()
    def query(self, points):
        if isinstance(self.grid, DiscreteGrid):
            return QueryBounds.from_bounds(*self.enclosure.query(points))
        dimension = grid_dimension(self.grid)
        if (
            not isinstance(points, torch.Tensor)
            or points.ndim != 2
            or points.shape[1] != dimension
            or points.is_complex()
            or not torch.isfinite(points).all()
        ):
            raise ValueError(
                "queries must be finite real points with shape [N, dimension]"
            )
        result = self.enclosure.query(points)
        return (
            result
            if isinstance(result, QueryBounds)
            else QueryBounds.from_bounds(*result)
        )
