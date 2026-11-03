import math
from dataclasses import dataclass, field

import torch

@dataclass(frozen=True)
class BurgersStability:

    viscosity: float
    time_period: float
    time_edges: tuple[float, ...]
    initial_l2: tuple[float, float]
    residual_l2: tuple[tuple[float, float], ...]
    energy_growth: tuple[tuple[float, float], ...]
    periodic: bool
    complete: bool

    def __post_init__(self):
        if not math.isfinite(self.viscosity) or self.viscosity <= 0:
            raise ValueError("positive finite viscosity required for viscous Burgers")
        if not math.isfinite(self.time_period) or self.time_period <= 0:
            raise ValueError("positive finite time period required")
        edges = self.time_edges
        if (
            len(edges) < 2
            or edges[0] != 0
            or edges[-1] > self.time_period
            or any(not math.isfinite(t) for t in edges)
            or any(b <= a for a, b in zip(edges, edges[1:]))
        ):
            raise ValueError(
                "ordered physical time edges from zero through the horizon required"
            )
        if len(self.residual_l2) != len(edges) - 1 or len(self.energy_growth) != len(
            self.residual_l2
        ):
            raise ValueError(
                "one residual L2 and energy-growth interval per slab required"
            )
        for pair in (self.initial_l2, *self.residual_l2, *self.energy_growth):
            if (
                len(pair) != 2
                or not all(math.isfinite(v) for v in pair)
                or not 0 <= pair[0] <= pair[1]
            ):
                raise ValueError(
                    "finite ordered nonnegative L2 and energy-growth intervals required"
                )
        if not self.periodic or not self.complete:
            raise ValueError(
                "verified spatial periodicity and complete slab coverage required"
            )

@dataclass(frozen=True)
class ResidualBounds:
    pde: str
    residual_linf_interval: tuple[float, float]
    boundary_linf_interval: tuple[float, float]
    length: tuple[float, ...]
    coverage: dict
    kappa: float | None = None
    boundary_lower: object = None
    boundary_upper: object = None
    burgers: BurgersStability | None = None
    residual_l2: float | None = None
    cells: object = None
    residual_l2_lower: float | None = None

    def __post_init__(self):
        if self.residual_l2_lower is not None and (
            self.residual_l2 is None
            or not math.isfinite(self.residual_l2_lower)
            or not 0 <= self.residual_l2_lower <= self.residual_l2
        ):
            raise ValueError("residual L2 endpoints must be finite and ordered")
        for low, high in (self.residual_linf_interval, self.boundary_linf_interval):
            if (
                not math.isfinite(low)
                or not math.isfinite(high)
                or not 0 <= low <= high
            ):
                raise ValueError(
                    "norm intervals must be finite, ordered and nonnegative"
                )
        if self.residual_l2 is not None and (
            not math.isfinite(self.residual_l2) or self.residual_l2 < 0
        ):
            raise ValueError("finite nonnegative residual L2 bound required")
        if not self.coverage.get("residual", {}).get("complete", False):
            raise ValueError("a complete residual cover is required")
        if self.pde != "ks" and not self.coverage.get("boundary", {}).get(
            "complete", False
        ):
            raise ValueError("a complete boundary cover is required")
        if (self.boundary_lower is None) != (self.boundary_upper is None):
            raise ValueError("both signed boundary endpoints are required")
        if self.boundary_lower is not None:
            lo, hi = (
                torch.as_tensor(self.boundary_lower),
                torch.as_tensor(self.boundary_upper),
            )
            if (
                lo.shape != hi.shape
                or not lo.numel()
                or not torch.isfinite(lo).all()
                or not torch.isfinite(hi).all()
                or (lo > hi).any()
            ):
                raise ValueError("boundary intervals must be finite and ordered")

    def report(self):
        result = dict(
            pde=self.pde,
            residual_linf_interval=self.residual_linf_interval,
            residual_l2_upper=self.residual_l2,
            boundary_linf_interval=self.boundary_linf_interval,
            coverage=self.coverage,
        )
        if self.residual_l2_lower is not None:
            result["residual_l2_lower"] = self.residual_l2_lower
        if self.burgers is not None:
            result["residual_l2_max_interval"] = [
                max(pair[k] for pair in self.burgers.residual_l2) for k in (0, 1)
            ]
            result["initial_l2_error_interval"] = self.burgers.initial_l2
        return result

    @property
    def residual_linf(self):
        return self.residual_linf_interval[1]

    @property
    def lower(self):
        return self.cells.lower

    @property
    def upper(self):
        return self.cells.upper

    @property
    def boundary_bounds(self):
        lo, hi = (
            torch.as_tensor(self.boundary_lower),
            torch.as_tensor(self.boundary_upper),
        )
        return torch.maximum(lo.abs(), hi.abs())

@dataclass(frozen=True)
class ResidualResult:
    bounds: ResidualBounds
    field: object
    grid: object
    inputs: object
    method: object
    forcing: object = None
    diagnostics: dict = field(default_factory=dict)
    roundoff_enclosed: bool = False

    @property
    def domain(self):
        return self.grid.domain

    @property
    def quantity(self):
        return self.bounds.pde + "_residual"

    def query(self, points):
        from lean_no.certificate import QueryBounds

        if self.bounds.cells is not None:
            return QueryBounds.from_bounds(*self.bounds.cells.query(points))
        from lean_no.quantities import Residual
        from lean_no.quantities.residual import residual_form

        reference = (
            self.field.forcing.error
            if hasattr(self.field, "forcing")
            else self.field.error
        )
        centers = points.to(reference) / reference.new_tensor(self.grid.lengths)
        form = residual_form(
            self.field,
            centers,
            torch.zeros_like(centers),
            Residual(self.bounds.pde, self.bounds.kappa),
            self.forcing,
        )
        interval = form.actual
        return QueryBounds.from_bounds(interval.lower, interval.upper)
