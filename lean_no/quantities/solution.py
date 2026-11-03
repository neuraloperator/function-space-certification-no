import math
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class SolutionBound:

    bound_expression_lower: float
    bound_expression_upper: float
    norm: str = "linf"
    initial_contribution: tuple[float, float] | None = None
    residual_contribution: tuple[float, float] | None = None
    accumulated_exponent: tuple[float, float] | None = None

    def __post_init__(self):
        if self.norm not in ("linf", "l2"):
            raise ValueError("solution norm must be linf or l2")
        if not (
            math.isfinite(self.bound_expression_lower)
            and math.isfinite(self.bound_expression_upper)
            and 0 <= self.bound_expression_lower <= self.bound_expression_upper
        ):
            raise ValueError(
                "bound expression must have finite ordered nonnegative endpoints"
            )

    @property
    def bound_expression_width(self):
        return self.bound_expression_upper - self.bound_expression_lower

    @property
    def solution_error_upper(self):
        return self.bound_expression_upper

    def report(self):
        result = {
            "bound_expression_lower": self.bound_expression_lower,
            "bound_expression_upper": self.bound_expression_upper,
            "bound_expression_width": self.bound_expression_width,
            f"solution_{self.norm}_upper": self.solution_error_upper,
        }
        for name in (
            "initial_contribution",
            "residual_contribution",
            "accumulated_exponent",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

def linf_interval(lower, upper):
    lower = (
        lower
        if isinstance(lower, torch.Tensor)
        else torch.as_tensor(lower, dtype=torch.float64)
    )
    upper = torch.as_tensor(upper, device=lower.device, dtype=lower.dtype)
    if lower.shape != upper.shape or not lower.numel():
        raise ValueError("matching nonempty interval arrays required")
    if not bool(
        (torch.isfinite(lower) & torch.isfinite(upper) & (lower <= upper)).all()
    ):
        raise ValueError("intervals must be finite and ordered")
    return (
        float(torch.maximum(lower, -upper).clamp_min(0).amax()),
        float(torch.maximum(lower.abs(), upper.abs()).amax()),
    )

def residual_norm_interval(residual, norm):
    from lean_no.quantities.norms import ResidualBounds

    if norm == "linf":
        return (
            residual.residual_linf_interval
            if isinstance(residual, ResidualBounds)
            else linf_interval(residual.lower, residual.upper)
        )
    if norm != "l2":
        raise ValueError("solution norm must be linf or l2")
    upper = residual.residual_l2
    if upper is None or not math.isfinite(upper) or upper < 0:
        raise ValueError("finite nonnegative residual L2 bound required")
    if isinstance(residual, ResidualBounds) and residual.cells is None:
        return residual.residual_l2_lower or 0.0, upper
    _, peak = linf_interval(residual.lower, residual.upper)
    if peak == 0:
        return 0.0, upper
    # Scale before squaring to avoid overflow. CPU float64 also supports MPS.
    magnitude = torch.maximum(residual.lower, -residual.upper).clamp_min(0)
    magnitude = magnitude.to(device="cpu", dtype=torch.float64) / peak
    lengths = residual.length
    volume = lengths if isinstance(lengths, (float, int)) else math.prod(lengths)
    lower = peak * math.sqrt(float(magnitude.square().mean())) * math.sqrt(volume)
    return lower, upper

def poisson_solution_bound(residual, norm="linf"):
    if residual.pde != "poisson":
        raise ValueError("Poisson stability applies only to Poisson certificates")
    lengths = residual.length
    if isinstance(lengths, (float, int)):
        lengths = (lengths,)
    if len(lengths) not in (1, 2) or any(
        not math.isfinite(v) or v <= 0 for v in lengths
    ):
        raise ValueError("one or two finite positive domain lengths required")
    from lean_no.quantities.norms import ResidualBounds

    rlo, rhi = residual_norm_interval(residual, norm)
    if isinstance(residual, ResidualBounds):
        blo, bhi = residual.boundary_linf_interval
    else:
        blo, bhi = linf_interval(residual.boundary_lower, residual.boundary_upper)
    if norm == "l2":
        constant = 1 / (math.pi**2 * sum(length**-2 for length in lengths))
        volume_scale = math.sqrt(math.prod(lengths))
        blo, bhi = volume_scale * blo, volume_scale * bhi
    else:
        constant = min(lengths) ** 2 / 8
    return SolutionBound(blo + constant * rlo, bhi + constant * rhi, norm=norm)

def helmholtz_stability_constants(kappa, length=1.0, norm="linf"):
    if norm not in ("linf", "l2"):
        raise ValueError("solution norm must be linf or l2")
    if (
        kappa is None
        or not math.isfinite(kappa)
        or kappa <= 0
        or not math.isfinite(length)
        or length <= 0
    ):
        raise ValueError("positive finite kappa and domain length required")
    phase = kappa * length / 2
    if not math.isfinite(phase):
        raise ValueError("finite kappa times domain length required")
    turns = phase / math.pi
    if turns.is_integer():
        raise ValueError("periodic Helmholtz operator is resonant")
    denominator = abs(math.sin(phase))
    if norm == "l2":
        # Only the two modes bracketing kappa can minimize the spectral gap.
        lower_mode = math.floor(turns)
        delta = min(
            abs((2 * math.pi * n / length) ** 2 - kappa**2)
            for n in (lower_mode, lower_mode + 1)
        )
        if delta == 0:
            raise ValueError("periodic Helmholtz operator is resonant")
        constant = 1 / delta
        boundary_scale = math.sqrt(length)
    else:
        periods = math.floor(turns)
        tail = phase - periods * math.pi
        integral = 2 * periods + (
            math.sin(tail) if tail <= math.pi / 2 else 2 - math.sin(tail)
        )
        constant = integral / (kappa**2 * denominator)
        boundary_scale = 1.0
    return (
        constant,
        boundary_scale / (2 * denominator),
        boundary_scale / (2 * kappa * denominator),
    )

def helmholtz_solution_bound(residual, norm="linf"):
    if residual.pde != "helmholtz":
        raise ValueError("Helmholtz stability applies only to Helmholtz certificates")
    lengths = residual.length
    if isinstance(lengths, (float, int)):
        lengths = (lengths,)
    if len(lengths) != 1:
        raise ValueError("periodic Helmholtz stability requires a 1D interval")
    constants = helmholtz_stability_constants(residual.kappa, lengths[0], norm=norm)
    if (
        residual.boundary_lower is None
        or residual.boundary_upper is None
        or len(residual.boundary_lower) != 2
        or len(residual.boundary_upper) != 2
    ):
        raise ValueError("separate periodic value and derivative jumps required")
    residual_interval = residual_norm_interval(residual, norm)
    intervals = [residual_interval] + [
        linf_interval(lo, hi)
        for lo, hi in zip(residual.boundary_lower, residual.boundary_upper)
    ]
    return SolutionBound(
        *(
            sum(
                constant * interval[side]
                for constant, interval in zip(constants, intervals)
            )
            for side in (0, 1)
        ),
        norm=norm,
    )

def burgers_solution_bound(residual):
    if residual.pde != "burgers":
        raise ValueError("Burgers stability requires a Burgers certificate")
    data = getattr(residual, "burgers", None)
    if data is None:
        raise ValueError("Burgers certificate is missing stability data")
    if tuple(residual.length) != (1.0, data.time_period):
        raise ValueError("Burgers stability geometry disagrees with residual geometry")
    initial = list(data.initial_l2)
    forced, exponent = [0.0, 0.0], [0.0, 0.0]
    try:
        for left, right, growth, forcing in zip(
            data.time_edges, data.time_edges[1:], data.energy_growth, data.residual_l2
        ):
            dt = right - left
            for side in (0, 1):
                a, r = growth[side], forcing[side]
                z = a * dt
                exponent[side] += z
                if initial[side] == 0 and forced[side] == 0 and r == 0:
                    continue
                gain = dt if a == 0 else math.expm1(z) / a
                initial[side] *= math.exp(z)
                forced[side] = math.exp(z) * forced[side] + gain * r
                if not math.isfinite(initial[side] + forced[side]):
                    raise OverflowError
    except OverflowError as error:
        raise FloatingPointError(
            "Burgers stability amplification is nonfinite; refine the certificate"
        ) from error
    return SolutionBound(
        *(a + b for a, b in zip(initial, forced)),
        norm="l2",
        initial_contribution=tuple(initial),
        residual_contribution=tuple(forced),
        accumulated_exponent=tuple(exponent),
    )

def solution_bound(certificate, norm=None):
    from lean_no.quantities.norms import ResidualBounds, ResidualResult

    if norm not in (None, "linf", "l2"):
        raise ValueError("solution norm must be linf or l2")
    residual = certificate
    if isinstance(certificate, ResidualResult):
        residual = certificate.bounds
    if not isinstance(residual, ResidualBounds):
        raise TypeError("solution_bound requires a residual certificate")
    if residual.pde == "poisson":
        return poisson_solution_bound(residual, norm=norm or "linf")
    if residual.pde == "helmholtz":
        return helmholtz_solution_bound(residual, norm=norm or "linf")
    if residual.pde == "burgers":
        if norm == "linf":
            raise ValueError("Burgers solution stability supports only l2")
        return burgers_solution_bound(residual)
    raise NotImplementedError(
        f"{residual.pde} residual-to-solution stability conversion is not implemented"
    )
