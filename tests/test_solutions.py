import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from scipy.integrate import quad

from lean_no.quantities.norms import BurgersStability, ResidualBounds
from lean_no.quantities.periodic_energy import periodic_energy_bounds
from lean_no.quantities.solution import (
    SolutionBound,
    helmholtz_stability_constants,
    linf_interval,
    poisson_solution_bound,
    solution_bound,
)

def poisson_certificate(lower, upper, boundary_lower, boundary_upper, length=(1.0,)):
    return SimpleNamespace(
        pde="poisson",
        lower=lower,
        upper=upper,
        boundary_lower=boundary_lower,
        boundary_upper=boundary_upper,
        length=length,
    )

def test_exact_quadratic_saturates_expression():
    # e=1+2*x*(1-x), hence B=1, R=4 and Q=1.5.
    result = poisson_solution_bound(poisson_certificate([-4], [-4], [1, 1], [1, 1]))
    assert result == SolutionBound(1.5, 1.5)
    assert result.report()["bound_expression_width"] == 0
    assert "solution_linf_lower" not in result.report()

def test_interval_signs_and_residual_uncertainty():
    result = poisson_solution_bound(poisson_certificate([-8], [-4], [-3, 1], [-2, 4]))
    assert result == SolutionBound(2.5, 5)
    assert linf_interval([-1, -2], [1, 2]) == (0, 2)

def test_rectangle_scaling_and_boundary_edge_arrays():
    # e=2*y*(2-y) on [0,3]x[0,2]. R=4, B=2 on vertical edges.
    # Exact Q is 2 + 4*(2**2/8) = 4, despite actual error being 2.
    result = poisson_solution_bound(
        poisson_certificate(
            [-4],
            [-4],
            [[0, 2, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 2, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0]],
            length=(3.0, 2.0),
        )
    )
    assert result == SolutionBound(4, 4)

def test_norm_only_poisson_l2_uses_computed_lower_endpoint():
    from dataclasses import replace

    certificate = replace(
        norm_certificate("poisson", lengths=(1.0, 1.0), r=4.0),
        residual_l2_lower=2.0,
    )
    result = poisson_solution_bound(certificate, norm="l2")
    assert result.bound_expression_lower == pytest.approx(1 / math.pi**2)
    assert result.bound_expression_upper == pytest.approx(2 / math.pi**2)

def test_cell_l2_floor_uses_whole_cells_not_point_witnesses():
    from types import SimpleNamespace

    from lean_no.bounds.interval import Interval
    from lean_no.quantities.cells import cell_norms, evaluate_cells

    centers = torch.arange(3, dtype=torch.float64)[:, None]

    def evaluate(points, widths):
        ids = points[:, 0].long()
        lo = torch.tensor([2.0, -4.0, -1.0], dtype=torch.float64)[ids]
        hi = torch.tensor([3.0, -2.0, 1.0], dtype=torch.float64)[ids]
        return SimpleNamespace(actual=Interval(lo, hi), center=(lo + hi) / 2)

    upper, _, floor = evaluate_cells(
        centers, torch.ones_like(centers), evaluate, return_floor=True
    )
    assert floor.tolist() == [2.0, 2.0, 0.0]
    assert cell_norms(floor, floor, 3.0)[1] == pytest.approx(math.sqrt(8))
    assert cell_norms(-upper, upper, 3.0)[1] == pytest.approx(math.sqrt(26))

def norm_certificate(
    pde, lengths=(1.0,), r=0.0, boundary=0.0, kappa=None, jumps=(0.0, 0.0)
):
    return ResidualBounds(
        pde,
        (0.0, r),
        (boundary, boundary),
        lengths,
        {"residual": {"complete": True}, "boundary": {"complete": True}},
        kappa=kappa,
        boundary_lower=jumps,
        boundary_upper=jumps,
        residual_l2=r,
    )

@pytest.mark.parametrize("lengths", [(0.7, 2.3)])
def test_poisson_eigenmode_and_constant_boundary_error(lengths):
    # e=B+product(sin(pi*x_i/L_i)): ||e-B||2=sqrt(volume/2**d).
    mode_norm = math.sqrt(math.prod(lengths) / 2 ** len(lengths))
    eigenvalue = math.pi**2 * sum(length**-2 for length in lengths)
    item = norm_certificate("poisson", lengths, r=eigenvalue * mode_norm)
    result = solution_bound(item, norm="l2")
    assert result.solution_error_upper == pytest.approx(mode_norm)
    shifted = solution_bound(
        replace(item, boundary_linf_interval=(0.3, 0.3)), norm="l2"
    )
    assert shifted.solution_error_upper == pytest.approx(
        mode_norm + 0.3 * math.sqrt(math.prod(lengths))
    )
    actual_squared = (
        0.3**2 * math.prod(lengths)
        + 0.6 * math.prod(2 * length / math.pi for length in lengths)
        + mode_norm**2
    )
    assert math.sqrt(actual_squared) <= shifted.solution_error_upper
    assert result.norm == "l2"
    assert "solution_l2_upper" in result.report()
    assert solution_bound(item).norm == "linf"

@pytest.mark.parametrize("kappa,length", [(0.7, 1.0), (2 * math.pi + 1e-5, 1.0)])
def test_helmholtz_nearest_eigenmode_saturates_bound(kappa, length):
    n = min(range(32), key=lambda n: abs((2 * math.pi * n / length) ** 2 - kappa**2))
    error_norm = math.sqrt(length if n == 0 else length / 2)
    delta = abs((2 * math.pi * n / length) ** 2 - kappa**2)
    item = norm_certificate("helmholtz", (length,), r=delta * error_norm, kappa=kappa)
    assert solution_bound(item, norm="l2").solution_error_upper == pytest.approx(
        error_norm
    )

@pytest.mark.parametrize("length,cosine,sine", [(2.3, 0.3, -0.7)])
def test_helmholtz_homogeneous_boundary_lift(length, cosine, sine):
    kappa = 5.0
    phase = kappa * length / 2
    jumps = (-2 * sine * math.sin(phase), 2 * kappa * cosine * math.sin(phase))
    item = norm_certificate("helmholtz", (length,), kappa=kappa, jumps=jumps)
    bound = solution_bound(item, norm="l2").solution_error_upper
    actual = math.sqrt(
        quad(
            lambda x: (
                (
                    cosine * math.cos(kappa * (x - length / 2))
                    + sine * math.sin(kappa * (x - length / 2))
                )
                ** 2
            ),
            0,
            length,
        )[0]
    )
    assert actual <= bound
    assert bound == pytest.approx(math.sqrt(length) * (abs(cosine) + abs(sine)))

@pytest.mark.parametrize("kappa,length", [(5.0, 2.3)])
def test_inverse_norm_against_independent_quadrature(kappa, length):
    constant, _, _ = helmholtz_stability_constants(kappa, length)
    a = kappa * length / 2
    zeros = [math.pi / 2 + i * math.pi for i in range(math.ceil(a / math.pi))]
    integral, _ = quad(
        lambda t: abs(math.cos(t)), 0, a, points=[t for t in zeros if t < a]
    )
    assert constant == pytest.approx(
        integral / (kappa**2 * abs(math.sin(a))), rel=2e-13
    )

def burgers_certificate(
    e=(0.0, 0.0), r=((0.0, 0.0),), a=((0.0, 0.0),), edges=(0.0, 1.0)
):
    from lean_no.quantities.norms import ResidualBounds

    return ResidualBounds(
        "burgers",
        (0, 0),
        (0, 0),
        (1.0, 1.0),
        {"residual": {"complete": True}, "boundary": {"complete": True}},
        burgers=BurgersStability(0.01, 1.0, edges, e, r, a, True, True),
    )

def test_recurrence_exact_cases():
    assert solution_bound(burgers_certificate(e=(2.0, 2.0))).solution_error_upper == 2
    assert (
        solution_bound(
            burgers_certificate(e=(2.0, 2.0), r=((3.0, 3.0),))
        ).solution_error_upper
        == 5
    )
    c = burgers_certificate(e=(2.0, 2.0), r=((3.0, 3.0),), a=((4.0, 4.0),))
    assert solution_bound(c).solution_error_upper == pytest.approx(
        2 * math.exp(4) + 3 * math.expm1(4) / 4
    )
    c = burgers_certificate(
        e=(1.0, 1.0),
        r=((2.0, 2.0), (3.0, 3.0)),
        a=((0.0, 0.0), (2.0, 2.0)),
        edges=(0.0, 0.25, 1.0),
    )
    assert solution_bound(c).solution_error_upper == pytest.approx(
        1.5 * math.exp(1.5) + 1.5 * math.expm1(1.5)
    )

@pytest.mark.parametrize("constant", [-3.0, 7.0])
def test_constant_potential_is_exact_without_zero_mean(constant):
    potential = torch.full((2, 128), constant, dtype=torch.float64)
    bounds, info = periodic_energy_bounds(potential, potential, 0.01, modes=8)
    torch.testing.assert_close(
        bounds, torch.full_like(bounds, constant), atol=1e-12, rtol=1e-12
    )
    assert info["all_omitted_modes_bounded"]
    assert not info["zero_mean_assumed"]
