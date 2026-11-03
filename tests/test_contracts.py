import math

import pytest
import torch

from lean_no.adapters.sfno import SFNOInput
from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.function_space import InputBounds, coord
from lean_no.methods import Method
from lean_no.quantities.solution import solution_bound
from lean_no.refine import audit_cover
from tests.helpers import averaging_model

def test_quadtree_audit_detects_gaps_and_overlaps():
    ids = torch.cartesian_prod(torch.arange(2), torch.arange(4))
    children = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    leaves = [dict(level=0, indices=ids[1:]), dict(level=1, indices=children)]
    assert audit_cover(leaves, 2, 4)["leaf_count"] == 11
    with pytest.raises(ArithmeticError, match="overlap or gap"):
        audit_cover([leaves[0], dict(level=1, indices=children[:-1])], 2, 4)
    with pytest.raises(ArithmeticError, match="overlap or gap"):
        audit_cover(leaves + [dict(level=1, indices=children[:1])], 2, 4)

def test_input_contract():
    coefficients = torch.zeros(1, 1, 1, dtype=torch.complex128)
    with pytest.raises(ValueError, match="full-input Lipschitz"):
        SFNOInput(coefficients, ("a",), 0.001)
    for budget in (-1, math.inf, math.nan, torch.ones(2)):
        with pytest.raises(ValueError, match="Lipschitz"):
            SFNOInput(coefficients, ("a",), 0.001, budget)
    inputs = SFNOInput(coefficients, ("a",), 0.001, 0.004)
    inputs.admit(0.001, 0.004)
    for deviation, budget in ((0.002, 0.004), (0.001, 0.005), (math.nan, 0), (0, -1)):
        with pytest.raises(ValueError, match="budgets exceed"):
            inputs.admit(deviation, budget)
    lower, upper = inputs.envelope(torch.zeros(3, 2, dtype=torch.float64))
    torch.testing.assert_close(lower, torch.full_like(lower, -0.001))
    torch.testing.assert_close(upper, torch.full_like(upper, 0.001))

@pytest.mark.parametrize("lower,upper", [(2.0, 1.0), (float("nan"), 1.0)])
def test_invalid_input_enclosures_are_rejected(lower, upper):
    from lean_no.enclosures.grid import GridBounds

    with pytest.raises(ValueError):
        GridBounds(torch.tensor([[[lower]]]), torch.tensor([[[upper]]]))

def test_nonperiodic_query_is_not_silently_wrapped():
    field = certify(
        averaging_model(),
        FourierGrid((16,), (2.3,)),
        InputBounds.exact((coord(),), [2.3], [1.0]),
        method=Method("ibp", "gaussian"),
    )
    with pytest.raises(ValueError):
        field.query(torch.tensor([[-0.01], [2.31]], dtype=torch.float64))

def test_resonance_and_incomplete_stability_data_are_rejected():
    from lean_no.quantities.norms import BurgersStability, ResidualBounds

    item = ResidualBounds(
        "helmholtz",
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0,),
        {"residual": {"complete": True}, "boundary": {"complete": True}},
        kappa=2 * math.pi,
        boundary_lower=(0.0, 0.0),
        boundary_upper=(0.0, 0.0),
        residual_l2=1.0,
    )
    with pytest.raises(ValueError, match="resonant"):
        solution_bound(item, norm="l2")
    with pytest.raises(ValueError, match="per slab"):
        BurgersStability(
            0.01,
            1.0,
            (0.0, 0.5, 1.0),
            (0.0, 0.0),
            ((0.0, 1.0),),
            ((0.0, 1.0),),
            True,
            True,
        )

def test_unknown_methods_fail_before_model_evaluation():
    with pytest.raises(ValueError, match="unsupported"):
        certify(None, None, None, method=Method("crown", "taylor_quadrature"))

@pytest.mark.parametrize(
    "radius", [0.03, lambda x: 0.03 + 0.01 * torch.cos(2 * math.pi * x)]
)
def test_prepared_input_band_preserves_uncertainty_and_curvature(radius):
    from lean_no.inputs import InputSet
    from lean_no.methods import Method

    source = lambda x: torch.sin(2 * math.pi * x)
    prepared = InputSet(source, radius, [7.0], [40.0]).prepare(
        averaging_model(), FourierGrid((16,)), Method("ibp", "gaussian")
    )
    points = torch.linspace(0, 1, 33, dtype=torch.float64)
    spread = radius(points) if callable(radius) else radius
    expected = source(points)
    torch.testing.assert_close(prepared.lower(points)[0], expected - spread)
    torch.testing.assert_close(prepared.upper(points)[0], expected + spread)
    torch.testing.assert_close(prepared.lipschitz, torch.tensor([7.0]))
    torch.testing.assert_close(prepared.curvature, torch.tensor([40.0]))

@pytest.mark.parametrize(
    "integration,propagation",
    [("gaussian", "ibp"), ("taylor_quadrature", "ibp"), ("gaussian", "crown")],
)
def test_callable_and_symbolic_inputs_give_the_same_bounds(integration, propagation):
    from lean_no.function_space.analytic import describe

    source = lambda x: 0.2 + torch.sin(2 * math.pi * x)
    model, grid = averaging_model(), FourierGrid((16,))
    points = torch.tensor([[0.13], [0.42], [0.78]], dtype=torch.float64)
    fields = [
        certify(model, grid, inputs, method=Method(propagation, integration))
        for inputs in (source, describe(source))
    ]
    actual, expected = [field.query(points) for field in fields]
    torch.testing.assert_close(actual.lower, expected.lower, rtol=0, atol=0)
    torch.testing.assert_close(actual.upper, expected.upper, rtol=0, atol=0)
