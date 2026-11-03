import math

import pytest
import torch

from lean_no import FourierGrid, InputSet, Method, certify
from lean_no.adapters.sfno import SFNOInput, SFNOModel
from lean_no.domain import DiscreteGrid
from lean_no.enclosures.grid import GridBounds
from lean_no.evaluation import query_bounds
from lean_no.graph import Graph, Operator
from lean_no.spectral.spherical import sfno_grid
from tests.helpers import averaging_model

def identity_model():
    model = averaging_model()
    with torch.no_grad():
        model.fno_blocks.convs[0].weight.tensor.zero_()
        model.fno_blocks.fno_skips[0].conv.weight.fill_(1)
    return model

@pytest.mark.parametrize("epsilon", [0.0, 0.05])
def test_pointwise_uncertainty_excludes_cell_variation(epsilon):
    field = certify(
        identity_model(),
        FourierGrid((4,), (2.0,)),
        InputSet(lambda x: x, epsilon=epsilon),
        method=Method("ibp", None, "linearized"),
    )
    assert field.pointwise_width.shape == field.cell_range_width.shape == (1,)
    assert float(field.pointwise_width[0]) == pytest.approx(2 * epsilon)
    assert float(field.cell_range_width[0]) == pytest.approx(0.5 + 2 * epsilon)
    assert field.node_width is None and field.query_width is None
    points = torch.tensor([[0.2], [0.8], [1.3]], dtype=torch.float64)
    bounds = field.query(points)
    torch.testing.assert_close(bounds.lower, points - epsilon)
    torch.testing.assert_close(bounds.upper, points + epsilon)
    torch.testing.assert_close(bounds.width, torch.full_like(points, 2 * epsilon))

def test_node_width_is_independent_of_variation_between_nodes():
    values = torch.arange(4, dtype=torch.float64)[None, None]
    field = certify(
        identity_model(),
        DiscreteGrid((4,)),
        GridBounds(values - 0.05, values + 0.05),
        method=Method("ibp", None),
    )
    torch.testing.assert_close(
        field.node_width, torch.tensor([0.1], dtype=torch.float64)
    )
    assert field.pointwise_width is field.cell_range_width is field.query_width is None

@pytest.mark.parametrize(
    "propagation,integration",
    [("ibp", "gaussian"), ("ibp", "taylor_quadrature"), ("crown", "rectangle")],
)
def test_query_width_covers_only_requested_points(propagation, integration):
    points = torch.tensor([[0.13], [0.8], [1.3]], dtype=torch.float64)
    field = certify(
        identity_model(),
        FourierGrid((8,), (2.0,)),
        lambda x: torch.sin(math.pi * x),
        method=Method(propagation, integration),
        query=points,
    )
    bounds = field.query(points)
    torch.testing.assert_close(field.query_width, bounds.width.amax(0))
    assert field.pointwise_width is field.cell_range_width is field.node_width is None
    lower, upper = query_bounds(field, points, batch_size=1)
    torch.testing.assert_close(lower, bounds.lower)
    torch.testing.assert_close(upper, bounds.upper)

def test_spherical_certificate_has_pointwise_width():
    grid = sfno_grid(8, dtype=torch.float64, rule="clenshaw-curtis")
    spectral = Operator(
        "average",
        "spectral",
        (0,),
        dict(weight=torch.ones(1, 1, 1, dtype=torch.complex128), degrees=1, orders=1),
    )
    model = SFNOModel(Graph((spectral,), {"basis": "spherical"}), grid, grid, ("a",))
    inputs = SFNOInput(torch.zeros(1, 1, 1, dtype=torch.complex128), ("a",), 0.1, 0.0)
    field = certify(model, grid, inputs, method=Method("ibp", "clenshaw_curtis"))
    torch.testing.assert_close(
        field.pointwise_width, torch.tensor([0.2], dtype=torch.float64)
    )
    assert field.cell_range_width is field.node_width is field.query_width is None

def test_derivative_certificate_does_not_claim_an_uncomputed_width():
    field = certify(
        identity_model(),
        FourierGrid((8,)),
        lambda x: torch.sin(2 * math.pi * x),
        derivatives=1,
    )
    assert (
        field.pointwise_width
        is field.cell_range_width
        is field.node_width
        is field.query_width
        is None
    )
    assert torch.isfinite(
        field.query(torch.tensor([[0.2]], dtype=torch.float64)).width
    ).all()

def test_certify_rejects_ambiguous_or_positional_options():
    with pytest.raises(TypeError, match="propagation"):
        certify(None, None, None, propagation="ibp")
    with pytest.raises(TypeError, match="epsilon"):
        certify(None, None, None, epsilon=0.01)
    with pytest.raises(TypeError):
        certify(None, None, None, 2)
    with pytest.raises(TypeError, match="Method"):
        certify(None, None, None, method="ibp")
