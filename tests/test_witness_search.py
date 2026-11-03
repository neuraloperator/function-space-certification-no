from dataclasses import replace

import pytest
import torch

from lean_no.adapters.fno import build_graph, native
from lean_no.function_space.analytic import describe
from scripts._shared.attacks import native_inputs, run_mc, sample_ball, sparse_samples
from scripts._shared.inputs import WitnessSearch, basis_values, witness_input
from tests.helpers import fno

def case():
    return WitnessSearch(1.0, 0.001, 16, 8, mc_batch_size=64)

@pytest.mark.parametrize("dimension", [1, 2])
def test_fourier_candidates_are_admissible_and_analytic(dimension):
    setup = replace(
        case(),
        spatial_dimension=dimension,
        fourier_modes=4 if dimension == 2 else 16,
        decay_mode=None,
    )
    axis = torch.linspace(0, 1, 19, dtype=torch.float64)
    grid = axis if dimension == 1 else torch.meshgrid(axis, axis, indexing="ij")
    basis = basis_values(setup, grid)
    assert len(basis) == (33 if dimension == 1 else 81)
    ref = axis
    generator = torch.Generator().manual_seed(0)
    vertices = sparse_samples(len(basis), ref)
    coefficients = torch.cat(
        (vertices, sample_ball(4096 - len(vertices), len(basis), ref, generator))
    )
    assert (coefficients.abs().sum(-1) <= 1 + 1e-14).all()
    assert (coefficients @ basis.flatten(1)).abs().max() <= setup.eps + 1e-14
    for c in (coefficients[0], coefficients[-1]):
        function = witness_input(setup, c)
        nominal = witness_input(setup)
        coordinates = (axis,) if dimension == 1 else grid
        actual = function(*coordinates) - nominal(*coordinates)
        torch.testing.assert_close(
            actual, (c @ basis.flatten(1)).reshape(actual.shape), atol=1e-14, rtol=1e-12
        )
        assert describe(function) is not None

@pytest.mark.parametrize("space_time", [False, True])
def test_search_winner_reproduces_native_width(space_time):
    setup = replace(
        case(), time_samples=5 if space_time else 0, time_period=1.25, decay_mode=None
    )
    model = fno(
        2 if space_time else 1,
        1,
        torch.nn.Tanh(),
        inputs=2 if space_time else 1,
        outputs=1,
    )
    result = run_mc(setup, model)
    repeat = run_mc(setup, model)
    assert result["global_width"] == repeat["global_width"]
    assert result["samples"] == 4096
    center, basis, points = native_inputs(setup, model)
    candidates = center[None] + (result["global_coef"] @ basis.flatten(1)).reshape(
        2, *center.shape
    )
    if space_time:
        torch.testing.assert_close(candidates[:, 1], center[None, 1].expand(2, -1, -1))
        perturbations = candidates[:, 0] - center[None, 0]
        torch.testing.assert_close(
            perturbations, perturbations[:, :, :1].expand_as(perturbations)
        )
    outputs = native(build_graph(model), candidates)[:, 0].flatten(1)
    index = result["global_index"]
    assert float((outputs[0, index] - outputs[1, index]).abs()) == pytest.approx(
        result["global_width"], abs=1e-13
    )
    torch.testing.assert_close(result["point"], points[index])
