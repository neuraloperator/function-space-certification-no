import math

import numpy as np
import pytest
import torch

from lean_no.adapters.fno import cast_module
from lean_no.domain import periodic_grid
from lean_no.function_space.analytic import describe
from lean_no.propagation.derivative_cells import DerivativeField
from lean_no.reference import cole_hopf_reference
from tests.helpers import burgers_identity
from training.burgers_solution import input_jet, period, solution_jet
from training.helmholtz_generate import (
    frequency_gap,
    resonant_mode,
    solve_helmholtz,
    wave_number,
)

def test_training_jet_and_certified_jet_against_exact_polynomial():
    model = cast_module(burgers_identity(), torch.float64)
    x = torch.arange(16, dtype=torch.float64) / 16
    initial = torch.sin(2 * math.pi * x)
    values, trace = solution_jet(model, initial[None], 64, 64, return_trace=True)
    inputs = input_jet(initial[None], 64, 64)
    a, t = inputs[0][:, :1], inputs[0][:, 1:2]
    ax, axx = inputs[1][:, :1], inputs[3][:, :1]
    expected = (a + t, ax, torch.ones_like(a), axx)
    for actual, truth in zip(values, expected):
        torch.testing.assert_close(actual, truth, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(trace, a[..., :1], atol=1e-11, rtol=1e-11)
    field = DerivativeField.from_profile(
        model, describe(lambda x: torch.sin(2 * math.pi * x))
    )
    field.certify_coefficients(order=16, progress=None)
    centers = torch.tensor([[0.21, 0], [0.43, 0.27], [0.78, 0.73]], dtype=torch.float64)
    jet = field.enclose(centers, torch.zeros_like(centers))
    a = torch.sin(2 * math.pi * centers[:, 0])
    ax = 2 * math.pi * torch.cos(2 * math.pi * centers[:, 0])
    axx = -((2 * math.pi) ** 2) * a
    t = centers[:, 1] * period
    expected = (
        a + t,
        ax,
        torch.ones_like(t),
        axx,
        torch.zeros_like(t),
        torch.zeros_like(t),
    )
    for actual, truth in zip(jet, expected):
        torch.testing.assert_close(actual.c[:, 0], truth, atol=1e-11, rtol=1e-11)

def test_resonant_mode() -> None:
    count = 64
    grid = periodic_grid(count, dtype=torch.float64)
    forcing = torch.cos(2 * math.pi * resonant_mode * grid).unsqueeze(0)
    solution = solve_helmholtz(forcing, wave_number())
    expected = forcing / frequency_gap
    torch.testing.assert_close(solution, expected, atol=2e-10, rtol=2e-10)

def test_poisson_data_satisfies_the_eigenmode_equation():
    from training.poisson_2d_generate import evaluate_fields

    coefficients = torch.zeros(1, 2, 2, dtype=torch.float64)
    coefficients[0, 1, 0] = 0.7
    forcing, solution = evaluate_fields(coefficients, 16)
    x = torch.arange(16, dtype=torch.float64) / 16
    exact = 0.7 * torch.sin(2 * math.pi * x)[:, None] * torch.sin(math.pi * x)[None, :]
    torch.testing.assert_close(forcing[0, 0], exact)
    torch.testing.assert_close(solution[0, 0], exact / (5 * math.pi**2))

def test_training_residual_parameter_gradients_match_manufactured_formula():
    model = cast_module(burgers_identity(), torch.float64)
    a, b = 0.7, -0.3
    with torch.no_grad():
        model.projection.fcs[0].weight[0, :, 0] = torch.tensor([a, b])
    initial = torch.sin(2 * math.pi * torch.arange(16, dtype=torch.float64) / 16)[None]
    u, ux, ut, uxx = solution_jet(model, initial, 32, 32)
    loss = (ut + u * ux - 0.01 * uxx).square().mean()
    loss.backward()
    jets = input_jet(initial, 32, 32)
    phi, t = jets[0][:, :1], jets[0][:, 1:2]
    phix, phixx = jets[1][:, :1], jets[3][:, :1]
    parameters = torch.tensor([a, b], dtype=torch.float64, requires_grad=True)
    alpha, beta = parameters
    expected = (
        (beta + (alpha * phi + beta * t) * alpha * phix - 0.01 * alpha * phixx)
        .square()
        .mean()
    )
    expected.backward()
    torch.testing.assert_close(
        model.projection.fcs[0].weight.grad[0, :, 0],
        parameters.grad,
        rtol=2e-7,
        atol=1e-10,
    )

def test_ks_solver_constant_solution_and_time_convergence():
    pytest.importorskip("ks_training")
    from ks_training.model import solve

    torch.set_num_threads(4)
    constant = torch.full((2, 128), 0.7, dtype=torch.float64)
    torch.testing.assert_close(solve(constant), constant[:, None].expand(2, 65, 128))
    x = torch.arange(128, dtype=torch.float64) * 2 * math.pi / 128
    initial = (torch.sin(x) + 0.5 * torch.cos(2 * x))[None]
    coarse = solve(initial, substeps=2)
    fine = solve(initial, substeps=4)
    finer = solve(initial, substeps=8)
    assert (fine - finer).abs().max() < 1e-9
    assert (coarse - finer).abs().max() > 5 * (fine - finer).abs().max()

def test_cole_hopf_reference_initial_mean_and_pde():
    x = np.arange(16) / 16
    initial = describe(
        lambda x: (
            0.3 + 0.2 * torch.sin(2 * math.pi * x) + 0.1 * torch.cos(16 * math.pi * x)
        )
    )
    expected = 0.3 + 0.2 * np.sin(2 * np.pi * x) + 0.1 * np.cos(16 * np.pi * x)
    times = np.array([0, 0.2 - 1e-5, 0.2, 0.2 + 1e-5, 1])
    values = cole_hopf_reference(initial, times, 512)
    assert np.max(np.abs(values[0, ::32] - expected)) < 1e-12
    assert np.max(np.abs(values.mean(axis=1) - 0.3)) < 1e-12
    k = 2 * np.pi * np.fft.rfftfreq(512, d=1 / 512)
    spectrum = np.fft.rfft(values[2])
    vx = np.fft.irfft(1j * k * spectrum)
    vxx = np.fft.irfft(-(k**2) * spectrum)
    residual = (values[3] - values[1]) / 2e-5 + values[2] * vx - 0.01 * vxx
    assert np.max(np.abs(residual)) < 1e-8
    finer = cole_hopf_reference(initial, times, 1024)
    assert np.max(np.abs(finer[:, ::2] - values)) < 1e-12
