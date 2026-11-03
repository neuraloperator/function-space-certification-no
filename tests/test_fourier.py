import itertools
import math

import numpy as np
import pytest
import torch
from neuralop.layers.spectral_convolution import SpectralConv

from lean_no.adapters.fno import build_graph
from lean_no.graph import Operator
from lean_no.integration import integrate
from lean_no.integration.kernel_norm import LinearizedKernel, coefficients_at
from lean_no.kernel import kernel
from lean_no.propagation.bilinear import BilinearBounds, cell_spectrum, relu
from lean_no.propagation.perturbation import KernelState, full_symbol
from tests.helpers import fno

def interpolate(values, x, y):
    weights = values.new_tensor(((1 - x) * (1 - y), (1 - x) * y, x * (1 - y), x * y))
    return torch.einsum("q,qcxy->cxy", weights, values)

def test_nonperiodic_moments():
    count = 32
    axis = torch.arange(count, dtype=torch.float64) / count
    corners = []
    for x in (0, 1):
        for y in (0, 1):
            xx, yy = torch.meshgrid(axis + x / count, axis + y / count, indexing="ij")
            corners.append((xx + 2 * yy + 3 * xx * yy)[None])
    modes = (
        torch.arange(-3, 3, dtype=torch.float64),
        torch.arange(3, dtype=torch.float64),
    )
    result = cell_spectrum(torch.stack(corners), modes)[0]
    moments = []
    for frequency in modes:
        zero = (frequency == 0).to(torch.complex128)
        linear = torch.where(
            frequency == 0,
            torch.full_like(frequency, 0.5).to(torch.complex128),
            1j
            / (
                2
                * math.pi
                * torch.where(frequency == 0, torch.ones_like(frequency), frequency)
            ),
        )
        moments.append((zero, linear))
    x0, x1 = moments[0]
    y0, y1 = moments[1]
    expected = x1[:, None] * y0 + 2 * x0[:, None] * y1 + 3 * x1[:, None] * y1
    torch.testing.assert_close(result, expected, atol=2e-13, rtol=2e-12)

def test_relu_crossings():
    lower = torch.randn(4, 3, 5, 5, dtype=torch.float64)
    upper = lower + torch.rand_like(lower)
    source = BilinearBounds(lower, upper)
    output = relu(source)
    for x, y in ((0.0, 0.0), (0.23, 0.77), (0.5, 0.5), (1.0, 1.0)):
        low = interpolate(source.lower, x, y)
        high = interpolate(source.upper, x, y)
        for fraction in (0.0, 0.29, 1.0):
            value = torch.relu(low + fraction * (high - low))
            assert torch.all(interpolate(output.lower, x, y) <= value + 1e-14)
            assert torch.all(interpolate(output.upper, x, y) >= value - 1e-14)

@pytest.mark.parametrize("unequal", [False, True])
def test_graph_kernel_matches_independent_dense_three_block_operator(unequal):
    torch.manual_seed(92)
    graph = build_graph(fno(1, 3, torch.nn.Tanh(), inputs=2, outputs=2))
    if unequal:
        from dataclasses import replace

        ops = []
        counts = iter((2, 1, 3))
        for op in graph.ops:
            if op.kind == "spectral":
                count = next(counts)
                weight = (
                    torch.randn(
                        op.out_channels, op.in_channels, count, dtype=torch.complex128
                    )
                    * 0.2
                )
                modes = (torch.arange(count, dtype=torch.float64),)
                op = replace(op, data={**op.data, "weight": weight, "modes": modes})
            ops.append(op)
        graph = replace(graph, ops=tuple(ops))
    n = 8
    nodes, weights = np.polynomial.legendre.leggauss(9)
    points = ((torch.arange(n)[:, None] + (torch.tensor(nodes) + 1) / 2) / n).flatten()
    weights = torch.tensor(weights).repeat(n) / (2 * n)
    cell_ids = torch.arange(n).repeat_interleave(len(nodes))
    count = len(points)
    full = torch.eye(2 * count, dtype=torch.float64)
    dense = {0: full}
    kernels = KernelState((n,), 2)
    states = {
        0: (
            torch.eye(2, dtype=torch.float64)[:, :, None].expand(2, 2, n),
            torch.zeros(2, 0, n, dtype=torch.float64),
        )
    }

    def pad(rows):
        total = sum(kernels.channels)
        return torch.cat((rows, rows.new_zeros(len(rows), total - rows.shape[1], n)), 1)

    for index, op in enumerate(graph.ops, 1):
        full = dense[op.parents[0]]
        local, rows = states[op.parents[0]]
        rows = pad(rows)
        if op.kind == "activation":
            slope = (
                torch.randn(full.shape[0] // count, n, dtype=torch.float64) * 0.3 + 0.5
            )
            full = slope[:, cell_ids].flatten()[:, None] * full
            local, rows = local * slope[:, None], rows * slope[:, None]
        elif op.kind == "affine":
            full = torch.kron(op.weight, torch.eye(count, dtype=torch.float64)) @ full
            local = torch.einsum("oi,ijx->ojx", op.weight, local)
            rows = torch.einsum("oi,ijx->ojx", op.weight, rows)
        elif op.kind == "add":
            full = sum(dense[parent] for parent in op.parents)
            local = sum(states[parent][0] for parent in op.parents)
            rows = sum(pad(states[parent][1]) for parent in op.parents)
        else:
            symbol, frequencies = full_symbol(op.weight, op.data["modes"])
            phase = torch.exp(
                2j
                * math.pi
                * (points[:, None] - points[None])
                * frequencies[:, 0].double()[:, None, None]
            )
            kernel = torch.einsum("oik,kpq->opiq", symbol, phase).real
            matrix = kernel * weights[None, None, None, :]
            full = matrix.flatten(0, 1).flatten(1) @ full
            kernels.append(op, local, rows)
            local = local.new_zeros(op.out_channels, 2, n)
            rows = rows.new_zeros(op.out_channels, sum(kernels.channels), n)
            ids = torch.arange(op.out_channels)
            rows[ids, ids + rows.shape[1] - op.out_channels] = 1
        dense[index], states[index] = full, (local, rows)
    for cell, channel in ((0, 0), (2, 1)):
        target = cell * len(nodes) + len(nodes) // 2
        coefficients = coefficients_at(
            kernels, points[target].reshape(1, 1), rows[channel, :, cell][None]
        )[0]
        reconstructed = torch.zeros(2, count, dtype=torch.float64)
        offset = 0
        for basis in kernels.bases:
            channels, modes = len(basis.amplitude), len(basis.frequencies)
            block = coefficients[offset : offset + channels * modes].reshape(
                channels, modes
            )
            phase = torch.exp(
                -2j * math.pi * points[:, None] * basis.frequencies[:, 0].double()
            )
            reconstructed += (
                torch.einsum(
                    "ck,ciq,qk->iq",
                    block,
                    basis.amplitude[:, :, cell_ids].to(block),
                    phase,
                ).real
                * weights
            )
            offset += channels * modes
        reconstructed[:, target] += local[channel, :, cell]
        torch.testing.assert_close(
            reconstructed.flatten(),
            full[channel * count + target],
            atol=1e-13,
            rtol=1e-12,
        )

@pytest.mark.parametrize("dimension", [1, 2])
def test_multichannel_kernel_against_native_spectral_convolution(dimension):
    # An asymmetric 2->3 map detects input/output transposes and phase-sign errors.
    conv = SpectralConv(
        2, 3, n_modes=(3,) * dimension, bias=False, factorization=None
    ).eval()
    # Native NeuralOperator uses fftshift twice in 2D, so use its even-grid convention.
    n = 9 if dimension == 1 else 8
    x = torch.randn(1, 2, *((n,) * dimension))
    axes = [torch.arange(n, dtype=torch.float32) / n for _ in range(dimension)]
    if dimension == 1:
        k = kernel(conv, axes[0], 1.0)
        ids = torch.arange(n)
        matrix = k[:, :, ((ids[:, None] - ids[None, :]) % n)] / n
        expected = torch.einsum("oipq,biq->bop", matrix, x)
    else:
        coordinates = tuple(torch.meshgrid(*axes, indexing="ij"))
        k = kernel(conv, coordinates, (1.0, 1.0))
        expected = torch.zeros(1, 3, n, n)
        for i, j in itertools.product(range(n), repeat=2):
            expected += (
                torch.einsum(
                    "oi,binm->bonm", k[:, :, i, j], torch.roll(x, (i, j), (-2, -1))
                )
                / n**2
            )
    torch.testing.assert_close(conv(x), expected, atol=2e-6, rtol=2e-6)

def test_continuous_kernel_phase_and_physical_derivative():
    from lean_no.kernel import kernel_gain, kernel_partial

    conv = SpectralConv(1, 1, n_modes=(2,), bias=False, factorization=None).eval()
    with torch.no_grad():
        conv.weight.tensor.zero_()
        conv.weight.tensor[..., 1] = 3 + 4j
    length = 2.3
    x = torch.linspace(0, length, 41, dtype=torch.float64)
    w = 2 * math.pi / length
    torch.testing.assert_close(
        kernel(conv, x, length)[0, 0], 6 * torch.cos(w * x) - 8 * torch.sin(w * x)
    )
    torch.testing.assert_close(
        kernel_partial(conv, x, length, 0)[0, 0],
        -6 * w * torch.sin(w * x) - 8 * w * torch.cos(w * x),
    )
    assert float(kernel_gain(conv, length, (2,))) == pytest.approx(10 * w * w)

def kernel_source(constant=False, jump=False, n=32):
    modes = (
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    weight = torch.zeros(1, 1, 1, 2, dtype=torch.complex128)
    weight[0, 0, 0, 0 if constant else 1] = 2 if constant else 0.5
    op = Operator(
        "spectral",
        "spectral",
        (0,),
        dict(weight=weight, bias=torch.zeros(1, dtype=torch.float64), modes=modes),
    )
    local = torch.ones(1, 1, n, n, dtype=torch.float64)
    if jump:
        local[:, :, n // 2 :] = 0
    kernels = KernelState((n, n), 1)
    kernels.discrete = True
    kernels.append(op, local, torch.zeros(1, 0, n, n, dtype=torch.float64))
    return LinearizedKernel(
        kernels,
        torch.ones(1, 1, n, n, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )

def test_nonsmooth_absolute_cosine_norm_uniform_over_output_cells():
    value, stats = integrate(
        kernel_source(),
        method="gaussian",
        options=dict(panels=16, output_panels=16, order=8),
    )
    assert (value >= 2 / math.pi).all()
    assert (value < 1).all()  # Strictly better than its exact L2 norm.
    assert stats["sign_certified"] > 0 and stats["cauchy_used"] > 0
    # The same bound covers every location inside each output cell.
    s = (torch.arange(32768, dtype=torch.float64) + 0.5) / 32768
    for x in (0.0, 0.123, 0.4999, 0.9999):
        numerical = torch.cos(2 * math.pi * (x - s)).abs().mean()
        assert numerical <= value[0, 0, min(int(x * 32), 31)]

def test_source_jumps_are_inherited_uncertainty_not_analytic():
    coarse, record = integrate(
        kernel_source(constant=True, jump=True),
        method="gaussian",
        options=dict(panels=1, output_panels=2, order=8),
    )
    fine, _ = integrate(
        kernel_source(constant=True, jump=True),
        method="gaussian",
        options=dict(panels=2, output_panels=2, order=8),
    )
    assert record["amplitude_uncertainty"] > 0
    assert record["cauchy_used"] == 0
    assert (coarse >= 1).all()
    torch.testing.assert_close(fine, torch.ones_like(fine), rtol=0, atol=1e-13)

@pytest.mark.parametrize("shared", [False, True])
def test_basis_gram_against_independent_tensor_quadrature(shared):
    from lean_no.propagation.perturbation import FourierBasis, basis_gram

    torch.manual_seed(19)
    shape = (6, 8)
    left = FourierBasis(
        torch.randn(2, 1, *shape, dtype=torch.float64),
        torch.tensor([[-1, -1], [0, 0], [1, 1]]),
    )
    right = FourierBasis(
        torch.randn(3, 1, *shape, dtype=torch.float64),
        torch.tensor([[-1, 1], [0, 0], [2, -1], [1, 2]]),
    )
    nodes, weights = np.polynomial.legendre.leggauss(9)
    axes = [
        ((torch.arange(n)[:, None] + (torch.tensor(nodes) + 1) / 2) / n).flatten()
        for n in shape
    ]
    quadrature = [torch.tensor(weights).repeat(n) / (2 * n) for n in shape]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
    ids = [(axis * n).long() for axis, n in zip(axes, shape)]

    def evaluate(basis):
        amplitude = basis.amplitude[:, 0, ids[0][:, None], ids[1][None]]
        phase = torch.exp(
            -2j
            * math.pi
            * torch.einsum("xyd,kd->kxy", coordinates, basis.frequencies.double())
        )
        values = amplitude[:, None] * phase
        if shared:
            return (values * quadrature[1]).sum(-1).flatten(0, 1)
        return values.flatten(0, 1).flatten(1)

    a, b = evaluate(left), evaluate(right)
    weight = (
        quadrature[0] if shared else (quadrature[0][:, None] * quadrature[1]).flatten()
    )
    expected = (a * weight) @ b.conj().T
    torch.testing.assert_close(
        basis_gram(left, right, shared)[0], expected, atol=2e-14, rtol=2e-12
    )
