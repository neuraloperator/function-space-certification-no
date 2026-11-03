import math

import pytest
import torch

from lean_no.enclosures.cauchy_jet import CauchyProfileJetCache
from lean_no.enclosures.taylor import polydisk
from lean_no.graph import affine, fourier, gelu
from lean_no.propagation.derivative_cells import enclose_cells, enclose_jet
from lean_no.propagation.rollout import (
    initial_state_handoff,
    output_bounds,
    rollout,
    values,
)
from lean_no.spectral.strip import StripGauss

def test_custom_rollout_geometry():
    ref = torch.zeros(1, dtype=torch.float64)
    state = {"fno_blocks.convs.bias": ref.new_zeros(1, 1, 1, 1)}
    for prefix, inputs in (
        ("lifting.fcs.0", 3),
        ("lifting.fcs.1", 1),
        ("projection.fcs.0", 1),
        ("projection.fcs.1", 1),
        ("fno_blocks.fno_skips.0", 1),
        ("fno_blocks.mlp.0.fcs.0", 1),
        ("fno_blocks.mlp.0.fcs.1", 1),
    ):
        state[prefix + ".weight"] = ref.new_zeros(1, inputs)
        state[prefix + ".bias"] = ref.clone()
    state["fno_blocks.convs.weight.0.tensor"] = ref.new_zeros(1, 1, 2, 3, 2)
    state["fno_blocks.mlp_skips.0.weight"] = ref.new_ones(1)
    state["_hard_initial_condition"] = True
    for index in (0, 1):
        state[f"fno_blocks.norm.{index}.weight"] = ref.new_ones(1)
        state[f"fno_blocks.norm.{index}.bias"] = ref.clone()
    rule = StripGauss(order=4, cells=((2, 4),), progress=lambda *args, **kwargs: None)
    field, _, _ = rollout(
        state,
        lambda x: 2.0,
        rule,
        lengths=(2.0, 3.0),
        split=0.6,
        native_shape=(8, 10),
        orders=((0, 0), (1, 0), (0, 1)),
    )
    points = ref.new_tensor([[0.0, 0.4], [0.6, 2.0], [1.2, 3.0]])
    lower, upper = output_bounds(field, points)
    expected = ref.new_tensor([[2.0], [1.0], [0.0]])
    torch.testing.assert_close(lower, expected)
    torch.testing.assert_close(upper, expected)
    t, x = ref.new_tensor([0.3]), ref.new_tensor([0.4])
    for nominal, actual in (
        enclose_jet(field, t, x, paired=True),
        enclose_cells(field, t, x),
    ):
        for jet in (nominal, actual):
            torch.testing.assert_close(
                jet[1].lower.flatten(), ref.new_tensor([-1 / 0.6])
            )
    centers, radii = ref.new_tensor([[0.3, 0.4]]), ref.new_tensor([[0.01, 0.02]])
    value, gradient, remainder = polydisk(field, centers, radii, True)
    torch.testing.assert_close(value, ref.new_tensor([[1.0]]).to(value.dtype))
    torch.testing.assert_close(
        gradient[..., 0], ref.new_tensor([[-2 / 0.6]]).to(gradient.dtype)
    )
    assert float(remainder.max()) < 1e-12
    terminal = initial_state_handoff(field, retain_derivatives=True)
    torch.testing.assert_close(
        values(terminal, t[None], x[None]), ref.new_zeros(1, 1, 1, 1)
    )
    assert terminal.metadata["lengths"] == (2.0, 3.0)
    with pytest.raises(ValueError, match="active physical slab"):
        output_bounds(field, [[1.21, 0.0]])

def test_profile_orders_and_periods():
    ref = torch.zeros(1, dtype=torch.float64)
    orders = ((0, 0), (0, 1), (0, 2))
    source = fourier(
        torch.tensor([[[0.0, -0.5j]]], dtype=torch.complex128),
        (ref, ref.new_tensor([0.0, 1.0])),
        ref,
        metadata=dict(lengths=(2.0, 3.0), split=0.6, orders=orders),
    )
    profile = initial_state_handoff(source, retain_derivatives=True)
    cache = CauchyProfileJetCache(capacity=8, degree=8)
    x = ref.new_tensor([0.13, 0.37])
    for length in (3.0, 5.0, 3.0):
        _, jets = enclose_cells(
            profile,
            x * 0,
            x,
            profile_cache=cache,
            orders=orders,
            periods=(2.0, length),
        )
        for j, jet in enumerate(jets):
            expected = (2 * math.pi / length) ** j * torch.sin(
                2 * math.pi * x + j * math.pi / 2
            )
            assert bool((jet.lower[:, 0] <= expected + 1e-12).all())
            assert bool((jet.upper[:, 0] >= expected - 1e-12).all())
    assert cache.hits == 2
    assert cache.misses == 4

def test_default_units_and_value_order():
    ref = torch.zeros(1, dtype=torch.float64)
    source = fourier(
        torch.tensor([[[0.0, -0.5j]]], dtype=torch.complex128),
        (ref, ref.new_tensor([0.0, 1.0])),
        ref,
    )
    source = gelu(affine(source, ref.new_tensor([[1.3]]), ref.new_tensor([0.2])))
    t, x = ref.new_tensor([0.2]), ref.new_tensor([0.3])
    _, jets = enclose_jet(source, t, x, 0.01, 0.02)
    assert len(jets) == 1
    for shift in (-0.02, 0.0, 0.02):
        value = values(source, t[None], (x + shift)[None])[0]
        assert bool((jets[0].lower <= value).all() & (value <= jets[0].upper).all())
