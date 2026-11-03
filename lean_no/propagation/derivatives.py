import torch

from lean_no.adapters.fno import build_graph
from lean_no.domain import FourierGrid
from lean_no.function_space.input import FourierInput, InputBounds
from lean_no.graph import Field
from lean_no.propagation.derivative_cells import DerivativeField, InputField

def derivative_field(
    model,
    grid,
    inputs,
    moment_order,
    moment_batch,
    progress,
    integration="gaussian",
    integration_options=None,
    strategy="general",
    orders=(),
):
    model = build_graph(model, grid.dtype)
    if not isinstance(grid, FourierGrid) or model.dimension != len(grid.resolution):
        raise ValueError(
            "derivative fields require an eval-mode Fourier model and matching domain"
        )
    if len(grid.lengths) != model.dimension:
        raise ValueError("derivative fields require matching geometry")
    if type(moment_batch) is not int or moment_batch < 1:
        raise ValueError("positive moment batch required")
    if isinstance(inputs, FourierInput):
        inputs = inputs.bounds()
    if orders and max(map(sum, orders)) > 1 and "relu" in model.activations:
        raise ValueError(
            "higher ReLU derivatives do not exist across activation boundaries"
        )
    arithmetic = "affine" if strategy == "taylor_cell" else "interval"
    if isinstance(inputs, Field):
        coefficients = inputs.data["coefficients"]
        if (
            len(inputs.data["modes"]) != model.dimension
            or coefficients.shape[0] != model.in_channels
        ):
            raise ValueError("input Fourier axes and channels must match the graph")
        if not torch.isfinite(coefficients).all() or bool((inputs.error != 0).any()):
            raise ValueError("finite exact Fourier input required")
        field = DerivativeField(model, inputs, arithmetic)
    elif isinstance(inputs, InputBounds):
        if strategy == "taylor_cell":
            raise ValueError("Taylor cells require an exact Fourier input")
        source = InputField(
            inputs, grid.lengths, model.reference.new_zeros(inputs.channels)
        )
        field = DerivativeField(model, source, arithmetic)
    else:
        field = DerivativeField.from_profile(model, inputs, grid.lengths[1])
        field.arithmetic = arithmetic
    if arithmetic == "affine" and any(
        name not in ("tanh", "identity") for name in field.network.activations
    ):
        raise ValueError("Taylor cell coefficients require tanh or identity")
    if any(
        op.kind not in ("affine", "add", "spectral", "activation") for op in model.ops
    ):
        raise NotImplementedError(
            "no derivative propagation rule for this graph operation"
        )
    field.requested = tuple(orders)
    field.resolution = grid.resolution
    if integration == "arb":
        if moment_order is not None or moment_batch != 8192:
            raise ValueError("Arb uses tolerance and digits, not Gaussian settings")
        field.certify_coefficients(
            progress=progress,
            integration="arb",
            **(integration_options or {}),
        )
    else:
        options = dict(integration_options or {})
        if set(options) - {"panels", "order"}:
            raise ValueError("Fourier integration options are panels and order")
        field.certify_coefficients(
            options.get("order", 16 if hasattr(inputs, "data") else 2)
            if moment_order is None
            else moment_order,
            moment_batch,
            progress,
            panels=options.get("panels"),
            integration=integration,
        )
    return field
