from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import torch
from flint import acb, ctx
from flint import arb as flint_arb

from lean_no.bounds.activations import gelu_first
from lean_no.bounds.operators import (
    affine_bias,
    affine_weight,
    convolution,
    interval_linear,
    is_gelu,
    is_identity,
    is_relu,
    is_tanh,
)
from lean_no.graph import execute, fourier
from lean_no.kernel import kernel_series, mode_weights

ArbPair = tuple[tuple[acb, ...], tuple[acb, ...]]
ArbFn = Callable[[acb, bool], ArbPair]

def arb_float(value: float) -> flint_arb:
    numerator, denominator = float(value).as_integer_ratio()
    return flint_arb(f"{numerator}/{denominator}")

def acb_float(value: complex) -> acb:
    return acb(arb_float(value.real), arb_float(value.imag))

def safe_float(value: flint_arb, upper: bool) -> float:
    endpoint = value.upper() if upper else value.lower()
    return float(endpoint)

def integrate_square(function, tolerance):

    def outer(x, analytic):
        return acb.integral(
            lambda y, branch: function((x, y), analytic or branch),
            0,
            1,
            abs_tol=tolerance / 4,
        )

    return acb.integral(outer, 0, 1, abs_tol=tolerance)

def arb_series(
    weights: tuple[acb, ...],
    positive_phase: acb,
    negative_phase: acb,
) -> acb:
    value = acb(weights[0].real)
    if len(weights) == 1:
        return value
    positive = weights[-1]
    negative = weights[-1].conjugate()
    for mode in range(len(weights) - 2, 0, -1):
        positive = positive * positive_phase + weights[mode]
        negative = negative * negative_phase + weights[mode].conjugate()
    return value + positive * positive_phase + negative * negative_phase

def arb_kernels(
    weights: tuple[tuple[acb, ...], ...],
    point: acb,
) -> tuple[acb, ...]:
    if point.real.is_exact() and point.imag.is_exact():
        positive = (2 * point).exp_pi_i()
        negative = (-2 * point).exp_pi_i()
        return tuple(arb_series(channel, positive, negative) for channel in weights)
    values = [acb(channel[0].real) for channel in weights]
    for mode in range(1, len(weights[0])):
        angle = 2 * mode * point
        cosine = angle.cos_pi()
        sine = angle.sin_pi()
        for index, channel in enumerate(weights):
            weight = channel[mode]
            values[index] += 2 * (weight.real * cosine - weight.imag * sine)
    return tuple(values)

@dataclass(frozen=True)
class State:

    value: ArbFn
    channels: int
    exact: bool = True

def affine_state(module, previous: State) -> State:
    tensor = affine_weight(module).detach().cpu().double()
    bias_tensor = affine_bias(module, tensor)
    if bias_tensor is None:
        bias_tensor = torch.zeros(tensor.shape[0], dtype=tensor.dtype)
    else:
        bias_tensor = bias_tensor.detach().cpu().double()
    raw = tensor.tolist()
    weights = tuple(tuple(arb_float(value) for value in row) for row in raw)
    biases = tuple(arb_float(value) for value in bias_tensor.tolist())

    def bounds(z: acb, analytic: bool) -> ArbPair:
        lower, _ = previous.value(z, analytic)
        output = tuple(
            acb(bias) + sum((w * v for w, v in zip(row, lower)), acb(0))
            for row, bias in zip(weights, biases)
        )
        return output, output

    return State(bounds, tensor.shape[0], previous.exact)

def add_state(left: State, right: State) -> State:
    def bounds(z: acb, analytic: bool) -> ArbPair:
        l_lo, _ = left.value(z, analytic)
        r_lo, _ = right.value(z, analytic)
        out = tuple(a + b for a, b in zip(l_lo, r_lo))
        return out, out

    return State(bounds, left.channels, left.exact and right.exact)

def activation_state(act, previous: State) -> State:
    if is_identity(act):
        return previous
    if is_relu(act):
        zero = acb(0)

        def bounds(z: acb, analytic: bool) -> ArbPair:
            lo, _ = previous.value(z, analytic)
            out = tuple(v.real_max(zero, analytic=analytic) for v in lo)
            return out, out

        return State(bounds, previous.channels, previous.exact)
    if is_gelu(act):
        if getattr(act, "approximate", "none") != "none":
            raise NotImplementedError("Arb GELU requires approximate='none'")

        def gelu(value):
            return value * (1 + (value / flint_arb(2).sqrt()).erf()) / 2

        def bounds(z, analytic):
            lo, _ = previous.value(z, analytic)
            values = tuple(gelu(value) for value in lo)
            return values, values

        return State(bounds, previous.channels, previous.exact)
    if is_tanh(act):

        def bounds(z: acb, analytic: bool) -> ArbPair:
            lo, _ = previous.value(z, analytic)

            def apply(value):
                if analytic and value.cosh().contains(0):
                    return acb("nan")
                if value.imag.is_zero():
                    return acb(value.real.tanh())
                return value.tanh()

            out = tuple(map(apply, lo))
            return out, out

        return State(bounds, previous.channels, previous.exact)
    raise NotImplementedError("Arb supports ReLU, tanh, exact GELU or Identity")

def fourier_state(coefficients, axes):
    frequencies = list(itertools.product(*(axis.tolist() for axis in axes)))
    indices = list(itertools.product(*(range(len(axis)) for axis in axes)))
    rows = []
    for channel in coefficients:
        row = []
        for index in indices:
            value = channel
            for i in index:
                value = value[i]
            row.append(value)
        rows.append(row)

    def evaluate(point, analytic):
        coordinates = (point,) if len(axes) == 1 else point
        phases = [
            (2 * sum(k * x for k, x in zip(frequency, coordinates))).exp_pi_i()
            for frequency in frequencies
        ]
        negative = [
            (-2 * sum(k * x for k, x in zip(frequency, coordinates))).exp_pi_i()
            for frequency in frequencies
        ]
        values = tuple(
            sum(
                (
                    (1 if frequency[-1] == 0 else 2)
                    * (value * positive + value.conjugate() * reflected)
                    / 2
                    for value, frequency, positive, reflected in zip(
                        row, frequencies, phases, negative
                    )
                ),
                acb(0),
            )
            for row in rows
        )
        return values, values

    return State(evaluate, len(rows), True)

def exact_coefficients(
    previous: State,
    weights: tuple,
    length: flint_arb,
    tolerance: flint_arb,
):
    modes = len(weights[0][0])
    coefficients = []
    for channel in range(previous.channels):
        values = []
        for mode in range(modes):

            def integrand(point, analytic):
                low, _ = previous.value(length * point, analytic)
                return low[channel] * (-2 * mode * point).exp_pi_i()

            value = acb.integral(integrand, acb(0), acb(1), abs_tol=tolerance)
            if not value.is_finite():
                raise ArithmeticError(
                    "Arb could not enclose a fixed Fourier coefficient"
                )
            values.append(value)
        coefficients.append(tuple(values))
    return tuple(
        tuple(
            sum(
                (
                    row[channel][mode] * coefficients[channel][mode]
                    for channel in range(previous.channels)
                ),
                acb(0),
            )
            for mode in range(modes)
        )
        for row in weights
    )

@torch.no_grad()
def prepare_coefficients(
    field, series=None, tolerance="1e-8", digits=30, progress=None
):
    dimension = len(field.network.blocks[0].conv.n_modes)
    if dimension not in (1, 2):
        raise ValueError("Arb coefficients require a one- or two-dimensional model")
    if series is not None and (len(series.slopes) != dimension or any(series.slopes)):
        raise ValueError("Arb coefficients require a fixed trigonometric input")
    if field.forcing.error.device.type != "cpu":
        raise ValueError("Arb coefficients require a CPU field")
    if type(digits) is not int or digits < 15:
        raise ValueError("Arb digits must be an integer of at least 15")
    if not flint_arb(tolerance).is_finite() or not flint_arb(tolerance) > 0:
        raise ValueError("Arb tolerance must be positive and finite")
    completed = 0
    layer_coefficients, records = {}, []

    forcing = field.forcing.data
    raw = forcing["coefficients"]

    def convert(value):
        return (
            [convert(part) for part in value]
            if isinstance(value, list)
            else acb_float(value)
        )

    coefficients = convert(raw.tolist())
    exact_input = fourier_state(coefficients, forcing["modes"])

    def source_value(point, analytic):
        coordinates = (point,) if dimension == 1 else point
        values = exact_input.value(point, analytic)[0]
        if getattr(field, "time_period", None) is not None:
            values = (*values, coordinates[1] * arb_float(field.time_period))
        if field.network.grid_embedding:
            values = (*values, *coordinates)
        return values, values

    source = State(
        source_value,
        (dimension if field.network.grid_embedding else 0)
        + (2 if getattr(field, "time_period", None) is not None else 1),
        True,
    )

    def apply(op, parents):
        nonlocal completed
        state, error = parents[0]
        if op.kind == "affine":
            return affine_state(op, state), affine_weight(op).abs() @ error
        if op.kind == "add":
            return add_state(state, parents[1][0]), error + parents[1][1]
        if op.kind == "activation":
            act = op.data["activation"]
            return activation_state(act, state), error * (
                gelu_first if is_gelu(act) else 1
            )
        module = op
        if getattr(module, "separable", False):
            raise NotImplementedError(
                "Arb coefficients do not support separable kernels"
            )
        w, axes = mode_weights(module, torch.float64)
        if dimension == 1:
            weights = tuple(
                tuple(tuple(acb_float(z) for z in row) for row in channel)
                for channel in w.tolist()
            )
            values = exact_coefficients(
                state, weights, flint_arb(1), flint_arb(tolerance)
            )
            if any(not z.is_finite() for row in values for z in row):
                raise ArithmeticError("Arb could not enclose a 1D Fourier coefficient")
            nominal = tuple(
                tuple(acb(z.real.mid(), z.imag.mid()) for z in row) for row in values
            )
            c = torch.tensor(
                [
                    [complex(float(z.real), float(z.imag)) for z in row]
                    for row in nominal
                ],
                dtype=torch.complex128,
            )
            ce = torch.tensor(
                [
                    [safe_float(abs(z - n), True) for z, n in zip(row, nr)]
                    for row, nr in zip(values, nominal)
                ],
                dtype=torch.float64,
            )
            ce += torch.einsum("oik,i->ok", w.abs(), error)
            output_error = (ce * torch.where(axes[0] == 0, 1.0, 2.0)).sum(-1)
            bias = affine_bias(module, error)
            if bias is not None:
                c[:, 0] += bias
                nominal = tuple(
                    tuple(
                        z + (arb_float(b) if i == 0 else 0) for i, z in enumerate(row)
                    )
                    for row, b in zip(nominal, bias.tolist())
                )
            node = fourier(c, axes, output_error)
            node.data["coefficient_error"] = ce
            layer_coefficients[op.name] = node
            completed += state.channels * len(axes[0])
            record = dict(
                operator=op.name,
                completed_coefficients=completed,
                coefficient_error_max=float(ce.max()),
            )
            records.append(record)
            if progress is not None:
                progress(record)

            def evaluate(point, analytic):
                result = arb_kernels(nominal, point)
                return result, result

            return State(evaluate, len(nominal), True), output_error
        if len(axes) != 2:
            raise ValueError("Arb square coefficients require a 2D kernel")
        kx, ky = [axis.tolist() for axis in axes]
        integrals = []
        for channel in range(state.channels):
            rows = []
            for a in kx:
                row = []
                for b in ky:

                    def integrand(point, analytic):
                        values, _ = state.value(point, analytic)
                        return (
                            values[channel]
                            * (-2 * (a * point[0] + b * point[1])).exp_pi_i()
                        )

                    value = integrate_square(integrand, flint_arb(tolerance))
                    if not value.is_finite():
                        raise ArithmeticError(
                            "Arb could not enclose a 2D Fourier coefficient"
                        )
                    row.append(value)
                    completed += 1
                    if progress is not None:
                        progress(
                            dict(operator=op.name, completed_coefficients=completed)
                        )
                rows.append(row)
            integrals.append(rows)
        values = [
            [
                [
                    sum(
                        (
                            acb_float(w[o, i, a, b].item()) * integrals[i][a][b]
                            for i in range(state.channels)
                        ),
                        acb(0),
                    )
                    for b in range(len(ky))
                ]
                for a in range(len(kx))
            ]
            for o in range(w.shape[0])
        ]
        bias = affine_bias(module, error)
        if bias is not None:
            for o, item in enumerate(bias.tolist()):
                values[o][kx.index(0.0)][0] += arb_float(item)
        nominal = [
            [[acb(z.real.mid(), z.imag.mid()) for z in row] for row in channel]
            for channel in values
        ]
        c = torch.tensor(
            [
                [
                    [complex(float(z.real), float(z.imag)) for z in row]
                    for row in channel
                ]
                for channel in nominal
            ],
            dtype=torch.complex128,
        )
        ce = torch.tensor(
            [
                [
                    [safe_float(abs(z - n), True) for z, n in zip(row, nr)]
                    for row, nr in zip(channel, nc)
                ]
                for channel, nc in zip(values, nominal)
            ],
            dtype=torch.float64,
        )
        ce += torch.einsum("oikl,i->okl", w.abs(), error)
        output_error = (ce * torch.where(axes[1] == 0, 1.0, 2.0)).sum((-2, -1))
        node = fourier(c, axes, output_error)
        node.data["coefficient_error"] = ce
        layer_coefficients[op.name] = node
        records.append(
            dict(
                operator=op.name,
                completed_coefficients=completed,
                coefficient_error_max=float(ce.max()),
            )
        )
        return fourier_state(nominal, axes), output_error

    with ctx.workdps(digits):
        execute(
            field.network,
            (source, torch.zeros(source.channels, dtype=torch.float64)),
            apply,
        )
    return layer_coefficients, records

def arb_kernel_moments(module, coordinates, lengths, steps, orders, options=None):
    options = options or {}
    tolerance = flint_arb(options.get("tolerance", "1e-10"))
    digits = options.get("digits", 15)
    dimension = len(coordinates)
    reference = coordinates[0]
    weight, modes = mode_weights(module, reference.dtype)
    weight = weight.to(device=reference.device)
    frequencies = tuple(
        2 * torch.pi * mode.to(reference) / float(length)
        for mode, length in zip(modes, lengths)
    )
    weighted = weight.clone()
    radius_factors = []
    with ctx.workdps(digits):
        for axis, (mode, length, step, order) in enumerate(
            zip(modes, lengths, steps, orders)
        ):
            shape = [1] * dimension
            shape[axis] = mode.numel()
            frequency = frequencies[axis]
            weighted = weighted * (1j * frequency).pow(order).reshape(1, 1, *shape)
            mids = []
            rads = []
            ratio = arb_float(step) / arb_float(length)
            for k in mode.tolist():
                alpha = arb_float(k) * ratio
                if alpha.is_zero():
                    val = acb(1)
                else:
                    val = acb.integral(
                        lambda u, analytic, a=alpha: (-2 * a * u).exp_pi_i(),
                        0,
                        1,
                        abs_tol=tolerance,
                    )
                mid = acb(val.real.mid(), val.imag.mid())
                mids.append(complex(float(mid.real), float(mid.imag)))
                rads.append(safe_float(abs(val - mid), True))
            mid_tensor = torch.tensor(
                mids, device=reference.device, dtype=weighted.dtype
            ).reshape(1, 1, *shape)
            weighted = weighted * mid_tensor
            radius_factors.append(
                torch.tensor(rads, device=reference.device, dtype=reference.dtype)
            )

    point_count = reference.numel()
    symmetry = torch.ones_like(modes[-1])
    symmetry[1:] = 2
    shape = [1] * dimension
    shape[-1] = symmetry.numel()
    weighted_sym = weighted * symmetry.reshape(1, 1, *shape)
    channels = weight.shape[0] * weight.shape[1]
    result = torch.empty(
        channels, point_count, device=reference.device, dtype=reference.dtype
    )
    flat_coordinates = tuple(value.reshape(-1) for value in coordinates)
    for begin in range(0, point_count, 16384):
        end = min(begin + 16384, point_count)
        phase = torch.ones(
            end - begin,
            *(mode.numel() for mode in modes),
            device=reference.device,
            dtype=weight.dtype,
        )
        for axis, (coordinate, frequency) in enumerate(
            zip(flat_coordinates, frequencies)
        ):
            s = [1] * dimension
            s[axis] = frequency.numel()
            phase = phase * torch.exp(
                1j
                * coordinate[begin:end].reshape(-1, *([1] * dimension))
                * frequency.reshape(1, *s)
            )
        values = phase.reshape(end - begin, -1) @ weighted_sym.reshape(channels, -1).mT
        result[:, begin:end] = values.real.mT
    center = result.reshape(weight.shape[0], weight.shape[1], *reference.shape)
    total_rad = torch.zeros(
        *(mode.numel() for mode in modes),
        device=reference.device,
        dtype=reference.dtype,
    )
    for axis, r in enumerate(radius_factors):
        s = [1] * dimension
        s[axis] = r.numel()
        total_rad = total_rad + r.reshape(s)
    magnitude = torch.hypot(weight.real, weight.imag)
    for axis, (frequency, order) in enumerate(zip(frequencies, orders)):
        s = [1] * dimension
        s[axis] = frequency.numel()
        magnitude = magnitude * frequency.abs().pow(order).reshape(1, 1, *s)
    rad_bound = (
        magnitude * total_rad[None, None] * symmetry.reshape(1, 1, *shape)
    ).sum(dim=tuple(range(2, weight.ndim)))
    return center, rad_bound

def arb_spatial_conv(module, bounds, channel_chunk=None, options=None):
    from lean_no.integration.spatial import slice_conv
    from lean_no.kernel import kernel_gain, kernel_with_partials
    from lean_no.propagation.linear import Bounds

    if channel_chunk is not None:
        if (
            isinstance(channel_chunk, bool)
            or not isinstance(channel_chunk, int)
            or channel_chunk <= 0
        ):
            raise ValueError("channel_chunk must be a positive integer")
        if channel_chunk < module.out_channels:
            names = (
                "node_lower",
                "node_upper",
                "partial_lower",
                "partial_upper",
                "partial_node_lower",
                "partial_node_upper",
            )
            output = None
            for start in range(0, module.out_channels, channel_chunk):
                stop = min(start + channel_chunk, module.out_channels)
                part = arb_spatial_conv(
                    slice_conv(module, start, stop), bounds, options=options
                )
                if output is None:
                    output = {
                        name: (
                            None
                            if getattr(part, name) is None
                            else getattr(part, name).new_empty(
                                module.out_channels, *getattr(part, name).shape[1:]
                            )
                        )
                        for name in names
                    }
                for name in names:
                    val = getattr(part, name)
                    if val is not None:
                        output[name][start:stop].copy_(val)
            return Bounds(bounds.axes, bounds.lengths, **output)

    axes = tuple(
        axis.to(device=bounds.node_lower.device, dtype=bounds.node_lower.dtype)
        for axis in bounds.axes
    )
    kernel_values, kernel_partials = kernel_with_partials(module, axes, bounds.lengths)
    coordinates = torch.meshgrid(*axes, indexing="ij")
    spacing = bounds.node_lower.new_tensor(bounds.steps)
    half_radius = sum(
        0.5 * h * bounds.partial_magnitude[:, a] for a, h in enumerate(bounds.steps)
    )
    mid_lower = (bounds.node_lower - half_radius).unsqueeze(0)
    mid_upper = (bounds.node_upper + half_radius).unsqueeze(0)
    cell_lower, cell_upper = bounds.cell_bounds
    cell_supremum = torch.maximum(cell_lower.abs(), cell_upper.abs())

    def arb_sum(derivative_order):
        kernel_avg, rad_bound = arb_kernel_moments(
            module, coordinates, bounds.lengths, bounds.steps, derivative_order, options
        )
        lo, hi = interval_linear(kernel_avg, mid_lower, mid_upper, convolution)
        extra = torch.einsum("oi,i...->o...", rad_bound, cell_supremum).mean(
            dim=tuple(range(1, cell_supremum.ndim))
        )
        extra = extra.reshape((module.out_channels,) + (1,) * bounds.dimension)
        return lo[0] - extra, hi[0] + extra

    output_lower, output_upper = arb_sum((0,) * bounds.dimension)
    bias = getattr(module, "bias", None)
    if bias is not None:
        shape = (module.out_channels,) + (1,) * bounds.dimension
        bias = bias.detach().to(output_lower).reshape(shape)
        output_lower = output_lower + bias
        output_upper = output_upper + bias

    kernel_shape = (
        module.out_channels,
        module.in_channels,
        *((1,) * bounds.dimension),
    )
    gain_cache = {}

    def gain(orders):
        if orders not in gain_cache:
            gain_cache[orders] = kernel_gain(
                module, bounds.lengths, orders, output_lower.dtype
            ).to(output_lower)
        return gain_cache[orders]

    hessian_suprema = []
    for output_axis in range(bounds.dimension):
        supremum_row = []
        for source_axis in range(bounds.dimension):
            orders = [0] * bounds.dimension
            orders[output_axis] += 1
            orders[source_axis] += 1
            key = tuple(orders)
            hessian = kernel_series(module, coordinates, bounds.lengths, key)
            radius = sum(
                spacing[third_axis]
                * gain(
                    tuple(
                        orders[i] + (1 if i == third_axis else 0)
                        for i in range(bounds.dimension)
                    )
                )
                for third_axis in range(bounds.dimension)
            )
            supremum_row.append(
                torch.minimum(
                    gain(key).reshape(kernel_shape),
                    hessian.abs() + radius.reshape(kernel_shape),
                )
            )
        hessian_suprema.append(supremum_row)

    partial_lower = []
    partial_upper = []
    node_part_lower = []
    node_part_upper = []
    for output_axis in range(bounds.dimension):
        order = tuple(int(i == output_axis) for i in range(bounds.dimension))
        lower_node, upper_node = arb_sum(order)
        cell_radius = torch.zeros_like(output_lower)
        for source_axis in range(bounds.dimension):
            kernel_term = convolution(
                hessian_suprema[output_axis][source_axis],
                cell_supremum[None],
            )[0]
            cell_radius = cell_radius + spacing[source_axis] * kernel_term
        partial_lower.append(lower_node - cell_radius)
        partial_upper.append(upper_node + cell_radius)
        node_part_lower.append(lower_node)
        node_part_upper.append(upper_node)

    return Bounds(
        axes=bounds.axes,
        lengths=bounds.lengths,
        node_lower=output_lower,
        node_upper=output_upper,
        partial_lower=torch.stack(partial_lower, dim=1),
        partial_upper=torch.stack(partial_upper, dim=1),
        partial_node_lower=(
            torch.stack(node_part_lower, dim=1)
            if bounds.partial_node_lower is not None
            else None
        ),
        partial_node_upper=(
            torch.stack(node_part_upper, dim=1)
            if bounds.partial_node_upper is not None
            else None
        ),
        stats={
            "quadrature_order": 0,
            "transport_distance": 0.0,
            "quadrature_correction_max": 0.0,
        },
    )

def arb(source, basis=None, domain=None, options=None):
    from lean_no.integration.cell_kernel import CellKernel, cell_kernel_integral

    if isinstance(source, CellKernel):
        return cell_kernel_integral(source, "arb", options)
    from lean_no.integration.kernel_norm import LinearizedKernel, kernel_norm

    if isinstance(source, LinearizedKernel):
        return kernel_norm(source, "arb", options)
    if hasattr(source, "sizes"):
        from lean_no.integration.sphere import spherical_integral

        return spherical_integral(source, basis, domain, "arb", options)
    from lean_no.integration.gaussian import AnalyticSource, moment_integral

    if isinstance(source, AnalyticSource):
        return moment_integral(source, basis, domain, options, method="arb")
    if hasattr(source, "node_lower"):
        from lean_no.integration import AdjointKernel
        from lean_no.integration.spatial import adjoint_convolution, cell_integral

        if domain is not None and tuple(domain) != tuple(source.lengths):
            raise ValueError("integration domain differs from source geometry")
        if options is None:
            options = {}
        if isinstance(options, int):
            options = {"channel_chunk": options}
        if isinstance(basis, AdjointKernel):
            return adjoint_convolution(
                basis.module, source, basis.cache, method="arb", order=2
            )
        if basis is None:
            return cell_integral(source, method="arb", order=2)
        return arb_spatial_conv(
            basis, source, options.get("channel_chunk"), options=options
        )
    if basis is not None or domain is None or len(domain) != 2:
        raise ValueError("Arb requires a complete integrand and an interval")
    return acb.integral(source, *domain, **({} if options is None else options))
