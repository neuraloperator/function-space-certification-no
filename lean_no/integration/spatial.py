from __future__ import annotations

import itertools
import math

import torch
from neuralop.layers.spectral_convolution import SpectralConv

from lean_no.bounds.operators import convolution, interval_linear
from lean_no.kernel import kernel_gain, kernel_series, kernel_with_partials

def spatial(source, basis=None, domain=None, method="rectangle", options=None):
    from lean_no.integration import AdjointKernel

    if domain is not None and tuple(domain) != tuple(source.lengths):
        raise ValueError("integration domain differs from source geometry")
    if options is None:
        options = {}
    if isinstance(options, int):
        options = {"channel_chunk": options}
    if not isinstance(options, dict):
        raise TypeError("spatial integration options must be a dictionary")
    order = options.get("order", 2)
    if isinstance(basis, AdjointKernel):
        return adjoint_convolution(
            basis.module, source, basis.cache, method=method, order=order
        )
    if basis is None:
        return cell_integral(source, method=method, order=order)
    return convolution_integral(
        basis, source, options.get("channel_chunk"), method=method, order=order
    )

def convolution_integral(
    module,
    bounds,
    channel_chunk: int | None = None,
    method: str = "rectangle",
    order: int = 2,
):
    from lean_no.propagation.linear import Bounds

    if channel_chunk is not None:
        if (
            isinstance(channel_chunk, bool)
            or not isinstance(channel_chunk, int)
            or channel_chunk <= 0
        ):
            raise ValueError("channel_chunk must be a positive integer")
        if channel_chunk < module.out_channels:
            return chunked_conv(
                module, bounds, channel_chunk, method=method, order=order
            )
    axes = tuple(
        axis.to(device=bounds.node_lower.device, dtype=bounds.node_lower.dtype)
        for axis in bounds.axes
    )
    kernel_values, kernel_partials = kernel_with_partials(module, axes, bounds.lengths)
    coordinates = torch.meshgrid(*axes, indexing="ij")
    lower = bounds.node_lower.unsqueeze(0)
    upper = bounds.node_upper.unsqueeze(0)
    # The ablation changes only this positive quadrature sum and its certified
    # Lipschitz remainder. The spatial IBP source and target grids are identical.
    from lean_no.integration.rules import interval_rule

    nodes, weights, distance = interval_rule(method, order)
    if method == "rectangle":

        def quadrature_sum(derivative_order, values):
            return interval_linear(values, lower, upper, convolution)
    else:

        def quadrature_sum(derivative_order, values):
            result_lower = result_upper = None
            for indices in itertools.product(
                range(len(nodes)), repeat=bounds.dimension
            ):
                offsets = [float(nodes[i]) * h for i, h in zip(indices, bounds.steps)]
                weight = math.prod(float(weights[i]) for i in indices)
                shifted = tuple(x - d for x, d in zip(coordinates, offsets))
                kernel = kernel_series(
                    module, shifted, bounds.lengths, derivative_order
                )
                # Evaluate the SAME stored spatial source enclosure at the
                # Gaussian node using its existing cellwise derivative bounds.
                radius = sum(
                    d * bounds.partial_magnitude[:, a] for a, d in enumerate(offsets)
                )
                lo, hi = interval_linear(
                    kernel, lower - radius[None], upper + radius[None], convolution
                )
                result_lower = (
                    weight * lo if result_lower is None else result_lower + weight * lo
                )
                result_upper = (
                    weight * hi if result_upper is None else result_upper + weight * hi
                )
            return result_lower, result_upper

    output_lower, output_upper = quadrature_sum((0,) * bounds.dimension, kernel_values)
    output_lower = output_lower[0]
    output_upper = output_upper[0]
    bias = getattr(module, "bias", None)
    if bias is not None:
        shape = (module.out_channels,) + (1,) * bounds.dimension
        bias = bias.detach().to(output_lower).reshape(shape)
        output_lower = output_lower + bias
        output_upper = output_upper + bias

    spacing = output_lower.new_tensor(bounds.steps)
    kernel_shape = (
        module.out_channels,
        module.in_channels,
        *((1,) * bounds.dimension),
    )
    hessian_values: list[list[torch.Tensor]] = []
    hessian_suprema: list[list[torch.Tensor]] = []
    third_gains: list[list[list[torch.Tensor]]] = []
    value_cache: dict[tuple[int, ...], torch.Tensor] = {}
    gain_cache: dict[tuple[int, ...], torch.Tensor] = {}

    def gain(orders: tuple[int, ...]) -> torch.Tensor:
        if orders not in gain_cache:
            gain_cache[orders] = kernel_gain(
                module,
                bounds.lengths,
                orders,
                output_lower.dtype,
            ).to(output_lower)
        return gain_cache[orders]

    for output_axis in range(bounds.dimension):
        value_row = []
        supremum_row = []
        gain_row = []
        for source_axis in range(bounds.dimension):
            orders = [0] * bounds.dimension
            orders[output_axis] += 1
            orders[source_axis] += 1
            key = tuple(orders)
            if key not in value_cache:
                value_cache[key] = kernel_series(
                    module,
                    coordinates,
                    bounds.lengths,
                    key,
                )
            hessian = value_cache[key]
            gains = []
            radius = torch.zeros(
                module.out_channels,
                module.in_channels,
                device=output_lower.device,
                dtype=output_lower.dtype,
            )
            for third_axis in range(bounds.dimension):
                third_orders = list(key)
                third_orders[third_axis] += 1
                third_key = tuple(third_orders)
                third_gain = gain(third_key)
                gains.append(third_gain)
                radius = radius + spacing[third_axis] * third_gain
            value_row.append(hessian)
            local_supremum = hessian.abs() + radius.reshape(kernel_shape)
            supremum_row.append(
                torch.minimum(
                    gain(key).reshape(kernel_shape),
                    local_supremum,
                )
            )
            gain_row.append(gains)
        hessian_values.append(value_row)
        hessian_suprema.append(supremum_row)
        third_gains.append(gain_row)

    kernel_partial_suprema = []
    for axis, partial in enumerate(kernel_partials):
        local_supremum = partial.abs()
        for source_axis in range(bounds.dimension):
            local_supremum = (
                local_supremum
                + spacing[source_axis] * hessian_suprema[axis][source_axis]
            )
        orders = tuple(int(index == axis) for index in range(bounds.dimension))
        kernel_partial_suprema.append(
            torch.minimum(
                gain(orders).reshape(kernel_shape),
                local_supremum,
            )
        )
    local_supremum = kernel_values.abs()
    for axis in range(bounds.dimension):
        local_supremum = local_supremum + spacing[axis] * kernel_partial_suprema[axis]
    kernel_supremum = torch.minimum(
        gain((0,) * bounds.dimension).reshape(kernel_shape),
        local_supremum,
    )
    cell_lower, cell_upper = bounds.cell_bounds
    # S_{a,j} = max(|a̲_j|, |ā_j|) on cell C_j;  M_{a,k,j} = max |∂_k a| on C_j.
    cell_supremum = torch.maximum(cell_lower.abs(), cell_upper.abs())
    partial_magnitude = bounds.partial_magnitude
    # |∂_s(K a)| ≤ |∂K| |a| + |K| |∂a| on each source cell.
    correction = torch.zeros_like(output_lower)
    shape = (module.out_channels,) + (1,) * bounds.dimension
    for axis in range(bounds.dimension):
        field_term = convolution(
            kernel_supremum,
            partial_magnitude[None, :, axis],
        )[0]
        kernel_term = convolution(kernel_partial_suprema[axis], cell_supremum[None])[0]
        correction = correction + distance * spacing[axis] * (kernel_term + field_term)
    output_lower = output_lower - correction
    output_upper = output_upper + correction

    partial_lower = []
    partial_upper = []
    node_part_lower = []
    node_part_upper = []
    for output_axis in range(bounds.dimension):
        order = tuple(int(i == output_axis) for i in range(bounds.dimension))
        lower_node, upper_node = quadrature_sum(order, kernel_partials[output_axis])
        lower_node = lower_node[0]
        upper_node = upper_node[0]
        partial_correction = torch.zeros_like(output_lower)
        cell_radius = torch.zeros_like(output_lower)
        for source_axis in range(bounds.dimension):
            kernel_term = convolution(
                hessian_suprema[output_axis][source_axis],
                cell_supremum[None],
            )[0]
            field_term = convolution(
                kernel_partial_suprema[output_axis],
                partial_magnitude[None, :, source_axis],
            )[0]
            partial_correction = partial_correction + (
                distance * spacing[source_axis] * (kernel_term + field_term)
            )
            cell_radius = cell_radius + spacing[source_axis] * kernel_term
        node_radius = partial_correction
        radius = partial_correction + cell_radius
        # ∂_p g̲_j = ∂_p ĝ̲_Q - η_1 - ∑_q h_q M_{g',q},  likewise for upper.
        partial_lower.append(lower_node - radius)
        partial_upper.append(upper_node + radius)
        node_part_lower.append(lower_node - node_radius)
        node_part_upper.append(upper_node + node_radius)

    partial_lower = torch.stack(partial_lower, dim=1)
    partial_upper = torch.stack(partial_upper, dim=1)
    node_part_lower = torch.stack(node_part_lower, dim=1)
    node_part_upper = torch.stack(node_part_upper, dim=1)
    return Bounds(
        axes=bounds.axes,
        lengths=bounds.lengths,
        node_lower=output_lower,
        node_upper=output_upper,
        partial_lower=partial_lower,
        partial_upper=partial_upper,
        partial_node_lower=(
            node_part_lower if bounds.partial_node_lower is not None else None
        ),
        partial_node_upper=(
            node_part_upper if bounds.partial_node_upper is not None else None
        ),
        stats={
            "quadrature_order": len(nodes),
            "transport_distance": distance,
            "quadrature_correction_max": float(correction.max()),
        },
    )

def cell_integral(
    field,
    method="rectangle",
    order=2,
) -> tuple[torch.Tensor, torch.Tensor]:
    from lean_no.integration.rules import interval_rule

    nodes, weights, distance = interval_rule(method, order)
    dimensions = tuple(
        range(field.node_lower.ndim - field.dimension, field.node_lower.ndim)
    )
    derivative_axis = field.node_lower.ndim - field.dimension
    if method == "rectangle":
        average_distance = 0.5
    else:
        # The stored lower/upper cell envelopes are linear cones. Positive
        # rules exact on linear functions integrate these bounds exactly;
        # their interpolation radius already encloses the actual function.
        average_distance = float(sum(nodes * weights))
    radius = sum(
        average_distance * h * field.partial_magnitude.select(derivative_axis, axis)
        for axis, h in enumerate(field.steps)
    )
    return (
        (field.node_lower - radius).mean(dim=dimensions),
        (field.node_upper + radius).mean(dim=dimensions),
    )

def slice_conv(module: SpectralConv, start: int, stop: int) -> SpectralConv:
    from lean_no.graph import Operator
    from lean_no.kernel import mode_weights

    reference = (
        module.weight if isinstance(module, Operator) else module.weight.to_tensor()
    )
    weight, modes = mode_weights(module, reference.real.dtype)
    bias = None if module.bias is None else module.bias.flatten()[start:stop]
    return Operator(
        "slice",
        "spectral",
        (0,),
        dict(weight=weight[start:stop], bias=bias, modes=modes, basis="fourier"),
    )

def chunked_conv(
    module: SpectralConv,
    bounds,
    channel_chunk: int,
    method: str = "rectangle",
    order: int = 2,
):
    names = (
        "node_lower",
        "node_upper",
        "partial_lower",
        "partial_upper",
        "partial_node_lower",
        "partial_node_upper",
    )
    from lean_no.propagation.linear import Bounds

    output = None
    for start in range(0, module.out_channels, channel_chunk):
        stop = min(start + channel_chunk, module.out_channels)
        part = convolution_integral(
            slice_conv(module, start, stop), bounds, method=method, order=order
        )
        if output is None:
            output = {}
            for name in names:
                value = getattr(part, name)
                output[name] = (
                    None
                    if value is None
                    else value.new_empty(module.out_channels, *value.shape[1:])
                )
        for name in names:
            value = getattr(part, name)
            if value is not None:
                output[name][start:stop].copy_(value)
        del part
    return Bounds(bounds.axes, bounds.lengths, **output)

def adjoint_convolution(
    conv: SpectralConv,
    field,
    cache: dict | None = None,
    method="rectangle",
    order=2,
):
    from lean_no.integration.rules import interval_rule
    from lean_no.kernel import kernel_series
    from lean_no.propagation.crown import (
        KernelField,
        add_constant,
        adjoint_plan,
        frequency_conv,
        frequency_plan,
    )

    nodes, weights, distance = interval_rule(method, order)

    plan = adjoint_plan(
        conv,
        field.axes,
        field.lengths,
        field.node_lower,
        cache,
    )
    frequency = frequency_plan(
        conv,
        field.axes,
        field.lengths,
        field.node_lower,
        cache,
    )
    dimension = field.dimension
    resolution = field.resolution
    dimensions = tuple(range(-dimension, 0))
    center_frequency = torch.fft.rfftn(
        0.5 * (field.node_lower + field.node_upper),
        dim=dimensions,
    )
    radius_frequency = torch.fft.rfftn(
        0.5 * (field.node_upper - field.node_lower),
        dim=dimensions,
    )
    if method == "rectangle":
        node_center = frequency_conv(
            frequency.values,
            center_frequency,
            resolution,
        )
        node_radius = frequency_conv(
            frequency.absolute_values,
            radius_frequency,
            resolution,
        )
        node_lower = node_center - node_radius
        node_upper = node_center + node_radius

    else:

        def quadrature_sum(derivative_axis=None):
            center = radius = None
            for indices in itertools.product(range(len(nodes)), repeat=dimension):
                offsets = tuple(
                    float(nodes[i]) * h for i, h in zip(indices, field.steps)
                )
                weight = math.prod(float(weights[i]) for i in indices)
                key = (
                    "positive_adjoint",
                    id(conv),
                    field.resolution,
                    field.lengths,
                    field.node_lower.dtype,
                    field.node_lower.device,
                    offsets,
                    derivative_axis,
                )
                found = None if cache is None else cache.get(key)
                if found is None:
                    coordinates = torch.meshgrid(*field.axes, indexing="ij")
                    coordinates = tuple(d - x for d, x in zip(offsets, coordinates))
                    powers = tuple(int(i == derivative_axis) for i in range(dimension))
                    kernel = kernel_series(
                        conv, coordinates, field.lengths, powers
                    ).transpose(0, 1)
                    if derivative_axis is not None:
                        kernel = -kernel
                    found = (
                        torch.fft.rfftn(kernel, dim=dimensions),
                        torch.fft.rfftn(kernel.abs(), dim=dimensions),
                    )
                    if cache is not None:
                        cache[key] = found
                inherited = sum(
                    d * field.partial_magnitude[:, :, i] for i, d in enumerate(offsets)
                )
                source_radius = (field.node_upper - field.node_lower) / 2 + inherited
                rfrequency = torch.fft.rfftn(source_radius, dim=dimensions)
                c = weight * frequency_conv(found[0], center_frequency, resolution)
                r = weight * frequency_conv(found[1], rfrequency, resolution)
                center = c if center is None else center + c
                radius = r if radius is None else radius + r
            return center - radius, center + radius

        node_lower, node_upper = quadrature_sum()
    hessian = plan.hessian_gain
    spatial = tuple(range(2, field.node_lower.ndim))
    partial_frequencies = tuple(
        torch.fft.rfftn(
            field.partial_magnitude[:, :, axis],
            dim=dimensions,
        )
        for axis in range(dimension)
    )
    partial_means = tuple(
        field.partial_magnitude[:, :, axis].mean(dim=spatial)
        for axis in range(dimension)
    )
    cell_supremum = field.cell_supremum
    cell_frequency = torch.fft.rfftn(cell_supremum, dim=dimensions)
    cell_mean = cell_supremum.mean(dim=spatial)
    # η_adj = ∑_k (h_k / 2) (S_K * M_{C,k} + S_{∂_k K} * S_C).
    correction = torch.zeros_like(node_lower)
    for axis, spacing in enumerate(field.steps):
        field_term = frequency_conv(
            frequency.absolute_values,
            partial_frequencies[axis],
            resolution,
        )
        field_term = add_constant(
            field_term,
            plan.value_radius,
            partial_means[axis],
            dimension,
        )
        kernel_term = frequency_conv(
            frequency.absolute_derivatives[axis],
            cell_frequency,
            resolution,
        )
        kernel_term = add_constant(
            kernel_term,
            plan.derivative_radii[axis],
            cell_mean,
            dimension,
        )
        correction = correction + distance * spacing * (field_term + kernel_term)
    # C̲(s_j) = Ĉ̲_Q - η_adj,  C̄(s_j) = Ĉ̄_Q + η_adj.
    node_lower = node_lower - correction
    node_upper = node_upper + correction

    partial_lower = []
    partial_upper = []
    for row in range(dimension):
        if method == "rectangle":
            derivative_center = frequency_conv(
                frequency.derivatives[row],
                center_frequency,
                resolution,
            )
            derivative_radius = frequency_conv(
                frequency.absolute_derivatives[row],
                radius_frequency,
                resolution,
            )
            lower = derivative_center - derivative_radius
            upper = derivative_center + derivative_radius
        else:
            lower, upper = quadrature_sum(row)
        # η_1 = ∑_q (h_q / 2) (S_{∂_p K} M_{C,q} + M_{K,pq} mean_j S_{C,j}).
        correction = torch.zeros_like(lower)
        cell_radius = field.node_lower.new_zeros(
            field.node_lower.shape[0],
            conv.in_channels,
        )
        for column, spacing in enumerate(field.steps):
            field_term = frequency_conv(
                frequency.absolute_derivatives[row],
                partial_frequencies[column],
                resolution,
            )
            field_term = add_constant(
                field_term,
                plan.derivative_radii[row],
                partial_means[column],
                dimension,
            )
            kernel_term = torch.einsum(
                "io,so->si",
                hessian[:, :, row, column],
                cell_mean,
            )
            correction = correction + distance * spacing * (
                field_term
                + kernel_term.reshape(
                    lower.shape[0],
                    lower.shape[1],
                    *((1,) * dimension),
                )
            )
            cell_radius = cell_radius + spacing * kernel_term
        # In-cell radius ∑_q h_q M_{K,pq} mean_j S_{C,j}.
        radius = cell_radius.reshape(
            lower.shape[0],
            lower.shape[1],
            *((1,) * dimension),
        )
        # ∂_p C̲_j = ∂_p Ĉ̲_Q - η_1 - ∑_q h_q M_{C',q},  likewise for upper.
        partial_lower.append(lower - correction - radius)
        partial_upper.append(upper + correction + radius)

    partial_lower = torch.stack(partial_lower, dim=2)
    partial_upper = torch.stack(partial_upper, dim=2)
    return KernelField(
        axes=field.axes,
        lengths=field.lengths,
        node_lower=node_lower,
        node_upper=node_upper,
        partial_lower=partial_lower,
        partial_upper=partial_upper,
        rule=field.rule,
    )
