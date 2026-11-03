import itertools
import math

import torch

from lean_no.bounds.operators import convolution
from lean_no.kernel import kernel_gain, kernel_series, mode_weights
from lean_no.spectral import real_inverse

def taylor_quadrature(source, basis=None, domain=None, options=None):
    if basis is None or options is not None:
        raise ValueError("Taylor integration requires a kernel and no extra options")
    if domain is not None and tuple(domain) != tuple(source.lengths):
        raise ValueError("Taylor domain differs from source geometry")
    return convolution_integral(basis, source)

def sampled_kernel(module, bounds, order):
    if bounds.dimension == 2:
        weight, modes = mode_weights(module, bounds.node_lower.dtype)
        for axis, power in enumerate(order):
            shape = [1, 1]
            shape[axis] = -1
            weight = weight * (
                (2 * math.pi * modes[axis] / bounds.lengths[axis]).pow(power)
                * (1j**power)
            ).reshape(shape)
        nx, nt = bounds.resolution
        spectrum = weight.new_zeros(*weight.shape[:2], nx, nt // 2 + 1)
        ix, it = torch.meshgrid(modes[0].long() % nx, modes[1].long(), indexing="ij")
        spectrum[:, :, ix, it] = weight
        return real_inverse(spectrum, bounds.resolution, "forward")
    order = order[0] if isinstance(order, tuple) else order
    axis = bounds.axes[0]
    result = None
    for start in range(0, axis.numel(), 65536):
        stop = min(start + 65536, axis.numel())
        values = kernel_series(module, (axis[start:stop],), bounds.lengths, (order,))
        if result is None:
            result = values.new_empty((*values.shape[:2], axis.numel()))
        result[..., start:stop] = values
    return result

def conv_channel(module, bounds):
    from lean_no.propagation import linear as ibp

    if bounds.second_lower is None:
        raise ValueError("Taylor quadrature requires curvature bounds")
    d = bounds.dimension
    steps = bounds.steps
    zero = (0,) * d
    units = tuple(tuple(int(i == j) for i in range(d)) for j in range(d))
    orders = sorted(
        (key for key in itertools.product(range(4), repeat=d) if sum(key) <= 3), key=sum
    )
    spatial = tuple(range(-d, 0))
    cell_low, cell_high = bounds.cell_bounds
    magnitude = torch.maximum(cell_low.abs(), cell_high.abs()).mean(spatial)
    first_mag = bounds.partial_magnitude.mean(spatial)
    second_mag = bounds.second_magnitude.mean(spatial).reshape(bounds.channels, d, d)
    value_mid = (bounds.node_lower + bounds.node_upper) / 2
    value_rad = (bounds.node_upper - bounds.node_lower) / 2
    dl, du = bounds.partial_nodes
    slope_mid, slope_rad = (dl + du) / 2, (du - dl) / 2
    shape = (-1,) + (1,) * d
    gains, kernels, nodes = {}, {}, {}

    def add(left, right):
        return tuple(a + b for a, b in zip(left, right))

    def gain(key):
        if key not in gains:
            gains[key] = kernel_gain(
                module, bounds.lengths, key, bounds.node_lower.dtype
            )
        return gains[key]

    def kernel(key):
        if key not in kernels:
            kernels[key] = sampled_kernel(module, bounds, key)
        return kernels[key]

    def apply(values, mid, rad):
        center = convolution(values, mid[None])[0]
        radius = convolution(values.abs(), rad[None])[0].clamp(min=0)
        return center, radius

    for key in orders:
        current = kernel(key)
        coefficient = current
        for axis, step in enumerate(steps):
            coefficient = coefficient - step / 2 * kernel(add(key, units[axis]))
        center, radius = apply(coefficient, value_mid, value_rad)
        for axis, step in enumerate(steps):
            correction, uncertainty = apply(
                step / 2 * current, slope_mid[:, axis], slope_rad[:, axis]
            )
            center = center + correction
            radius = radius + uncertainty
        remainder = magnitude.new_zeros(center.shape[0])
        for left in range(d):
            for right in range(left, d):
                mixed = gain(add(add(key, units[left]), units[right])) @ magnitude
                mixed = mixed + gain(add(key, units[left])) @ first_mag[:, right]
                mixed = mixed + gain(add(key, units[right])) @ first_mag[:, left]
                mixed = mixed + gain(key) @ second_mag[:, left, right]
                factor = steps[left] * steps[right] / (6 if left == right else 4)
                remainder = remainder + factor * mixed
        radius = radius + remainder.reshape(shape)
        nodes[key] = (center - radius, center + radius)
        del kernels[key]

    # The paper's target-cell recursion: third derivatives, then Hessian, then gradient.
    cells = {}
    for key in reversed(orders[1:]):
        radius = torch.zeros_like(nodes[key][0])
        for axis, step in enumerate(steps):
            following = add(key, units[axis])
            if sum(key) == 3:
                bound = (gain(following) @ magnitude).reshape(shape)
            else:
                bound = torch.maximum(
                    cells[following][0].abs(), cells[following][1].abs()
                )
            radius = radius + step * bound
        cells[key] = (nodes[key][0] - radius, nodes[key][1] + radius)
    low, high = nodes[zero]
    bias = getattr(module, "bias", None)
    if bias is not None:
        bias = bias.detach().to(low).reshape(shape)
        low, high = low + bias, high + bias
    partials = [
        torch.stack([cells[key][side] for key in units], dim=1) for side in range(2)
    ]
    partial_nodes = [
        torch.stack([nodes[key][side] for key in units], dim=1) for side in range(2)
    ]
    seconds = [
        torch.stack(
            [
                torch.stack([cells[add(a, b)][side] for b in units], dim=1)
                for a in units
            ],
            dim=1,
        )
        for side in range(2)
    ]
    if d == 1:
        seconds = [value[:, 0, 0] for value in seconds]
    return ibp.Bounds(
        bounds.axes, bounds.lengths, low, high, *partials, *partial_nodes, *seconds
    )

def convolution_integral(module, bounds):
    from lean_no.integration.spatial import slice_conv
    from lean_no.propagation import linear as ibp

    count = module.out_channels
    fields = (
        "node_lower",
        "node_upper",
        "partial_lower",
        "partial_upper",
        "partial_node_lower",
        "partial_node_upper",
        "second_lower",
        "second_upper",
    )
    output = {}
    for channel in range(count):
        current = slice_conv(module, channel, channel + 1)
        part = conv_channel(current, bounds)
        for name in fields:
            value = getattr(part, name)
            if name not in output:
                output[name] = value.new_empty((count, *value.shape[1:]))
            output[name][channel : channel + 1] = value
        del part
    return ibp.Bounds(bounds.axes, bounds.lengths, **output)
