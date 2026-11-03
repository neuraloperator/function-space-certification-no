import math

import torch

def stable_rms(value, dim=-1):
    scale = value.abs().amax(dim, keepdim=True)
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (scale * (value / divisor).square().mean(dim, keepdim=True).sqrt()).squeeze(
        dim
    )

def denominator_lower(base, error, epsilon):
    floor = base.new_tensor(math.sqrt(epsilon))
    return torch.maximum(base - error, floor)

def instance_radius(radius, centered_magnitude, base, weight, epsilon):
    lower = denominator_lower(base, radius, epsilon)
    error = radius * weight.abs() * (2 / lower + centered_magnitude / (base * lower))
    return torch.where(weight == 0, 0.0, error)

def normalization_affine_bounds(lower, upper, weight, bias, epsilon, groups):
    if lower.shape != upper.shape or lower.ndim < 2:
        raise ValueError("normalization bounds require matching channel/grid shapes")
    channels = len(lower)
    if type(groups) is not int or groups < 1 or channels % groups:
        raise ValueError("normalization groups must divide the channel count")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization epsilon must be positive and finite")
    if (
        not torch.isfinite(lower).all()
        or not torch.isfinite(upper).all()
        or (lower > upper).any()
    ):
        raise ValueError("normalization input bounds must be finite and ordered")
    weight, bias = weight.to(lower), bias.to(lower)
    if weight.shape != (channels,) or bias.shape != (channels,):
        raise ValueError("normalization affine parameters must match channels")
    lo, hi = lower.reshape(groups, -1), upper.reshape(groups, -1)
    center, radius = (lo + hi) / 2, (hi - lo) / 2
    centered = center - center.mean(-1, keepdim=True)
    deviation = stable_rms(centered)
    error = stable_rms(radius)
    denominator_lo = ((deviation - error).clamp_min(0).square() + epsilon).sqrt()
    denominator_hi = ((deviation + error).square() + epsilon).sqrt()
    inverse_lo, inverse_hi = denominator_hi.reciprocal(), denominator_lo.reciprocal()
    inverse = (inverse_lo + inverse_hi) / 2
    variation = (inverse_hi - inverse_lo) / 2
    count = lo.shape[-1]
    centered_lo = lo - (hi.sum(-1, keepdim=True) - hi + lo) / count
    centered_hi = hi - (lo.sum(-1, keepdim=True) - lo + hi) / count
    magnitude = torch.maximum(centered_lo.abs(), centered_hi.abs())
    channel_shape = (channels, *((1,) * (lower.ndim - 1)))
    scale = weight * inverse.repeat_interleave(channels // groups)
    residual = magnitude * variation[:, None]
    residual = residual.reshape_as(lower) * weight.abs().reshape(channel_shape)
    return scale, bias, residual

def normalization_adjoint(coefficient, scale, groups):
    channel_shape = (1, len(scale), *((1,) * (coefficient.ndim - 2)))
    result = coefficient * scale.reshape(channel_shape)
    grouped = result.reshape(len(result), groups, -1)
    return (grouped - grouped.mean(-1, keepdim=True)).reshape_as(result)

def normalization_parameters(module, channels, reference):
    from lean_no.graph import Operator

    if isinstance(module, Operator):
        data = module.data
        return (
            data["weight"].to(reference),
            data["bias"].to(reference),
            data["epsilon"],
            data["groups"],
        )
    from neuralop.layers.normalization_layers import InstanceNorm
    from torch import nn

    if isinstance(module, InstanceNorm):
        data = module.kwargs
        if not data.get("use_input_stats", True) or any(
            data.get(k) is not None for k in ("running_mean", "running_var")
        ):
            raise NotImplementedError("normalization requires native input statistics")
        weight, bias = data.get("weight"), data.get("bias")
        epsilon, groups = data.get("eps", 1e-5), channels
    elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        if module.track_running_stats or module.num_features != channels:
            raise NotImplementedError(
                "InstanceNorm requires input statistics and matching channels"
            )
        weight, bias, epsilon, groups = module.weight, module.bias, module.eps, channels
    elif isinstance(module, nn.GroupNorm):
        if module.num_channels != channels:
            raise ValueError("GroupNorm channel count does not match its input")
        weight, bias, epsilon, groups = (
            module.weight,
            module.bias,
            module.eps,
            module.num_groups,
        )
    else:
        raise NotImplementedError(
            "FNO normalization supports InstanceNorm or GroupNorm"
        )
    weight = reference.new_ones(channels) if weight is None else weight.to(reference)
    bias = reference.new_zeros(channels) if bias is None else bias.to(reference)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization epsilon must be positive and finite")
    return weight, bias, epsilon, groups

def normalization_rule(source, module):
    weight, bias, epsilon, groups = normalization_parameters(
        module, source.channels, source.node_lower
    )
    lo, hi = (
        source.node_lower.reshape(groups, -1),
        source.node_upper.reshape(groups, -1),
    )
    center, radius = (lo + hi) / 2, (hi - lo) / 2
    deviation = stable_rms(center - center.mean(-1, keepdim=True))
    error = stable_rms(radius)
    inv_hi = ((deviation - error).clamp_min(0).square() + epsilon).rsqrt()
    inv_lo = ((deviation + error).square() + epsilon).rsqrt()
    inverse, variation = (inv_lo + inv_hi) / 2, (inv_hi - inv_lo) / 2
    expand = lambda x: x.repeat_interleave(source.channels // groups)
    mean_center, mean_radius = expand(center.mean(-1)), expand(radius.mean(-1))
    scale = weight * expand(inverse)
    cell_lo, cell_hi = source.cell_bounds
    shape = (source.channels, *((1,) * source.dimension))
    magnitude = (
        torch.maximum(
            (cell_lo - (mean_center + mean_radius).reshape(shape)).abs(),
            (cell_hi - (mean_center - mean_radius).reshape(shape)).abs(),
        )
        .flatten(1)
        .amax(-1)
    )
    scale_radius = weight.abs() * expand(variation)
    residual = scale.abs() * mean_radius + scale_radius * magnitude
    shift = bias - scale * mean_center
    return scale, shift - residual, shift + residual, scale_radius
