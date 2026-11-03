from __future__ import annotations

import math

import torch
import torch.nn as nn
from neuralop.layers.skip_connections import Flattened1dConv, SoftGating

from lean_no.bounds.activations import (
    crown_relu_rule,
    crown_tanh_rule,
    gelu_bounds,
    gelu_lipschitz,
    is_gelu,
    is_identity,
    is_relu,
    is_tanh,
)
from lean_no.bounds.interval import interval_linear
from lean_no.spectral import real_inverse

__all__ = [
    "affine_bias",
    "affine_weight",
    "convolution",
    "crown_relu_rule",
    "crown_tanh_rule",
    "gelu_bounds",
    "gelu_lipschitz",
    "interval_linear",
    "is_gelu",
    "is_identity",
    "is_relu",
    "is_tanh",
    "validate_fno",
]


def validate_fno(model: nn.Module, normalization=False) -> None:
    blocks = model.fno_blocks
    unsupported = (
        model.domain_padding is not None,
        blocks.preactivation,
        blocks.norm is not None and not normalization,
        blocks.stabilizer is not None,
        blocks.complex_data,
        blocks.resolution_scaling_factor is not None,
    )
    if any(unsupported):
        raise NotImplementedError("unsupported FNO configuration")

def affine_weight(module: nn.Module) -> torch.Tensor:
    from lean_no.graph import Operator

    if isinstance(module, Operator):
        return module.weight
    if isinstance(module, nn.Conv1d):
        return module.weight[..., 0]
    if isinstance(module, Flattened1dConv):
        return affine_weight(module.conv)
    if isinstance(module, SoftGating):
        return torch.diag(module.weight.reshape(-1))
    return module.weight

def affine_bias(
    module: nn.Module,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    source = module.conv if isinstance(module, Flattened1dConv) else module
    bias = getattr(source, "bias", None)
    if bias is None:
        return None
    return bias.detach().to(reference).reshape(-1)

def convolution(
    kernel: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    dimension = values.ndim - 2
    dimensions = tuple(range(-dimension, 0))
    kernel_frequency = torch.fft.rfftn(kernel, dim=dimensions)
    values_frequency = torch.fft.rfftn(values, dim=dimensions)
    output_frequency = torch.einsum(
        "oi...,si...->so...",
        kernel_frequency,
        values_frequency,
    )
    scale = math.prod(values.shape[-dimension:])
    return real_inverse(output_frequency, values.shape[-dimension:]) / scale
