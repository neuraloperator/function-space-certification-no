import math

import torch

from lean_no.bounds.positive import positive_map

def spectral_radii(radius, data, remainder=None, chunk_size=None):
    degrees = data["degrees"]
    factors = 2 * torch.arange(degrees, device=radius.device, dtype=radius.dtype) + 1
    if data.get("projection", False):
        inherited = degrees * radius
        fresh = (
            torch.zeros_like(inherited)
            if remainder is None
            else (remainder * factors).sum(-1) / (4 * math.pi)
        )
        return inherited, fresh
    weight = data["weight"]
    step = max(degrees, weight.shape[0]) if chunk_size is None else chunk_size
    square = radius.new_zeros(weight.shape[:2])
    fresh = radius.new_zeros(weight.shape[0])
    for start in range(0, degrees, step):
        stop = min(degrees, start + step)
        for channel in range(0, weight.shape[0], step):
            end = min(weight.shape[0], channel + step)
            block = weight[channel:end, :, start:stop].abs().to(radius)
            square[channel:end] += (block.square() * factors[start:stop]).sum(-1)
            if remainder is not None:
                terms = torch.where(
                    block == 0, 0.0, block * remainder[None, :, start:stop]
                )
                fresh[channel:end] += (terms * factors[start:stop]).sum((1, 2)) / (
                    4 * math.pi
                )
    return positive_map(radius, square.sqrt().T), fresh
