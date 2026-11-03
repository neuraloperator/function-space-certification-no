from __future__ import annotations

import torch

def tanh_value(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

def tanh_first(x: torch.Tensor) -> torch.Tensor:
    return 1 - torch.tanh(x).square()

def pick_lines(
    slopes: torch.Tensor,
    low_bias: torch.Tensor,
    high_bias: torch.Tensor,
    midpoint: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    low_at = slopes * midpoint[..., None] + low_bias
    high_at = slopes * midpoint[..., None] + high_bias
    low_index = low_at.argmax(dim=-1, keepdim=True)
    high_index = high_at.argmin(dim=-1, keepdim=True)
    low_slope = slopes.gather(-1, low_index).squeeze(-1)
    high_slope = slopes.gather(-1, high_index).squeeze(-1)
    low = low_bias.gather(-1, low_index).squeeze(-1)
    high = high_bias.gather(-1, high_index).squeeze(-1)
    return low_slope, low, high_slope, high

def tanh_rule(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = upper - lower
    safe = width.clamp_min(torch.finfo(lower.dtype).eps)
    midpoint = (lower + upper) / 2
    points = torch.stack(
        (lower, upper, midpoint, 0.9 * upper + 0.1 * lower, 0.2 * upper + 0.8 * lower),
        dim=-1,
    )
    secant = (tanh_value(upper) - tanh_value(lower)) / safe
    slopes = torch.cat(
        (secant[..., None], tanh_first(points), torch.zeros_like(secant[..., None])),
        dim=-1,
    )

    low = lower[..., None]
    high = upper[..., None]
    positive = slopes > 0
    roots = torch.acosh(slopes.clamp_min(torch.finfo(lower.dtype).tiny).rsqrt())
    candidates = torch.stack(
        (
            low.expand_as(slopes),
            high.expand_as(slopes),
            roots,
            -roots,
        )
    )
    endpoints = torch.ones_like(slopes, dtype=torch.bool)
    root_valid = positive & (slopes <= 1)
    valid = torch.stack((endpoints, endpoints, root_valid, root_valid))
    valid = valid & (candidates >= low) & (candidates <= high)
    residual = tanh_value(candidates) - slopes.unsqueeze(0) * candidates
    low_bias = residual.masked_fill(~valid, torch.inf).amin(dim=0)
    high_bias = residual.masked_fill(~valid, -torch.inf).amax(dim=0)
    result = pick_lines(slopes, low_bias, high_bias, midpoint)

    exact = width == 0
    exact_slope = tanh_first(lower)
    exact_bias = tanh_value(lower) - exact_slope * lower
    return tuple(
        torch.where(exact, exact_slope if index % 2 == 0 else exact_bias, value)
        for index, value in enumerate(result)
    )
