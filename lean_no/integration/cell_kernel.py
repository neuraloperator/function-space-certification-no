import itertools
import math
from dataclasses import dataclass

import torch

from lean_no.integration.rules import interval_rule

@dataclass(frozen=True)
class CellKernel:
    # [independent input channels, source cells, source frequencies].
    coefficients: torch.Tensor
    frequencies: torch.Tensor
    centers: torch.Tensor
    widths: torch.Tensor

def cell_kernel_integral(source, method, options=None):
    options = options or {}
    unknown = set(options) - {"order", "stages", "progress"}
    if unknown:
        raise ValueError(
            "general linearization integration options are order and stages"
        )
    order = options.get("order", 4)
    nodes, weights, distance = interval_rule(method, order)
    indices = list(itertools.product(range(len(nodes)), repeat=len(source.widths)))
    offsets = source.centers.new_tensor(
        [[nodes[i] - 0.5 for i in item] for item in indices]
    )
    weights = source.centers.new_tensor(
        [math.prod(weights[i] for i in item) for item in indices]
    )
    points = source.centers[:, None] + offsets[None] * source.widths
    frequencies = source.frequencies.to(source.centers)
    phase = torch.exp(-2j * math.pi * torch.einsum("pqd,kd->pqk", points, frequencies))
    coefficients = source.coefficients
    values = torch.einsum("ipk,pqk->ipq", coefficients, phase).real
    quadrature = (values.abs() * weights).sum(-1)
    speed = 2 * math.pi * frequencies.abs()
    lipschitz = torch.einsum("ipk,kd->ipd", coefficients.abs(), speed)
    center_phase = torch.exp(-2j * math.pi * (source.centers @ frequencies.T))
    center = (coefficients * center_phase).sum(-1).real
    sign = center.abs() > (lipschitz * source.widths / 2).sum(-1)
    first = distance * (lipschitz * source.widths).sum(-1)
    degree = (
        2 * order
        if method == "gaussian"
        else order
        if method == "clenshaw_curtis"
        else 1
    )
    cap = torch.einsum(
        "ipk,kd->ipd", coefficients.abs(), torch.exp(2 * source.widths * speed)
    )
    cauchy = (2 / 0.75 * 0.25**degree) * cap.sum(-1)
    cauchy = torch.nan_to_num(cauchy, nan=torch.inf, posinf=torch.inf)
    error = torch.where(sign, torch.minimum(first, cauchy), first)
    norm = (quadrature + error).mean(-1)
    return norm, dict(
        sign_certified=int(sign.sum()), cauchy_used=int((sign & (cauchy < first)).sum())
    )
