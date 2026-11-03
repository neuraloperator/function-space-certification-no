import math

import torch

def positive_map(bounds, weight):
    weight = weight.to(bounds)
    result = bounds @ weight
    if torch.isnan(result).any():
        finite = torch.where(torch.isfinite(bounds), bounds, 0.0)
        infinite = torch.isinf(bounds).to(weight) @ weight > 0
        result = torch.where(infinite, math.inf, finite @ weight)
    return result
