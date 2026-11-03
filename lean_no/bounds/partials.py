from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class PartialBounds:
    node_lower: torch.Tensor
    node_upper: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor

def field_parts(field):
    dimension = field.dimension
    zero = (0,) * dimension
    lower, upper = field.cell_bounds
    parts = {zero: PartialBounds(field.node_lower, field.node_upper, lower, upper)}
    units = tuple(
        tuple(int(index == axis) for index in range(dimension))
        for axis in range(dimension)
    )
    nodes = field.partial_nodes
    for axis, key in enumerate(units):
        parts[key] = PartialBounds(
            nodes[0][:, axis],
            nodes[1][:, axis],
            field.partial_lower[:, axis],
            field.partial_upper[:, axis],
        )
    for left in range(dimension):
        for right in range(left, dimension):
            key = tuple(a + b for a, b in zip(units[left], units[right]))
            low = (
                field.second_lower
                if dimension == 1
                else field.second_lower[:, left, right]
            )
            high = (
                field.second_upper
                if dimension == 1
                else field.second_upper[:, left, right]
            )
            parts[key] = PartialBounds(low, high, low, high)
    return parts
