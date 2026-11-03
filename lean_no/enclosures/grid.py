from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class GridBounds:
    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self):
        if (
            not isinstance(self.lower, torch.Tensor)
            or not isinstance(self.upper, torch.Tensor)
            or self.lower.shape != self.upper.shape
            or self.lower.ndim < 3
            or self.lower.shape[0] != 1
            or any(size < 1 for size in self.lower.shape)
            or self.lower.dtype not in (torch.float32, torch.float64)
            or self.upper.dtype != self.lower.dtype
            or self.upper.device != self.lower.device
            or not torch.isfinite(self.lower).all()
            or not torch.isfinite(self.upper).all()
            or (self.lower > self.upper).any()
        ):
            raise ValueError(
                "grid bounds require ordered finite [1, channels, *resolution] tensors"
            )

    @property
    def is_exact(self):
        # The tensors define the entire discrete vector, rather than samples
        # used to infer equality of two continuum functions.
        return torch.equal(self.lower, self.upper)

    @property
    def node_bounds(self):
        return self.lower[0], self.upper[0]

    @property
    def width(self):
        return (self.upper - self.lower).amax()

    @property
    def channel_width(self):
        return (self.upper - self.lower)[0].flatten(1).amax(1)

    def query(self, indices):
        dimension = self.lower.ndim - 2
        if (
            not isinstance(indices, torch.Tensor)
            or indices.ndim != 2
            or indices.shape[1] != dimension
            or indices.dtype
            not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        ):
            raise ValueError("discrete queries require integer [N, dimension] indices")
        indices = indices.to(device=self.lower.device, dtype=torch.long)
        sizes = indices.new_tensor(self.lower.shape[2:])
        if ((indices < 0) | (indices >= sizes)).any():
            raise ValueError("grid query index is out of range")
        key = (0, slice(None), *indices.T)
        return self.lower[key].T, self.upper[key].T
