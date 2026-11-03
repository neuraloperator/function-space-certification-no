import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import eval_legendre, roots_jacobi

@dataclass(frozen=True)
class SphereGrid:

    latitude: torch.Tensor
    quadrature: torch.Tensor
    nlon: int
    kind: str | None = None
    gauss_nodes: int | None = None  # nodes per composite latitude cell, if applicable

    def validate(self):
        if (
            type(self.latitude) is not torch.Tensor
            or self.latitude.ndim != 1
            or len(self.latitude) < 2
            or type(self.quadrature) is not torch.Tensor
            or self.quadrature.shape != self.latitude.shape
            or self.latitude.dtype not in (torch.float32, torch.float64)
            or self.quadrature.dtype != self.latitude.dtype
            or self.quadrature.device != self.latitude.device
            or not torch.isfinite(self.latitude).all()
            or not torch.isfinite(self.quadrature).all()
            or (self.quadrature <= 0).any()
            or (self.latitude < 0).any()
            or (self.latitude > math.pi).any()
            or (self.latitude[1:] <= self.latitude[:-1]).any()
            or type(self.nlon) is not int
            or self.nlon < 3
        ):
            raise ValueError(
                "sphere grids need ordered colatitudes and positive quadrature"
            )

    @property
    def shape(self):
        return (len(self.latitude), self.nlon)

    @property
    def edges(self):
        return torch.cat(
            (
                self.latitude.new_zeros(1),
                (self.latitude[:-1] + self.latitude[1:]) / 2,
                self.latitude.new_full((1,), math.pi),
            )
        )

    @property
    def longitude(self):
        return torch.arange(
            self.nlon, device=self.latitude.device, dtype=self.latitude.dtype
        ) * (2 * math.pi / self.nlon)

    @property
    def points(self):
        return torch.stack(
            torch.meshgrid(self.latitude, self.longitude, indexing="ij"), -1
        ).reshape(-1, 2)

    @property
    def latitude_area(self):
        edges = self.edges
        return edges[:-1].cos() - edges[1:].cos()

    @property
    def areas(self):
        return (
            self.latitude_area[:, None].expand(self.shape) * (2 * math.pi / self.nlon)
        ).flatten()

    @property
    def radius(self):
        edges = self.edges
        radial = torch.maximum(self.latitude - edges[:-1], edges[1:] - self.latitude)
        radial = radial + self.latitude.sin().abs() * (math.pi / self.nlon)
        return radial.clamp(max=math.pi)[:, None].expand(self.shape).flatten()

    @property
    def cover_radius(self):
        edges = self.edges
        cosine = math.cos(math.pi / self.nlon)
        left = self.latitude.cos() * edges[:-1].cos()
        left += self.latitude.sin() * edges[:-1].sin() * cosine
        right = self.latitude.cos() * edges[1:].cos()
        right += self.latitude.sin() * edges[1:].sin() * cosine
        return (
            torch.minimum(left, right)
            .clamp(-1, 1)
            .acos()[:, None]
            .expand(self.shape)
            .flatten()
        )

    def indices(self, points):
        if (
            type(points) is not torch.Tensor
            or points.ndim != 2
            or points.shape[1] != 2
            or points.dtype != self.latitude.dtype
            or points.device != self.latitude.device
            or not torch.isfinite(points).all()
            or (points[:, 0] < 0).any()
            or (points[:, 0] > math.pi).any()
        ):
            raise ValueError(
                "queries need finite colatitude/longitude on the grid dtype/device"
            )
        latitude = torch.bucketize(points[:, 0].contiguous(), self.edges[1:-1])
        longitude = torch.floor(
            points[:, 1].remainder(2 * math.pi) / (2 * math.pi / self.nlon) + 0.5
        )
        return latitude * self.nlon + longitude.long().remainder(self.nlon)

def sphere_grid(shape, kind="legendre-gauss", dtype=torch.float64, device=None):
    if (
        type(shape) is not tuple
        or len(shape) != 2
        or any(type(size) is not int for size in shape)
        or shape[0] < 2
        or shape[1] < 3
        or dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("sphere shape must contain latitude >=2 and longitude >=3")
    nlat, nlon = shape
    if kind == "legendre-gauss":
        nodes, weights = np.polynomial.legendre.leggauss(nlat)
        latitude = np.arccos(nodes[::-1].copy())
        weights = weights[::-1].copy()
    elif kind == "lobatto":
        middle = roots_jacobi(nlat - 2, 1, 1)[0] if nlat > 2 else np.empty(0)
        nodes = np.concatenate(([-1.0], middle, [1.0]))
        weights = 2 / (nlat * (nlat - 1) * eval_legendre(nlat - 1, nodes) ** 2)
        latitude, weights = np.arccos(nodes[::-1].copy()), weights[::-1].copy()
    elif kind == "equiangular":
        degree = nlat - 1
        latitude = np.linspace(0, math.pi, nlat)
        weights = np.zeros(nlat)
        values = np.ones(nlat - 2)
        for mode in range(1, (degree + 1) // 2):
            values -= 2 * np.cos(2 * mode * latitude[1:-1]) / (4 * mode**2 - 1)
        if degree % 2 == 0:
            values -= np.cos(degree * latitude[1:-1]) / (degree**2 - 1)
            weights[0] = weights[-1] = 1 / (degree**2 - 1)
        else:
            weights[0] = weights[-1] = 1 / degree**2
        weights[1:-1] = 2 * values / degree
    else:
        raise ValueError("unknown spherical quadrature")
    result = SphereGrid(
        torch.tensor(latitude, dtype=dtype, device=device),
        torch.tensor(weights, dtype=dtype, device=device),
        nlon,
        kind,
    )
    result.validate()
    return result
