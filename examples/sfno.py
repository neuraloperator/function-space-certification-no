import math

import torch

from lean_no.adapters.sfno import SFNOModel
from lean_no.certify import certify
from lean_no.graph import Graph, Operator
from lean_no.methods import Method
from lean_no.propagation.spherical import linear_field
from lean_no.quantities.spherical import regional_maximum
from lean_no.spectral.spherical import sfno_grid

grid = sfno_grid(q=16, dtype=torch.float64, rule="clenshaw-curtis", nlon=32)

def weather(latitude, longitude):
    return 0.2 / math.sqrt(4 * math.pi) + 0.01 * math.sqrt(
        3 / (4 * math.pi)
    ) * torch.cos(latitude)

spectral = Operator(
    name="spectral",
    kind="spectral",
    parents=(0,),
    data=dict(weight=torch.ones(1, 1, 2, dtype=torch.float64), degrees=2, orders=2),
)
model = SFNOModel(Graph((spectral,), {"basis": "spherical"}), grid, grid, ("t2m",))
forecast = certify(
    model=model, grid=grid, inputs=weather, method=Method("ibp", "clenshaw_curtis")
)
# Illustrative normalization: T[K] = 10 * model_output + 280.
temperature = linear_field(
    source=forecast.enclosure,
    weight=torch.tensor([[10.0]], dtype=torch.float64),
    bias=torch.tensor([280 - 273.15], dtype=torch.float64),
)
region, _ = regional_maximum(field=temperature, box=(32.5, 42.0, -124.5, -114.0))
print(
    "Continuous regional maximum [C]:",
    region["maximum_lower"],
    region["maximum_upper"],
)
