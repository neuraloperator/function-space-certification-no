import math

import torch

from examples.common import make_model
from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.inputs import InputSet
from lean_no.methods import Method
from lean_no.quantities import Residual
from lean_no.quantities.residual import residual_bound
from lean_no.quantities.solution import solution_bound

model = make_model(dimension=2, channels=2)

def initial(x):
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * x)

field = certify(
    model=model,
    grid=FourierGrid(resolution=(16, 4), lengths=(1.0, 1.0)),
    inputs=initial,
    moment_order=16,
    derivatives=2,
    method=Method("ibp", "gaussian", strategy="taylor_cell"),
)
residual = residual_bound(
    field, Residual(pde="burgers", coefficient=0.01), energy_modes=2
)
robustness = certify(
    model=model,
    grid=FourierGrid(resolution=(16, 16), lengths=(1.0, 1.0)),
    inputs=InputSet(initial, epsilon=0.001),
    method=Method("ibp", None, strategy="linearized"),
)
norms = residual.bounds
print("Residual L2:", norms.burgers.residual_l2)
print("Solution L2 upper:", solution_bound(residual).solution_error_upper)
print("Robustness pointwise width:", float(robustness.pointwise_width.amax()))
