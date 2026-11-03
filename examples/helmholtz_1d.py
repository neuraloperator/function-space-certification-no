from examples.common import make_model
from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.inputs import InputSet
from lean_no.methods import Method
from lean_no.quantities import Residual
from lean_no.quantities.residual import residual_bound
from lean_no.quantities.solution import solution_bound

kappa = 1.0  # Helmholtz wave number.
model = make_model(dimension=1)
grid = FourierGrid(resolution=(64,))

def forcing(x):
    return 1.0

field = certify(
    model=model,
    grid=grid,
    inputs=forcing,
    moment_order=16,
    derivatives=2,
    method=Method("ibp", "gaussian"),
)
residual = residual_bound(field, Residual(pde="helmholtz", coefficient=kappa))
robustness = certify(
    model=model,
    grid=grid,
    inputs=InputSet(forcing, epsilon=0.001, lipschitz=[0.0]),
    method=Method("crown", "rectangle"),
)

norms = residual.bounds
print("Residual Linf upper:", norms.residual_linf)
print("Solution Linf upper:", solution_bound(residual).solution_error_upper)
print("Robustness cell-range width:", float(robustness.cell_range_width.amax()))
