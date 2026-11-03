from examples.common import make_model
from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.inputs import InputSet
from lean_no.methods import Method
from lean_no.quantities import Residual
from lean_no.quantities.residual import residual_bound
from lean_no.quantities.solution import solution_bound

model = make_model(dimension=2)
grid = FourierGrid(resolution=(8, 8), lengths=(1.0, 1.0))

def forcing(x, y):
    return 1.0

field = certify(
    model=model,
    grid=grid,
    inputs=forcing,
    moment_order=16,
    derivatives=2,
    method=Method("ibp", "gaussian", strategy="taylor_cell"),
)
residual = residual_bound(field, Residual(pde="poisson"))
robustness = certify(
    model=model,
    grid=grid,
    inputs=InputSet(forcing, epsilon=0.001),
    method=Method("ibp", None, strategy="linearized"),
)
norms = residual.bounds
print("Residual Linf:", norms.residual_linf_interval)
print("Solution Linf upper:", solution_bound(residual).solution_error_upper)
print("Robustness pointwise width:", float(robustness.pointwise_width.amax()))
