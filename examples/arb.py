import torch

from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.inputs import InputSet
from lean_no.methods import Method
from lean_no.models import load_fno

model = load_fno("models/poisson_1d/robustness.pt", device="cpu")

def forcing(x):
    return 1.0

points = torch.tensor([[0.125]], dtype=torch.float64)
fixed = certify(
    model=model,
    grid=FourierGrid((64,), (1.0,)),
    inputs=forcing,
    method=Method("ibp", "arb"),
)
print("Fixed output:", fixed.query(points=points))
for propagation in ("ibp", "crown"):
    certificate = certify(
        model=model,
        grid=FourierGrid((64,), (1.0,)),
        inputs=InputSet(forcing, epsilon=0.001, lipschitz=[0.0]),
        method=Method(propagation, "arb"),
    )
    print(propagation.upper(), "interval at x=0.125:", certificate.query(points=points))
