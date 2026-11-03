from lean_no.certify import certify
from lean_no.domain import DiscreteGrid
from lean_no.inputs import InputSet
from lean_no.methods import Method
from lean_no.models import load_fno

model = load_fno("models/poisson_1d/robustness.pt", device="cpu")

def forcing(x):
    return 1.0

for propagation in ("ibp", "crown"):
    certificate = certify(
        model=model,
        grid=DiscreteGrid(resolution=(64,)),
        inputs=InputSet(forcing, epsilon=0.001),
        method=Method(propagation, None),
    )
    print(propagation.upper(), "node width:", float(certificate.node_width.amax()))
