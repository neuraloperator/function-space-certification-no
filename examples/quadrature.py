from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.inputs import InputSet
from lean_no.methods import Method
from lean_no.models import load_fno

model = load_fno("models/poisson_1d/robustness.pt", device="cpu")
grid = FourierGrid(resolution=(64,))

def forcing(x):
    return 1.0

fixed = certify(
    model=model, grid=grid, inputs=forcing, method=Method("ibp", "rectangle")
)
print("Fixed output cell-range width:", float(fixed.cell_range_width.amax()))
for propagation in ("ibp", "crown"):
    certificate = certify(
        model=model,
        grid=grid,
        inputs=InputSet(forcing, epsilon=0.001, lipschitz=[0.0]),
        method=Method(propagation, "rectangle"),
    )
    print(
        propagation.upper(),
        "cell-range width:",
        float(certificate.cell_range_width.amax()),
    )
