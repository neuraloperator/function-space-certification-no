import json
import math
import time

import torch

from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.methods import Method
from lean_no.models import load_fno
from lean_no.paths import repo_path
from lean_no.quantities import Residual
from lean_no.quantities.residual import residual_bound
from lean_no.quantities.solution import solution_bound
from lean_no.runtime import digest

def forcing(x):
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * x)

def main():
    torch.set_num_threads(1)
    rows = []
    for problem in ("poisson", "helmholtz"):
        checkpoint = f"models/{problem}_1d/solution.pt"
        model = load_fno(path=checkpoint, device="cpu").eval().requires_grad_(False)
        started = time.perf_counter()
        field = certify(
            model,
            FourierGrid((4096,)),
            forcing,
            derivatives=2,
            integration_options={"tolerance": "1e-14", "digits": 30},
            method=Method("ibp", "arb"),
        )
        quantity = (
            Residual("poisson")
            if problem == "poisson"
            else Residual("helmholtz", coefficient=math.sqrt((8 * math.pi) ** 2 - 0.02))
        )
        residual = residual_bound(field, quantity)
        norms = residual.bounds
        solution = solution_bound(residual).solution_error_upper
        if not all(math.isfinite(value) for value in (norms.residual_linf, solution)):
            raise ArithmeticError("nonfinite Arb PDE check")
        rows.append(
            dict(
                problem=problem,
                checkpoint=checkpoint,
                checkpoint_sha256=digest(repo_path(checkpoint)),
                output_cells=4096,
                tolerance="1e-14",
                digits=30,
                residual_linf_upper=norms.residual_linf,
                solution_error_upper=solution,
                seconds=time.perf_counter() - started,
            )
        )
    print(
        json.dumps(
            dict(roundoff_enclosed=False, device="cpu", threads=1, rows=rows), indent=2
        )
    )

if __name__ == "__main__":
    main()
