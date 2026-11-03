import json
import math

import torch

from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.inputs import InputSet
from lean_no.methods import Method
from lean_no.models import load_fno
from lean_no.paths import repo_path
from lean_no.runtime import Timer, digest, write_json
from scripts._shared.attacks import evaluate_witness, run_mc
from scripts._shared.inputs import WitnessSearch, input_limits

checkpoint_path = repo_path("models/poisson_1d/robustness.pt")
summary_path = repo_path("results/poisson_1d_robustness/summary.json")
case = WitnessSearch(length=1.0, eps=0.001, full_resolution=2**20, grid_resolution=64)

def forcing(x):
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * x)

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identity = digest(checkpoint_path)
    model = load_fno(checkpoint_path, dtype=torch.float64).requires_grad_(False)
    device = next(model.parameters()).device
    mc = run_mc(case, model)
    witnesses = evaluate_witness(case, model, mc)
    lipschitz, curvature = input_limits(case)
    grid = FourierGrid((case.full_resolution,), (case.length,), torch.float64)
    rows = []
    for propagation, chunk, cells in (("ibp", 64, None), ("crown", 8, 131072)):
        with Timer(device) as timing:
            certificate = certify(
                model,
                grid,
                InputSet(
                    forcing, epsilon=case.eps, lipschitz=lipschitz, curvature=curvature
                ),
                chunk_size=chunk,
                output_cells=cells,
                method=Method(propagation, "gaussian"),
            )
        width = float(certificate.cell_range_width.amax())
        lower = witnesses["continuum"]["lower_bound"]
        if not math.isfinite(width) or width < 0 or lower <= 0:
            raise ArithmeticError(
                "finite bounds and positive witness separation required"
            )
        rows.append(
            dict(
                method=propagation,
                width_upper=width,
                witness_lower=lower,
                gap=width / lower,
                seconds=timing.seconds,
                peak_gib=timing.peak_bytes / 2**30,
            )
        )
        del certificate
    if rows[-1]["gap"] >= 10:
        raise ArithmeticError("reported CROWN robustness gap must be below 10")
    if digest(checkpoint_path) != identity:
        raise RuntimeError("checkpoint changed during certification; rerun")
    summary = dict(
        checkpoint=str(checkpoint_path.relative_to(repo_path("."))),
        checkpoint_sha256=identity,
        epsilon=case.eps,
        resolution=grid.resolution,
        lipschitz=lipschitz,
        curvature=curvature,
        roundoff_enclosed=False,
        witnesses=witnesses,
        bounds=rows,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary

if __name__ == "__main__":
    main()
