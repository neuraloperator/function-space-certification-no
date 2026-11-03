import json
import math

import torch

from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.methods import Method
from lean_no.models import load_fno, pick_device
from lean_no.paths import repo_path
from lean_no.quantities import Residual
from lean_no.quantities.residual import residual_bound
from lean_no.quantities.solution import solution_bound
from lean_no.runtime import Timer, digest, write_json

resolution = 2**25
checkpoint_path = repo_path("models/poisson_1d/solution.pt")
summary_path = repo_path("results/poisson_1d_solution_error/summary.json")

def forcing(x):
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * x)

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = pick_device()
    checkpoint_hash = digest(checkpoint_path)
    model = load_fno(checkpoint_path, device=device)
    model.requires_grad_(False)
    print(f"Certifying residual and endpoints: {resolution} cells", flush=True)
    with Timer(device) as timing:
        field = certify(
            model=model,
            grid=FourierGrid((resolution,), (1.0,), torch.float64),
            inputs=forcing,
            moment_order=16,
            derivatives=2,
            method=Method("ibp", "gaussian"),
        )
        certificate = residual_bound(field, Residual("poisson"), chunk_size=65536)
    if digest(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("checkpoint changed during certification; rerun")
    summary = dict(
        **certificate.bounds.report(),
        checkpoint=str(repo_path(checkpoint_path).relative_to(repo_path("."))),
        checkpoint_sha256=checkpoint_hash,
        resolution=field.grid.resolution,
        dtype=str(field.grid.dtype),
        roundoff_enclosed=False,
        seconds=timing.seconds,
        peak_gib=timing.peak_bytes / 2**30,
    )
    summary["solution"] = solution_bound(certificate).report()
    summary["solution_l2"] = solution_bound(certificate, norm="l2").report()
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary

if __name__ == "__main__":
    main()
