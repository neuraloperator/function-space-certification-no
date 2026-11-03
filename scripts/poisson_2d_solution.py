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

checkpoint_path = repo_path("models/poisson_2d/solution.pt")
summary_path = repo_path("results/poisson_2d_solution_error/summary.json")
gaussian_order = 16

def forcing(x, y):
    return torch.sin(math.pi * x) * torch.sin(math.pi * y) + 0.5 * torch.sin(
        2 * math.pi * x
    ) * torch.sin(2 * math.pi * y)

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = pick_device()
    checkpoint_hash = digest(checkpoint_path)
    model = load_fno(checkpoint_path, device=device).requires_grad_(False)
    with Timer(device) as timing:
        field = certify(
            model=model,
            grid=FourierGrid((16384, 16384), (1.0, 1.0), torch.float64),
            inputs=forcing,
            moment_order=gaussian_order,
            moment_batch=8192,
            chunk_size=262144,
            progress=lambda row: print(json.dumps(row), flush=True),
            derivatives=2,
            method=Method("ibp", "gaussian", strategy="taylor_cell"),
        )
        certificate = residual_bound(
            field,
            Residual("poisson"),
            chunk_size=262144,
            progress=lambda row: print(json.dumps(row), flush=True),
        )
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
