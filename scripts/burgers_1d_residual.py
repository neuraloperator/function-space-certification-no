import json
import math

import torch

from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.methods import Method
from lean_no.models import pick_device, restore_fno
from lean_no.paths import repo_path
from lean_no.quantities import Residual
from lean_no.quantities.residual import residual_bound
from lean_no.runtime import Timer, digest, write_json

gaussian_order = 16
query_base = (16384, 4096)
energy_modes = 128
checkpoint_path = repo_path("models/burgers_1d/solution.pt")
summary_path = repo_path("results/burgers_1d_residual/summary.json")

def initial(x):
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * x)

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = pick_device()
    checkpoint_hash = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("solution_ansatz") != "direct_fno":
        raise ValueError("this experiment requires the direct-output fine-tuned FNO")
    if checkpoint["data"]["times"] != [j / 16 for j in range(17)]:
        raise ValueError("checkpoint physical time grid changed")
    if (
        checkpoint["data"]["viscosity"] != 0.01
        or not checkpoint["data"]["viscosity_verified"]
    ):
        raise ValueError("verified viscosity 0.01 required")
    model = restore_fno(checkpoint, device=device, dtype=torch.float64)
    with Timer(device) as timing:
        field = certify(
            model=model,
            grid=FourierGrid(query_base, (1.0, 17 / 16), torch.float64),
            inputs=initial,
            moment_order=gaussian_order,
            moment_batch=65536,
            chunk_size=65536,
            progress=lambda row: print(json.dumps(row), flush=True),
            derivatives=2,
            method=Method("ibp", "gaussian", strategy="taylor_cell"),
        )
        certificate = residual_bound(
            field,
            Residual("burgers", 0.01),
            time_horizon=1.0,
            energy_modes=energy_modes,
            chunk_size=65536,
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
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary

if __name__ == "__main__":
    main()
