import gc
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
from scripts._shared.inputs import WitnessSearch

epsilon = 0.001
resolution = 2048
dtype = torch.float64
checkpoint_path = "models/poisson_2d/robustness.pt"
summary_path = repo_path("results/poisson_2d_linearized.json")

case = WitnessSearch(
    length=1.0,
    eps=epsilon,
    full_resolution=resolution,
    grid_resolution=32,
    spatial_dimension=2,
    fourier_modes=4,
    decay_mode=None,
    mc_batch_size=64,
)

def forcing(x, y):
    return torch.sin(math.pi * x) * torch.sin(math.pi * y) + 0.5 * torch.sin(
        2 * math.pi * x
    ) * torch.sin(2 * math.pi * y)

def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    identity = digest(repo_path(checkpoint_path))
    model = load_fno(
        checkpoint_path, dtype=torch.float64, device=device
    ).requires_grad_(False)
    mc = run_mc(case, model)

    print("Certifying winning Fourier witnesses", flush=True)
    witnesses = evaluate_witness(case, model, mc)
    evidence = witnesses["continuum"]
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Linearized perturbation bounds: {resolution} x {resolution}", flush=True)
    with Timer(device) as timing:
        run = certify(
            model=model,
            grid=FourierGrid((resolution,) * 2, (1.0, 1.0), dtype),
            inputs=InputSet(forcing, epsilon=epsilon),
            chunk_size=1,
            method=Method("ibp", None, strategy="linearized"),
        )
    width = float(run.pointwise_width.amax())
    if evidence["lower_bound"] <= 0:
        raise ArithmeticError("witness enclosures do not prove positive separation")
    if digest(repo_path(checkpoint_path)) != identity:
        raise RuntimeError("checkpoint changed during certification; rerun")
    summary = dict(
        checkpoint=checkpoint_path,
        checkpoint_sha256=identity,
        epsilon=epsilon,
        resolution=[resolution, resolution],
        domain=[1.0, 1.0],
        roundoff_enclosed=False,
        width_upper=width,
        witness_lower=evidence["lower_bound"],
        gap=width / evidence["lower_bound"],
        witnesses=witnesses,
        seconds=timing.seconds,
        peak_gib=timing.peak_bytes / 2**30,
        device=torch.cuda.get_device_name(),
    )
    if summary["gap"] >= 10:
        raise ArithmeticError("reported robustness gap must be below 10")
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary

if __name__ == "__main__":
    main()
