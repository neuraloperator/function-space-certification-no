import gc
import json
import math

import torch

from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.inputs import InputSet
from lean_no.methods import Method
from lean_no.models import restore_fno
from lean_no.paths import repo_path
from lean_no.runtime import Timer, digest, write_json
from scripts._shared.attacks import evaluate_witness, run_mc
from scripts._shared.inputs import WitnessSearch

epsilon = 1e-3
resolution = 4096
time_period = 17 / 16
checkpoint_path = repo_path("models/burgers_1d/robustness.pt")
summary_path = repo_path("results/burgers_spacetime_robustness/summary.json")

case = WitnessSearch(
    length=1.0,
    eps=epsilon,
    full_resolution=resolution,
    grid_resolution=16,
    time_samples=17,
    time_period=time_period,
    decay_mode=None,
    mc_batch_size=64,
)

def initial(x):
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * x)

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    identity = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_times = torch.arange(17, dtype=torch.float64) / 16
    if not torch.equal(
        torch.tensor(checkpoint["data"]["times"]), expected_times.float()
    ):
        raise ValueError("checkpoint time grid differs from the reported experiment")
    model = restore_fno(checkpoint, device=device, dtype=torch.float64)
    mc = run_mc(case, model)
    grid = FourierGrid((resolution, resolution), (1.0, time_period))
    n = resolution
    stop = math.ceil(n / time_period)
    print("Certifying winning Fourier witnesses", flush=True)
    witnesses = evaluate_witness(case, model, mc)
    evidence = witnesses["continuum"]
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Linearized perturbation bounds: {n}²", flush=True)
    with Timer(device) as timing:
        run = certify(
            model=model,
            grid=grid,
            inputs=InputSet(initial, epsilon=epsilon),
            chunk_size=1,
            method=Method("ibp", None, strategy="linearized"),
        )
    output = run.enclosure
    del run
    # The last retained cell covers t=1; including its far corners is conservative.
    width = float((output.upper - output.lower)[..., :stop].amax())
    del output
    gc.collect()
    torch.cuda.empty_cache()
    if evidence["lower_bound"] <= 0:
        raise ArithmeticError("witness enclosures do not prove positive separation")
    gap = width / evidence["lower_bound"]
    if digest(checkpoint_path) != identity:
        raise RuntimeError("checkpoint changed during certification; rerun")

    summary = dict(
        checkpoint=str(checkpoint_path.relative_to(repo_path("."))),
        checkpoint_sha256=identity,
        epsilon=epsilon,
        domain={"x": [0, 1], "t": [0, 1]},
        time_period=time_period,
        input_set="one spatial perturbation shared over time; time fixed",
        resolution=[n, n],
        dtype="float64",
        roundoff_enclosed=False,
        width_upper=width,
        witness_lower=evidence["lower_bound"],
        gap=gap,
        witnesses=witnesses,
        seconds=timing.seconds,
        peak_gib=timing.peak_bytes / 2**30,
        device=torch.cuda.get_device_name(),
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)

    return summary

if __name__ == "__main__":
    main()
