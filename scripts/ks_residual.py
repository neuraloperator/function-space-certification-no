import math
import sys

import torch

from lean_no.enclosures.cauchy_jet import CauchyProfileJetCache
from lean_no.paths import repo_path
from lean_no.propagation.rollout import (
    initial_state_handoff,
    rollout,
    values,
)
from lean_no.quantities import Residual
from lean_no.quantities.ks import (
    geometry,
    initial_mismatch,
    paper_profile,
    terminal_residual_l2,
)
from lean_no.quantities.residual import residual_bound
from lean_no.runtime import Timer, digest, write_json
from lean_no.spectral.strip import StripGauss

def initial(x):
    return torch.sin(x / 3) + 0.5 * torch.cos(2 * x / 3)

cases = (
    ("ks_64", 360, 8, (1, 64)),
    ("ks_128", 1024, 8, (1, 64)),
    ("ks_pde", (160, 144), 2, (32, 64)),
)

@torch.no_grad()
def run(name, order, slabs, panels):
    checkpoint = repo_path(f"models/ks/{name}.pt")
    identity = digest(checkpoint)
    state = {
        key: value.to(device="cuda", dtype=torch.float64)
        for key, value in torch.load(checkpoint, weights_only=True, map_location="cpu")[
            "model"
        ].items()
    }
    layers = state["fno_blocks.convs.bias"].shape[0]
    profile = paper_profile(device="cuda", inputs=initial)
    reports = []
    for index in range(1, slabs + 1):
        print(f"{name}: rollout {index}/{slabs}", flush=True)
        window_order = order[index - 1] if name == "ks_pde" else order
        with Timer("cuda") as generation:
            rule = StripGauss(
                order=window_order,
                cells=(panels,) * layers,
                batch=(128, 256)[index - 1] if name == "ks_pde" else 32,
                cap_limit=1e100 if name == "ks_pde" else 1e20,
                space_only=layers == 1,
                max_cover_cells=2097152,
                gelu_order=32,
                profile_degree=16,
            )
            field, _, _ = rollout(state, profile, rule, **geometry)
            width = float(2 * field.error.max())
            if not math.isfinite(width):
                raise ArithmeticError("nonfinite propagated width")
            terminal = initial_state_handoff(field, retain_derivatives=True)
            x = torch.linspace(0, 1, 1025, device="cuda", dtype=torch.float64)[None]
            t = x.new_tensor([[32 / 34]])
            torch.testing.assert_close(
                values(field, t, x, active=True),
                values(terminal, t * 0, x),
                rtol=3e-10,
                atol=3e-10,
            )
        with Timer("cuda") as query:
            cache = CauchyProfileJetCache(capacity=131072, degree=32)
            certificate = residual_bound(
                field,
                Residual("ks", 0.01),
                index=index,
                batch=16384,
                enclosure_batch=8192,
                profile_cache=cache,
            )
            report = dict(certificate.diagnostics)
            report.update(
                output_enclosure_width=width,
                gaussian_order=window_order,
                initial_mismatch=initial_mismatch(field, profile, profile_cache=cache),
                terminal_residual_l2=terminal_residual_l2(field, profile_cache=cache),
            )
            report["terminal_residual_l2"]["physical_time"] = index * 0.1
        report.update(
            generation_seconds=generation.seconds,
            query_seconds=query.seconds,
            peak_gib=max(generation.peak_bytes, query.peak_bytes) / 2**30,
        )
        report["certificate_seconds"] = report.pop("seconds")
        reports.append(report)
        if digest(checkpoint) != identity:
            raise RuntimeError("checkpoint changed during certification; rerun")
        write_json(
            repo_path(f"results/ks_residual/{name}/summary.json"),
            dict(
                checkpoint=str(checkpoint.relative_to(repo_path("."))),
                checkpoint_sha256=identity,
                initial_panels=panels,
                rollouts=reports,
                target_gap=0.099,
                roundoff_enclosed=False,
            ),
        )
        profile = terminal
        del cache, field, rule, certificate
    return reports

def main():
    sys.setrecursionlimit(8192)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    summary = {}
    for case in cases:
        summary[case[0]] = run(*case)
        torch.cuda.empty_cache()
    return summary

if __name__ == "__main__":
    main()
