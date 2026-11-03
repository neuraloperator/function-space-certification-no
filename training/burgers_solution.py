import hashlib
import json
import math
import time

import numpy as np
import torch

from lean_no.bounds.operators import affine_weight
from lean_no.function_space.analytic import Series
from lean_no.models import make_fno
from lean_no.paths import repo_path
from lean_no.reference import cole_hopf_reference
from lean_no.runtime import digest, write_json
from training.burgers_time import download_data, save_checkpoint

period = 17 / 16
root = repo_path("models/burgers_1d")

def derivatives(values, period=period):
    nx, nt = values.shape[-2:]
    wx = 2 * math.pi * torch.fft.fftfreq(nx, device=values.device, dtype=values.dtype)
    wt = (
        2
        * math.pi
        * torch.fft.rfftfreq(nt, device=values.device, dtype=values.dtype)
        / period
    )
    # fftfreq's default spacing is one sample; convert to unit/physical domain.
    wx, wt = wx * nx, wt * nt
    spectrum = torch.fft.rfftn(values, dim=(-2, -1))
    return [values] + [
        torch.fft.irfftn(spectrum * factor, s=(nx, nt), dim=(-2, -1))
        for factor in (1j * wx[:, None], 1j * wt[None, :], -(wx[:, None] ** 2))
    ]

def activate(jet):
    value, x, t, xx = jet
    a = value.tanh()
    first = 1 - a.square()
    return [a, first * x, first * t, first * xx - 2 * a * first * x.square()]

def affine(jet, module):
    weight = affine_weight(module)
    source = module.conv if hasattr(module, "conv") else module
    bias = getattr(source, "bias", None)
    result = [torch.einsum("oi,bixy->boxy", weight, value) for value in jet]
    if bias is not None:
        result[0] = result[0] + bias.reshape(1, -1, 1, 1)
    return result

def input_jet(initial, nx, nt):
    batch, n = initial.shape
    k = torch.arange(nx // 2 + 1, device=initial.device, dtype=initial.dtype)
    spectrum = torch.zeros(
        batch,
        len(k),
        device=initial.device,
        dtype=torch.complex64 if initial.dtype == torch.float32 else torch.complex128,
    )
    spectrum[:, : n // 2 + 1] = torch.fft.rfft(initial, norm="forward")
    spectrum[:, n // 2] *= 0.5
    spectrum = spectrum * torch.exp(1j * math.pi * k / nx)
    fields = [
        torch.fft.irfft(spectrum * (2j * math.pi * k) ** order, n=nx, norm="forward")
        for order in (0, 1, 2)
    ]
    a, ax, axx = [value[:, None, :, None].expand(-1, 1, -1, nt) for value in fields]
    time_grid = (
        (torch.arange(nt, device=initial.device, dtype=initial.dtype) + 0.5)
        * period
        / nt
    )
    t = time_grid[None, None, None, :].expand(batch, 1, nx, -1)
    zeros = torch.zeros_like(a)
    return [
        torch.cat((a, t), 1),
        torch.cat((ax, zeros), 1),
        torch.cat((zeros, torch.ones_like(a)), 1),
        torch.cat((axx, zeros), 1),
    ]

def solution_jet(model, initial, nx=128, nt=128, *, return_trace=False):
    inputs = input_jet(initial, nx, nt)
    jet = inputs
    trace = inputs[0][..., :1].clone()
    trace[:, 1] = 0
    for i, layer in enumerate(model.lifting.fcs):
        jet = affine(jet, layer)
        trace = affine([trace], layer)[0]
        if i + 1 < len(model.lifting.fcs):
            jet = activate(jet)
            trace = trace.tanh()
    for i, conv in enumerate(model.fno_blocks.convs):
        skip = affine(jet, model.fno_blocks.fno_skips[i])
        spectral_value = conv(jet[0])
        spectral = derivatives(spectral_value)
        # Evaluate each block's periodic convolution at physical time zero
        # from the SAME midpoint moments used by the interior prediction.
        frequencies = (
            torch.fft.fftfreq(nt, device=initial.device, dtype=initial.dtype) * nt
        )
        phase = torch.exp(-1j * math.pi * frequencies / nt)
        spectral_trace = (
            (torch.fft.fft(spectral_value, dim=-1, norm="forward") * phase)
            .sum(-1, keepdim=True)
            .real
        )
        trace = spectral_trace + affine([trace], model.fno_blocks.fno_skips[i])[0]
        jet = [a + b for a, b in zip(skip, spectral)]
        if i + 1 < model.n_layers:
            jet = activate(activate(jet))
            trace = trace.tanh().tanh()
    for i, layer in enumerate(model.projection.fcs):
        jet = affine(jet, layer)
        trace = affine([trace], layer)[0]
        if i + 1 < len(model.projection.fcs):
            jet = activate(jet)
            trace = trace.tanh()
    return (jet, trace) if return_trace else jet

@torch.no_grad()
def heldout_evaluation(model):
    path = repo_path("data/burgers_mini/burgers_test_16.pt")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    inputs = saved["x"][:16].float()
    times = (np.arange(128) + 0.5) * period / 128
    mask = times <= 1
    reference = np.stack(
        [
            cole_hopf_reference(Series.from_samples(a), times[mask], 1024)[:, ::8].T
            for a in inputs
        ]
    )
    # Evaluate at the SAME midpoint x coordinates as the continuous jet.
    spectrum = np.fft.rfft(reference, axis=1, norm="forward")
    spectrum[:, -1] = 0
    spectrum *= np.exp(1j * np.pi * np.arange(65) / 128)[None, :, None]
    reference = np.fft.irfft(spectrum, n=128, axis=1, norm="forward")
    predictions = []
    device = next(model.parameters()).device
    for batch in inputs.split(4):
        predictions.append(
            solution_jet(model, batch.to(device))[0][:, 0, :, mask].cpu().numpy()
        )
    prediction = np.concatenate(predictions)
    error = prediction - reference
    relative = np.sqrt(
        np.sum(error**2, axis=(1, 2)) / np.sum(reference**2, axis=(1, 2))
    )
    return {
        "certified": False,
        "public_test_indices": [0, 16],
        "test_dataset_sha256": digest(path),
        "mean_relative_trajectory_l2": float(relative.mean()),
        "max_relative_trajectory_l2": float(relative.max()),
        "mean_max_time_l2_error": float(
            np.sqrt(np.mean(error**2, axis=1)).max(axis=1).mean()
        ),
    }

def main():
    torch.set_num_threads(4)
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_path = repo_path("models/burgers_1d/robustness.pt")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    data_path = repo_path("data/burgers_mini/burgers_train_16.pt")
    download_data(data_path, [source["data"]["download"]["urls"][data_path.name]])
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    if digest(data_path) != source["data"]["sha256"][data_path.name]:
        raise ValueError("public training dataset identity changed")
    model = make_fno(source["model_config"]).cuda()
    model.load_state_dict(source["model_state_dict"])
    x = torch.arange(16, dtype=torch.float64) / 16
    nominal = (torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * x)).float()
    # Fixed public training indices; exclude validation/test splits.
    initial = data["x"][:64].float() * torch.linspace(0.75, 2.5, 64)[:, None]
    initial = torch.cat((nominal[None], initial))
    nx = nt = 128
    times = (np.arange(nt) + 0.5) * period / nt
    # Numerical training targets; the final solution certificate is independent.
    targets = []
    for a in initial:
        values = cole_hopf_reference(Series.from_samples(a), times, 1024)
        # Periodic Fourier interpolation to spatial midpoint training nodes.
        spectrum = np.fft.rfft(values, axis=1, norm="forward")[:, : nx // 2 + 1]
        spectrum[:, -1] = 0
        spectrum *= np.exp(1j * np.pi * np.arange(nx // 2 + 1) / nx)
        targets.append(np.fft.irfft(spectrum, n=nx, axis=1, norm="forward").T)
    targets = torch.tensor(np.asarray(targets), dtype=torch.float32, device="cuda")[
        :, None
    ]
    inputs = initial.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    started = time.perf_counter()
    history = []
    model.train()
    # The nominal input is present in every batch. Other inputs keep an
    # operator signal; selection below uses the nominal residual and accuracy.
    for step in range(2000):
        ids = torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.long),
                torch.randint(1, len(inputs), (3,), device="cuda"),
            )
        )
        optimizer.zero_grad(set_to_none=True)
        (v, vx, vt, vxx), trace = solution_jet(
            model, inputs[ids], nx, nt, return_trace=True
        )
        mask = times <= 1
        residual = vt[..., mask] + v[..., mask] * vx[..., mask] - 0.01 * vxx[..., mask]
        fit = (v - targets[ids]).square().mean()
        physics = residual.square().mean()
        a0 = input_jet(inputs[ids], nx, nt)[0][:, :1, :, :1]
        initial_loss = (trace - a0).square().mean()
        loss = fit + initial_loss + 0.05 * physics
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite fine-tuning loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 999:
            for group in optimizer.param_groups:
                group["lr"] = 0.0003
        if step == 0 or (step + 1) % 100 == 0:
            row = dict(
                step=step + 1,
                fit=float(fit.detach()),
                initial_mse=float(initial_loss.detach()),
                residual_mse=float(physics.detach()),
                seconds=time.perf_counter() - started,
            )
            history.append(row)
            print(json.dumps(row), flush=True)
    evaluation = heldout_evaluation(model.eval())
    print(json.dumps(dict(phase="held-out evaluation", **evaluation)), flush=True)
    root.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        root / "solution.pt",
        {
            "model_state_dict": model.cpu().state_dict(),
            "model_config": source["model_config"],
            "data": {k: v for k, v in source["data"].items() if k != "runner_sha256"},
            "solution_ansatz": "direct_fno",
            "training": {
                "seed": 41,
                "steps": 2000,
                "midpoint_grid": [nx, nt],
                "batch_size": 4,
                "public_train_indices": [0, 64],
                "amplitude_scale_range": [0.75, 2.5],
                "nominal_in_training": True,
                "source_checkpoint_sha256": digest(source_path),
                "input_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                "loss": "reference MSE + initial MSE + 0.05 * continuous-coordinate Burgers residual MSE",
                "history": history,
                "heldout_evaluation": evaluation,
            },
        },
    )
    write_json(
        repo_path("results/burgers_solution_training/summary.json"),
        {
            "checkpoint": "models/burgers_1d/solution.pt",
            "history": history,
            "nominal_in_training": True,
            "heldout_evaluation": evaluation,
        },
        strict=True,
    )

if __name__ == "__main__":
    main()
