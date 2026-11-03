import json
import time
import urllib.request
from pathlib import Path

import neuralop
import numpy as np
import torch
from scipy.io import loadmat

from lean_no.models import make_fno
from lean_no.paths import repo_path
from lean_no.runtime import digest, write_json

data_root = repo_path("data/burgers_pino")
model_root = repo_path("models/burgers_pino")
result_root = repo_path("results/burgers_pino")
data_urls = [
    "https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/burgers_pino.mat",
    "https://hkzdata.s3.us-east-1.amazonaws.com/PINO/data/burgers_pino.mat",
]
config = {
    "seed": 41,
    "epochs": 500,
    "batch_size": 20,
    "learning_rate": 0.001,
    "step_size": 100,
    "gamma": 0.5,
    "threads": 4,
    "resolution": 128,
    "train_count": 720,
    "validation_count": 80,
    "test_count": 200,
    "test_start": 800,
    "viscosity": 0.01,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model": {
        "n_modes": [32, 16],
        "hidden_channels": 16,
        "n_layers": 2,
        "in_channels": 2,
        "out_channels": 1,
        "non_linearity": "tanh",
        "lifting_channel_ratio": 2,
        "projection_channel_ratio": 2,
        "positional_embedding": None,
        "use_channel_mlp": False,
        "fno_skip": "linear",
        "factorization": None,
        "domain_padding": None,
    },
}

def save_checkpoint(path, value):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)

def download_data(path=data_root / "burgers_pino.mat", urls=data_urls):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    errors = []
    for url in urls:
        try:
            print(f"Downloading {url}", flush=True)
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if temporary.stat().st_size == 0:
                raise ValueError("empty dataset download")
            temporary.replace(path)
            write_json(
                path.with_suffix(".source.json"),
                {"url": url, "sha256": digest(path)},
                strict=True,
            )
            return path
        except (OSError, ValueError) as error:
            errors.append(f"{url}: {error}")
            temporary.unlink(missing_ok=True)
    raise RuntimeError(
        "Could not download the PINO data. Supply burgers_pino.mat at "
        f"{path}.\n" + "\n".join(errors)
    )

def input_tensor(initial, times):
    initial = torch.as_tensor(initial, dtype=torch.float32)
    times = torch.as_tensor(times, dtype=torch.float32)
    value = initial[:, None, :, None].expand(-1, 1, -1, len(times))
    coordinate = times[None, None, None].expand(len(initial), 1, initial.shape[-1], -1)
    return torch.cat((value, coordinate), dim=1)

def make_pack(initial, output, times, resolution):
    initial = torch.as_tensor(initial, dtype=torch.float32)
    output = torch.as_tensor(output, dtype=torch.float32)
    if initial.ndim != 2 or output.ndim != 3:
        raise ValueError("expected initial (sample,x) and output (sample,time,x)")
    if output.shape != (len(initial), len(times), initial.shape[-1]):
        raise ValueError("initial, trajectory and time dimensions disagree")
    if resolution < 8 or resolution % 2 or initial.shape[-1] % resolution:
        raise ValueError(
            "spatial resolution must be even and divide the raw resolution"
        )
    if not torch.isfinite(initial).all() or not torch.isfinite(output).all():
        raise ValueError("dataset contains nonfinite values")
    step = initial.shape[-1] // resolution
    initial = initial[:, ::step]
    target = output[:, :, ::step].transpose(1, 2).unsqueeze(1).contiguous()
    return {"initial": initial, "input": input_tensor(initial, times), "target": target}

def load_data(path, settings=config):
    path = Path(path)
    if path.is_file():
        saved = loadmat(path)
        if not all(key in saved for key in ("input", "output", "visc")):
            raise ValueError("PINO MAT data must contain input, output and visc")
        viscosity = float(np.asarray(saved["visc"]).item())
        if not np.isclose(viscosity, settings["viscosity"], rtol=0, atol=1e-8):
            raise ValueError("dataset viscosity does not match the experiment")
        if saved["output"].ndim != 3:
            raise ValueError("PINO output must have axes (sample,time,x)")
        count = saved["output"].shape[1]
        if count not in (100, 101):
            raise ValueError(
                "expected 100 positive-time snapshots or 101 including t=0"
            )
        times = np.arange(1, 101) / 100 if count == 100 else np.arange(101) / 100
        pool = make_pack(saved["input"], saved["output"], times, settings["resolution"])
        test_start = settings["test_start"]
        if len(pool["target"]) < test_start + settings["test_count"]:
            raise ValueError("not enough trajectories for the configured test split")
        test = {
            key: value[test_start : test_start + settings["test_count"]]
            for key, value in pool.items()
        }
        identity = {path.name: digest(path)}
        verified = True
        source = "PINO MAT"
    else:
        paths = [
            path / f"burgers_{name}_{settings['resolution']}.pt"
            for name in ("train", "test")
        ]
        saved = [
            torch.load(item, map_location="cpu", weights_only=True) for item in paths
        ]
        verified = all("visc" in item for item in saved)
        for item in saved:
            if "visc" in item and not np.isclose(
                float(item["visc"].item()), settings["viscosity"], rtol=0, atol=1e-8
            ):
                raise ValueError("PT dataset viscosity does not match the experiment")
        count = saved[0]["y"].shape[-2]
        if count < 2:
            raise ValueError("PT trajectories need at least two time snapshots")
        times = np.linspace(0, 1, count)
        pool, test = [
            make_pack(item["x"], item["y"], times, settings["resolution"])
            for item in saved
        ]
        if len(test["target"]) < settings["test_count"]:
            raise ValueError("not enough trajectories in the PT test file")
        test = {key: value[: settings["test_count"]] for key, value in test.items()}
        identity = {item.name: digest(item) for item in paths}
        source = "NeuralOperator PT; original MAT identity unverified"
    train_count = settings["train_count"]
    end = train_count + settings["validation_count"]
    if (
        train_count < 1
        or settings["validation_count"] < 1
        or settings["test_count"] < 1
    ):
        raise ValueError("all data splits must be nonempty")
    if end > settings["test_start"] or len(pool["target"]) < end:
        raise ValueError("training and validation splits exceed the training pool")
    packs = {
        "train": {key: value[:train_count] for key, value in pool.items()},
        "validation": {key: value[train_count:end] for key, value in pool.items()},
        "test": test,
    }
    metadata = {
        "source": source,
        "sha256": identity,
        "viscosity": settings["viscosity"],
        "viscosity_verified": verified,
        "times": times.tolist(),
        "x": (np.arange(settings["resolution"]) / settings["resolution"]).tolist(),
        "shape": list(packs["train"]["target"].shape[-2:]),
        "split_counts": {name: len(pack["target"]) for name, pack in packs.items()},
        "train_indices": [0, train_count],
        "validation_indices": [train_count, end],
        "test_indices": (
            [settings["test_start"], settings["test_start"] + settings["test_count"]]
            if path.is_file()
            else [0, settings["test_count"]]
        ),
        "includes_initial": bool(times[0] == 0),
    }
    source_path = (
        path.with_suffix(".source.json") if path.is_file() else path / "source.json"
    )
    if source_path.exists():
        metadata["download"] = json.loads(source_path.read_text())
    return packs, metadata

def relative_errors(prediction, target):
    error = prediction - target
    trajectory = error.flatten(1).norm(dim=1) / target.flatten(1).norm(dim=1).clamp_min(
        1e-12
    )
    terminal = error[..., -1].flatten(1).norm(dim=1) / target[..., -1].flatten(1).norm(
        dim=1
    ).clamp_min(1e-12)
    return trajectory, terminal

def evaluate(model, pack, batch_size, device):
    model.eval()
    with torch.no_grad():
        prediction = torch.cat(
            [model(batch.to(device)).cpu() for batch in pack["input"].split(batch_size)]
        )
    trajectory, terminal = relative_errors(prediction, pack["target"])
    metrics = {}
    for name, errors in (("trajectory", trajectory), ("terminal", terminal)):
        metrics[name] = {
            "mean_relative_l2": errors.mean().item(),
            "p95_relative_l2": errors.quantile(0.95).item(),
            "max_relative_l2": errors.max().item(),
        }
    return metrics, prediction

def train_model(
    settings,
    packs,
    metadata,
    model_dir=model_root,
    result_dir=result_root,
    stop_epoch=None,
):
    model_dir, result_dir = Path(model_dir), Path(result_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings["threads"])
    torch.manual_seed(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(settings["device"])
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True)
    model = make_fno(settings["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, settings["step_size"], settings["gamma"]
    )
    runtime = {
        "torch": str(torch.__version__),
        "neuraloperator": str(neuralop.__version__),
        "tf32": False,
    }
    latest = model_dir / "latest.pt"
    history, best, elapsed = [], float("inf"), 0.0
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        checkpoint["data"].pop("runner_sha256", None)
        if any(
            checkpoint[key] != value
            for key, value in (
                ("config", settings),
                ("data", metadata),
            )
        ):
            raise ValueError("resume configuration or dataset mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, best, elapsed = (
            checkpoint["history"],
            checkpoint["best_validation"],
            checkpoint["elapsed_seconds"],
        )
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device)
    write_json(
        result_dir / "config.json",
        {"config": settings, "data": metadata, "runtime": runtime},
        strict=True,
    )
    start = time.perf_counter()
    end_epoch = min(stop_epoch or settings["epochs"], settings["epochs"])
    x, y = packs["train"]["input"], packs["train"]["target"]
    for epoch in range(len(history), end_epoch):
        model.train()
        order = torch.randperm(
            len(x), generator=torch.Generator().manual_seed(settings["seed"] + epoch)
        )
        total = 0.0
        for indices in order.split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[indices].to(device))
            loss = relative_errors(prediction, y[indices].to(device))[0].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        scheduler.step()
        validation, _ = evaluate(
            model, packs["validation"], settings["batch_size"], device
        )
        score = validation["trajectory"]["mean_relative_l2"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_relative_l2": total / len(x),
                "validation": validation,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        improved = score < best
        best = min(best, score)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": settings["model"],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": settings,
            "data": metadata,
            "runtime": runtime,
            "epoch": epoch + 1,
            "history": history,
            "best_validation": best,
            "elapsed_seconds": elapsed + time.perf_counter() - start,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
        }
        save_checkpoint(latest, checkpoint)
        if improved:
            save_checkpoint(model_dir / "best.pt", checkpoint)
        write_json(result_dir / "history.json", history, strict=True)
        if epoch < 3 or (epoch + 1) % 10 == 0 or epoch + 1 == end_epoch:
            print(
                f"epoch {epoch + 1}/{settings['epochs']} train={total / len(x):.6f} "
                f"validation={score:.6f}",
                flush=True,
            )
    summary = {
        "status": "complete" if len(history) == settings["epochs"] else "partial",
        "epochs": len(history),
        "best_validation": best,
        "elapsed_seconds": elapsed + time.perf_counter() - start,
    }
    if summary["status"] == "complete":
        checkpoint = torch.load(
            model_dir / "best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics, prediction = evaluate(
            model, packs["test"], settings["batch_size"], device
        )
        trajectory, terminal = relative_errors(prediction, packs["test"]["target"])
        np.savez_compressed(
            result_dir / "test_predictions.npz",
            prediction=prediction.numpy(),
            target=packs["test"]["target"].numpy(),
            initial=packs["test"]["initial"].numpy(),
            trajectory_errors=trajectory.numpy(),
            terminal_errors=terminal.numpy(),
            x=np.asarray(metadata["x"]),
            t=np.asarray(metadata["times"]),
        )
        summary.update(
            {
                "metrics": metrics,
                "best_epoch": checkpoint["epoch"],
                "checkpoint_sha256": digest(model_dir / "best.pt"),
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "provisional_gate": metrics["trajectory"]["mean_relative_l2"] < 0.01
                and metrics["trajectory"]["p95_relative_l2"] < 0.03,
            }
        )
    write_json(result_dir / "summary.json", summary, strict=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary

def main():
    path = download_data()
    packs, metadata = load_data(path)
    train_model(config, packs, metadata)

if __name__ == "__main__":
    main()
