from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from lean_no.models import make_fno, pick_device
from lean_no.paths import repo_path

class LinfLoss:

    def __call__(self, y_pred, y, **kwargs):
        return (y_pred - y).abs().flatten(1).amax(dim=1).mean()

class CombinedLoss:

    def __init__(self, *terms: tuple[object, float]):
        self.terms = terms

    def __call__(self, y_pred, y, **kwargs):
        total = 0.0
        for loss_fn, weight in self.terms:
            total = total + weight * loss_fn(y_pred, y, **kwargs)
        return total

class PairSet(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def eval_abs_linf(model, loader, device, data_processor=None):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                batch = {key: value.to(device) for key, value in batch.items()}
                if data_processor is not None:
                    sample = data_processor.preprocess(batch)
                    pred = model(sample["x"])
                    pred, sample = data_processor.postprocess(pred, sample)
                    y = sample["y"]
                else:
                    pred = model(batch["x"])
                    y = batch["y"]
            else:
                x, y = (value.to(device) for value in batch)
                pred = model(x)
            total += (pred - y).abs().flatten(1).amax(dim=1).sum().item()
            count += y.size(0)
    return total / count

def model_config(
    modes: int = 32,
    channels: int = 4,
    activation: str = "relu",
) -> dict:
    return {
        "n_modes": [modes],
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": channels,
        "n_layers": 1,
        "lifting_channel_ratio": 2,
        "projection_channel_ratio": 2,
        "non_linearity": activation,
        "use_channel_mlp": False,
        "positional_embedding": None,
        "fno_skip": "linear",
        "factorization": None,
    }

def batch_values(pred: torch.Tensor, target: torch.Tensor) -> dict:
    difference = (pred - target).flatten(1)
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    l2_scale = torch.linalg.vector_norm(target_flat, dim=1).clamp_min(1e-8)
    max_scale = target_flat.abs().amax(dim=1).clamp_min(1e-8)
    rel_l2 = torch.linalg.vector_norm(difference, dim=1) / l2_scale
    abs_linf = difference.abs().amax(dim=1)
    mean_error = (pred_flat.mean(dim=1) - target_flat.mean(dim=1)).abs() / max_scale
    # Relative L2 plus a light mean match to remove Dirichlet DC bias without
    # grid embeddings or extra boundary/Linf terms.
    objective = rel_l2 + 0.25 * mean_error
    return {
        "objective": objective,
        "relative_l2": rel_l2,
        "absolute_linf": abs_linf,
        "mean_error": mean_error,
    }

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    names = ("objective", "relative_l2", "absolute_linf", "mean_error")
    values = {name: [] for name in names}
    for x, y in loader:
        metrics = batch_values(model(x.to(device)), y.to(device))
        for name, value in metrics.items():
            values[name].append(value.cpu())
    values = {name: torch.cat(items) for name, items in values.items()}
    return {
        "objective": values["objective"].mean().item(),
        "mean_relative_l2": values["relative_l2"].mean().item(),
        "mean_absolute_linf": values["absolute_linf"].mean().item(),
        "max_absolute_linf": values["absolute_linf"].max().item(),
        "mean_mean_error": values["mean_error"].mean().item(),
        "max_mean_error": values["mean_error"].max().item(),
    }

def train_model(
    train_set: Dataset,
    val_set: Dataset,
    config: dict,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0)
    model = make_fno(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_value = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_values(model(x), y)["objective"].mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        scheduler.step()
        train_value = total / len(train_set)
        val_metrics = evaluate(model, val_loader, device)
        current = val_metrics["objective"]
        history.append(
            {
                "epoch": epoch,
                "train_objective": train_value,
                "validation_objective": current,
            }
        )
        if current < best_value:
            best_value = current
            best_epoch = epoch
            best_state = copy.deepcopy(model).cpu().state_dict()
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:3d}: train={train_value:.6e} "
                f"val_l2={val_metrics['mean_relative_l2']:.6e}",
                flush=True,
            )
        if stale >= patience:
            break

    seconds = time.perf_counter() - start
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    model.load_state_dict(best_state)
    model.to(device)
    validation = evaluate(model, val_loader, device)
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    )
    return {
        "model_state_dict": best_state,
        "model_config": config,
        "best_epoch": best_epoch,
        "seed": seed,
        "parameter_count": sum(item.numel() for item in model.parameters()),
        "metrics": {"validation": validation, "test": None},
        "training": {
            "seconds": seconds,
            "peak_memory": peak_memory,
            "device": device_name,
            "history": history,
        },
    }

def save_model(checkpoint: dict, path: str | Path) -> None:
    output = repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

def train_poisson_1d(
    train_path: str | Path = "data/poisson/poisson_1d_train_64.pt",
    test_path: str | Path = "data/poisson/poisson_1d_test_64.pt",
    output_path: str | Path = "models/poisson_1d/robustness.pt",
    modes: int = 32,
    channels: int = 4,
    seed: int = 0,
    epochs: int = 500,
    patience: int = 75,
    batch_size: int = 32,
    device: torch.device | None = None,
    activation: str = "relu",
) -> dict:
    device = device or pick_device()
    train_data = torch.load(
        repo_path(train_path), map_location="cpu", weights_only=False
    )
    train_set = PairSet(train_data["x"][:720], train_data["y"][:720])
    val_set = PairSet(train_data["x"][720:800], train_data["y"][720:800])
    checkpoint = train_model(
        train_set,
        val_set,
        model_config(modes, channels, activation),
        seed,
        epochs,
        patience,
        batch_size,
        device,
    )

    test_data = torch.load(repo_path(test_path), map_location="cpu", weights_only=False)
    test_set = PairSet(test_data["x"], test_data["y"])
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)
    model = make_fno(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint["metrics"]["test"] = evaluate(model, test_loader, device)
    save_model(checkpoint, output_path)

    validation = checkpoint["metrics"]["validation"]
    test = checkpoint["metrics"]["test"]
    print(f"saved {output_path}")
    print(f"best epoch: {checkpoint['best_epoch']}")
    print(f"validation relative L2: {validation['mean_relative_l2']:.6e}")
    print(f"test relative L2: {test['mean_relative_l2']:.6e}")
    print(f"test absolute L-inf: {test['mean_absolute_linf']:.6e}")
    return checkpoint

if __name__ == "__main__":
    train_poisson_1d()
