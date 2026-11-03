import copy
import time

import torch
from neuralop import H1Loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from lean_no.models import make_fno, pick_device
from lean_no.paths import repo_path
from lean_no.runtime import digest, write_json

class PoissonDataset(Dataset):
    def __init__(self, path):
        self.path = repo_path(path)
        data = torch.load(self.path, weights_only=False)
        self.x = data["x"]
        self.y = data["y"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}

def relative_l2(prediction, target):
    error = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    scale = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-8)
    return (error / scale).mean()

def absolute_linf(prediction, target):
    return (prediction - target).abs().flatten(1).amax(dim=1).mean()

def boundary_linf(prediction):
    row = prediction[..., 0, :].abs().flatten(1).amax(dim=1)
    column = prediction[..., :, 0].abs().flatten(1).amax(dim=1)
    return torch.maximum(row, column)

@torch.no_grad()
def evaluate(model, loader, device, h1_loss):
    model.eval()
    relative_l2_values = []
    linf_values = []
    boundary_values = []
    h1_total = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        prediction = model(x)
        difference = (prediction - y).flatten(1)
        relative_l2_values.append(
            (
                torch.linalg.vector_norm(difference, dim=1)
                / torch.linalg.vector_norm(y.flatten(1), dim=1).clamp_min(1e-8)
            ).cpu()
        )
        linf_values.append(difference.abs().amax(dim=1).cpu())
        boundary_values.append(boundary_linf(prediction).cpu())
        h1_total += h1_loss(prediction, y).item() * len(x)

    relative_l2_values = torch.cat(relative_l2_values)
    linf_values = torch.cat(linf_values)
    boundary_values = torch.cat(boundary_values)
    return {
        "mean_relative_l2": relative_l2_values.mean().item(),
        "mean_h1": h1_total / len(loader.dataset),
        "mean_absolute_linf": linf_values.mean().item(),
        "max_absolute_linf": linf_values.max().item(),
        "max_represented_boundary": boundary_values.max().item(),
    }

def train_poisson_2d(
    data_dir="data/poisson",
    output_path="models/poisson_2d/robustness.pt",
    train_resolution=32,
    test_resolutions=(32, 64),
    n_modes=12,
    hidden_channels=8,
    n_layers=2,
    n_epochs=200,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-6,
    h1_weight=0.1,
    linf_weight=0.1,
    seed=0,
    device=None,
    num_workers=0,
    activation="relu",
    positional_embedding=None,
):
    if activation not in ("relu", "tanh"):
        raise ValueError("activation must be relu or tanh")
    data_dir = repo_path(data_dir)
    torch.manual_seed(seed)
    device = torch.device(device or "cpu")

    train_loader = DataLoader(
        PoissonDataset(data_dir / f"poisson_2d_train_{train_resolution}.pt"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        PoissonDataset(data_dir / f"poisson_2d_val_{train_resolution}.pt"),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loaders = {
        resolution: DataLoader(
            PoissonDataset(data_dir / f"poisson_2d_test_{resolution}.pt"),
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for resolution in test_resolutions
    }

    model_config = {
        "n_modes": [n_modes, n_modes],
        "hidden_channels": hidden_channels,
        "n_layers": n_layers,
        "in_channels": 1,
        "positional_embedding": positional_embedding,
        "non_linearity": activation,
    }

    model = make_fno(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    h1_loss = H1Loss(
        d=2,
        reduction="mean",
        periodic_in_x=False,
        periodic_in_y=False,
    )

    best_epoch = 0
    best_validation = float("inf")
    best_state = None
    history = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset_paths = [train_loader.dataset, val_loader.dataset] + [
        loader.dataset for loader in test_loaders.values()
    ]
    dataset_hashes = {
        (
            str(dataset.path.relative_to(repo_path(".")))
            if dataset.path.is_relative_to(repo_path("."))
            else str(dataset.path)
        ): digest(dataset.path)
        for dataset in dataset_paths
    }
    train_start = time.perf_counter()

    print(f"Training {activation} on {device}", flush=True)
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = (
                relative_l2(prediction, y)
                + h1_weight * h1_loss(prediction, y)
                + linf_weight * absolute_linf(prediction, y)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        scheduler.step()

        validation = evaluate(model, val_loader, device, h1_loss)
        train_loss = total_loss / len(train_loader.dataset)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_relative_l2": validation["mean_relative_l2"],
            }
        )
        if validation["mean_relative_l2"] < best_validation:
            best_validation = validation["mean_relative_l2"]
            best_epoch = epoch
            best_state = copy.deepcopy(model).cpu().state_dict()

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print(
                f"epoch {epoch:3d}/{n_epochs}: "
                f"train={train_loss:.6e} "
                f"val={validation['mean_relative_l2']:.6e}",
                flush=True,
            )

    train_seconds = time.perf_counter() - train_start
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

    model.load_state_dict(best_state)
    model.to(device)
    metrics = {
        "validation": evaluate(model, val_loader, device, h1_loss),
        "test": {
            str(resolution): evaluate(model, loader, device, h1_loss)
            for resolution, loader in test_loaders.items()
        },
    }
    accepted = (
        metrics["validation"]["mean_relative_l2"] <= 0.05
        and metrics["test"]["32"]["mean_relative_l2"] <= 0.05
        and metrics["test"]["64"]["mean_relative_l2"] <= 0.10
    )
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    )
    training = {
        "final_train_loss": history[-1]["train_loss"],
        "seconds": train_seconds,
        "peak_memory": peak_memory,
        "device": device_name,
        "accepted": accepted,
        "history": history,
        "settings": dict(
            train_resolution=train_resolution,
            test_resolutions=list(test_resolutions),
            epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            h1_weight=h1_weight,
            linf_weight=linf_weight,
            seed=seed,
        ),
        "dataset_sha256": dataset_hashes,
    }
    checkpoint = {
        "model_state_dict": best_state,
        "model_config": model_config,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "training": training,
    }
    output_path = repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    summary = {
        key: value for key, value in checkpoint.items() if key != "model_state_dict"
    }
    summary["training"] = {
        key: value for key, value in training.items() if key != "history"
    }
    summary["checkpoint_sha256"] = digest(output_path)
    write_json(output_path.with_suffix(".json"), summary)
    print(f"Saved {output_path} from epoch {best_epoch}")
    print(f"  final training objective  {training['final_train_loss']:.6e}")
    print(f"  best validation L2       {metrics['validation']['mean_relative_l2']:.6e}")
    print(f"  training time            {training['seconds']:.3f} s")
    print(f"  peak GPU memory          {training['peak_memory'] / 2**30:.3f} GiB")
    print(f"  accepted                 {training['accepted']}")
    for name, values in (
        ("validation", metrics["validation"]),
        ("test-32", metrics["test"]["32"]),
        ("test-64", metrics["test"]["64"]),
    ):
        print(
            f"  {name:10s} relative L2={values['mean_relative_l2']:.6e} "
            f"H1={values['mean_h1']:.6e} "
            f"mean L-inf={values['mean_absolute_linf']:.6e} "
            f"max L-inf={values['max_absolute_linf']:.6e} "
            f"boundary={values['max_represented_boundary']:.6e}"
        )
    return checkpoint

def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_poisson_2d(
        output_path="models/poisson_2d/solution.pt",
        activation="tanh",
        positional_embedding="grid",
        n_epochs=400,
        device=pick_device(),
    )

if __name__ == "__main__":
    main()
