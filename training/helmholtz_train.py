from __future__ import annotations

from pathlib import Path

import torch
from neuralop import H1Loss, LpLoss, Trainer
from neuralop.data.datasets import PTDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from lean_no.models import make_fno, pick_device
from lean_no.paths import repo_path
from training.poisson_1d_train import CombinedLoss, LinfLoss, eval_abs_linf

class HelmholtzDataset(PTDataset):
    def __init__(
        self,
        root_dir: str | Path = "data/helmholtz",
        dataset_file: str = "helmholtz_1d_64.pt",
        n_train: int = 800,
        n_tests: list[int] | None = None,
        batch_size: int = 32,
        test_batch_sizes: list[int] | None = None,
        train_resolution: int = 64,
        test_resolutions: list[int] | None = None,
        encode_input: bool = False,
        encode_output: bool = False,
    ) -> None:
        root = repo_path(root_dir)
        test_counts = n_tests or [200]
        test_batches = test_batch_sizes or [batch_size]
        test_grids = test_resolutions or [train_resolution]
        train_path = root / f"helmholtz_1d_train_{train_resolution}.pt"
        test_path = root / f"helmholtz_1d_test_{test_grids[0]}.pt"
        full_path = root / dataset_file
        need_split = (
            not train_path.exists()
            or not test_path.exists()
            or (
                full_path.exists()
                and (
                    full_path.stat().st_mtime > train_path.stat().st_mtime
                    or full_path.stat().st_mtime > test_path.stat().st_mtime
                )
            )
        )
        if need_split:
            data = torch.load(full_path, weights_only=False)
            test_count = test_counts[0]
            x, y = data["x"], data["y"]
            if n_train + test_count > x.shape[0]:
                raise ValueError(f"not enough samples in {dataset_file}")
            torch.save(
                {"x": x[:n_train], "y": y[:n_train]},
                train_path,
            )
            torch.save(
                {
                    "x": x[n_train : n_train + test_count],
                    "y": y[n_train : n_train + test_count],
                },
                test_path,
            )

        super().__init__(
            root_dir=root,
            dataset_name="helmholtz_1d",
            n_train=n_train,
            n_tests=test_counts,
            batch_size=batch_size,
            test_batch_sizes=test_batches,
            train_resolution=train_resolution,
            test_resolutions=test_grids,
            encode_input=encode_input,
            encode_output=encode_output,
            channel_dim=1,
            channels_squeezed=False,
        )

def save_model(
    model: torch.nn.Module,
    output_path: str | Path = "models/helmholtz_1d/robustness.pt",
    config: dict | None = None,
) -> None:
    path = repo_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        model.state_dict()
        if config is None
        else {
            "model_state_dict": model.state_dict(),
            "model_config": config,
        }
    )
    torch.save(checkpoint, path)
    print(f"Saved model weights to {path.resolve()}")

def train(
    resolution: int = 64,
    hidden_channels: int = 32,
    n_layers: int = 1,
    n_epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    non_linearity: str = "relu",
    train_loss: str = "h1",
    encode: bool = False,
    linf_weight: float = 0.1,
    positional_embedding: str | None = None,
    verbose: bool = True,
    save: bool = True,
    eval_interval: int = 10,
) -> float:
    device = pick_device()
    dataset = HelmholtzDataset(
        dataset_file=f"helmholtz_1d_{resolution}.pt",
        batch_size=batch_size,
        train_resolution=resolution,
        test_resolutions=[resolution],
        encode_input=encode,
        encode_output=encode,
    )
    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        num_workers=0,
    )
    test_loaders = {
        grid: DataLoader(data, batch_size=batch_size, num_workers=0)
        for grid, data in dataset.test_dbs.items()
    }

    config = {
        "n_modes": [32],
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": hidden_channels,
        "n_layers": n_layers,
        "lifting_channel_ratio": 2,
        "projection_channel_ratio": 2,
        "non_linearity": non_linearity,
        "use_channel_mlp": False,
        "positional_embedding": positional_embedding,
        "fno_skip": "linear",
        "factorization": None,
    }
    model = make_fno(config).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    h1_loss = H1Loss(d=1)
    l2_loss = LpLoss(d=1, p=2, reduction="mean")
    linf_loss = LinfLoss()
    if train_loss == "h1":
        primary = h1_loss
    elif train_loss == "l2":
        primary = l2_loss
    else:
        raise ValueError(f"unsupported train_loss {train_loss!r}")
    training_loss = (
        CombinedLoss((primary, 1.0), (linf_loss, linf_weight))
        if linf_weight > 0
        else primary
    )

    trainer = Trainer(
        model=model,
        n_epochs=n_epochs,
        device=device,
        data_processor=dataset.data_processor,
        wandb_log=False,
        eval_interval=eval_interval,
        use_distributed=False,
        verbose=verbose,
    )
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=training_loss,
        eval_losses={"h1": h1_loss, "l2": l2_loss, "linf": linf_loss},
    )

    test_linf = eval_abs_linf(
        model,
        test_loaders[resolution],
        device,
        data_processor=dataset.data_processor,
    )
    print(f"Test abs L-inf error at N={resolution}: {test_linf:.6e}")
    if save:
        name = "solution" if non_linearity == "tanh" else "robustness"
        save_model(model.cpu(), f"models/helmholtz_1d/{name}.pt", config)
    return test_linf

def main() -> None:
    train(resolution=64, hidden_channels=4, n_layers=1, n_epochs=1000)

if __name__ == "__main__":
    main()
