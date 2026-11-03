from training.helmholtz_train import train
from training.poisson_1d_train import train_poisson_1d

def main() -> None:
    train_poisson_1d(
        output_path="models/poisson_1d/solution.pt",
        activation="tanh",
    )
    train(
        resolution=64,
        hidden_channels=4,
        n_layers=1,
        n_epochs=1000,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        non_linearity="tanh",
        train_loss="h1",
        encode=False,
        linf_weight=0.1,
        positional_embedding=None,
    )

if __name__ == "__main__":
    main()
