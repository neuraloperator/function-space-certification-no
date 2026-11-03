import math

import torch

from lean_no.paths import repo_path

def sample_coefficients(n_samples=5000, n_modes=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(
            n_samples,
            n_modes,
            n_modes,
            generator=generator,
            dtype=torch.float64,
        )
        * 2
        / n_modes
    )

def sine_basis(n_modes, coordinates):
    modes = torch.arange(1, n_modes + 1, dtype=coordinates.dtype)
    return torch.sin(math.pi * modes[:, None] * coordinates[None, :])

def poisson_eigenvalues(n_modes, dtype=torch.float64):
    modes = torch.arange(1, n_modes + 1, dtype=dtype)
    return math.pi**2 * (modes[:, None] ** 2 + modes[None, :] ** 2)

def evaluate_fields(coefficients, resolution, chunk_size=128):
    coefficients = coefficients.double()
    coordinates = torch.arange(resolution, dtype=torch.float64) / resolution
    basis = sine_basis(coefficients.shape[1], coordinates)
    eigenvalues = poisson_eigenvalues(coefficients.shape[1])

    forcing = []
    solution = []
    for start in range(0, len(coefficients), chunk_size):
        chunk = coefficients[start : start + chunk_size]
        forcing.append(torch.einsum("bmn,mi,nj->bij", chunk, basis, basis))
        solution.append(
            torch.einsum("bmn,mi,nj->bij", chunk / eigenvalues, basis, basis)
        )

    return torch.cat(forcing).unsqueeze(1), torch.cat(solution).unsqueeze(1)

def generate_dataset(
    output_dir="data/poisson",
    resolutions=(32, 64),
    n_train=4000,
    n_val=500,
    n_test=500,
    n_modes=12,
    seed=0,
    chunk_size=128,
):
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficients = sample_coefficients(n_train + n_val + n_test, n_modes, seed)
    splits = {
        "train": coefficients[:n_train],
        "val": coefficients[n_train : n_train + n_val],
        "test": coefficients[n_train + n_val :],
    }

    paths = {}
    for resolution in resolutions:
        for split, split_coefficients in splits.items():
            x, y = evaluate_fields(split_coefficients, resolution, chunk_size)
            path = output_dir / f"poisson_2d_{split}_{resolution}.pt"
            torch.save(
                {
                    "x": x.float(),
                    "y": y.float(),
                    "coefficients": split_coefficients.float(),
                },
                path,
            )
            paths[(split, resolution)] = path
            print(f"Saved {path}: {tuple(x.shape)}")
    return paths

def main():
    generate_dataset()

if __name__ == "__main__":
    main()
