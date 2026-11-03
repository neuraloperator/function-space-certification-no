import numpy as np
import torch

from lean_no.domain import periodic_grid
from lean_no.paths import repo_path

def sample_functions(
    n_samples=1000,
    n_points=256,
    modes=50,
    length_scale=0.2,
    sigma=1.0,
):
    x = periodic_grid(n_points, dtype=torch.float32)

    k = torch.arange(1, modes + 1).float()
    weights = sigma * torch.exp(-((np.pi * k * length_scale) ** 2))

    data = []

    for _ in range(n_samples):
        cosine = torch.randn(modes) * weights
        sine = torch.randn(modes) * weights

        cos_part = torch.cos(2 * np.pi * k[:, None] * x[None, :])
        sin_part = torch.sin(2 * np.pi * k[:, None] * x[None, :])

        field = (cosine[:, None] * cos_part + sine[:, None] * sin_part).sum(dim=0)
        data.append(field)

    data = torch.stack(data)  # (n_samples, n_points)
    return x, data

def green_function(x, y):
    return 0.5 * (x + y - np.abs(x - y)) - x * y

def compute_u(x, f):
    x_np = x.numpy()
    f_np = f.numpy()

    # Build Green's matrix G_ij = G(x_i, x_j)
    x_grid, y_grid = np.meshgrid(x_np, x_np, indexing="ij")
    green = green_function(x_grid, y_grid)  # shape (n_points, n_points)

    # Approximate integral by trapezoidal rule:
    # u_i = \int_0^1 G(x_i, y) f(y) dy
    # For batch: u = f @ G^T? careful with axes:
    # G[i,j] = G(x_i, y_j), so for each sample:
    # u[:, i] = trapz(G[i,:] * f[:,:], x)
    u = np.trapezoid(
        green[None, :, :] * f_np[:, None, :], x_np, axis=2
    )  # (n_samples, n_points)

    return torch.from_numpy(u).float()

def main():
    discretizations = [32, 64, 128, 256, 512]
    n_train = 800
    n_test = 200
    n_total = n_train + n_test

    path = repo_path("data/poisson")
    path.mkdir(parents=True, exist_ok=True)

    max_mode = 15

    for n_points in discretizations:
        torch.manual_seed(0)
        x, f = sample_functions(n_samples=n_total, n_points=n_points, modes=max_mode)
        u = compute_u(x, f)
        f = f.unsqueeze(1)
        u = u.unsqueeze(1)

        data = {"x": f, "y": u}

        train_path = path / f"poisson_1d_{n_points}.pt"
        torch.save(data, train_path)

if __name__ == "__main__":
    main()
