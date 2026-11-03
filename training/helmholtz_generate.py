from __future__ import annotations

import math

import torch

from lean_no.domain import periodic_grid
from lean_no.paths import repo_path

length = 1.0
resonant_mode = 4
frequency_gap = 0.02

def wave_number(
    mode: int = resonant_mode,
    gap: float = frequency_gap,
    domain_length: float = length,
) -> float:
    frequency = 2 * math.pi * mode / domain_length
    if gap <= 0 or gap >= frequency**2:
        raise ValueError("gap must lie between zero and the resonant eigenvalue")
    return math.sqrt(frequency**2 - gap)

def sample_functions(
    n_samples: int = 1000,
    n_points: int = 256,
    modes: int = 15,
    length_scale: float = 0.2,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = periodic_grid(n_points, length, dtype=torch.float32)
    indices = torch.arange(1, modes + 1, dtype=x.dtype)
    weights = sigma * torch.exp(-((math.pi * indices * length_scale) ** 2))
    cosine = torch.randn(n_samples, modes) * weights
    sine = torch.randn(n_samples, modes) * weights
    angles = 2 * math.pi * indices[:, None] * x[None, :] / length
    fields = cosine @ torch.cos(angles) + sine @ torch.sin(angles)
    return x, fields

def solve_helmholtz(
    forcing: torch.Tensor,
    kappa: float,
    domain_length: float = length,
) -> torch.Tensor:
    count = forcing.shape[-1]
    dtype = forcing.dtype
    values = forcing.to(torch.float64)
    frequencies = (
        2
        * math.pi
        * torch.fft.rfftfreq(
            count,
            d=domain_length / count,
            device=forcing.device,
            dtype=values.dtype,
        )
    )
    denominator = frequencies.square() - kappa**2
    if denominator.abs().min() < 1e-10:
        raise ValueError("kappa is a periodic Helmholtz eigenfrequency")
    solution = torch.fft.irfft(
        torch.fft.rfft(values) / denominator,
        n=count,
    )
    return solution.to(dtype)

def main() -> None:
    resolutions = [32, 64, 128, 256, 512]
    train_count = 800
    test_count = 200
    sample_count = train_count + test_count
    max_mode = 15
    kappa = wave_number()

    path = repo_path("data/helmholtz")
    path.mkdir(parents=True, exist_ok=True)

    for resolution in resolutions:
        torch.manual_seed(0)
        _, forcing = sample_functions(sample_count, resolution, max_mode)
        solution = solve_helmholtz(forcing, kappa)
        data = {
            "x": forcing.unsqueeze(1),
            "y": solution.unsqueeze(1),
            "kappa": kappa,
            "resonant_mode": resonant_mode,
            "frequency_gap": frequency_gap,
        }
        output = path / f"helmholtz_1d_{resolution}.pt"
        torch.save(data, output)
        print(f"Saved {output}")

if __name__ == "__main__":
    main()
