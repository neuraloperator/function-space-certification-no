# Function-space certification for neural operators

Codebase for the paper **Function-Space Certification for Neural Operators**.
Package version: **0.1.0**. 

## System requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Dependencies and platform-specific versions are pinned in [uv.lock](uv.lock);
  optional dependency groups are defined in [pyproject.toml](pyproject.toml).
- Full paper experiments require an NVIDIA GPU.

## Install

Use Python 3.12 and run commands from the repository root:

```bash
uv sync --frozen --group dev
```

## Paper experiments

Each checkmark links to a standalone script with fixed experiment settings.
Run a script as a module, for example:

```bash
uv run python -m scripts.poisson_1d_residual
```

| PDE / Model | Solution error | Residual | Robustness | Max temp. |
| --- | :---: | :---: | :---: | :---: |
| 1D Poisson | [✓](scripts/poisson_1d_solution.py) | [✓](scripts/poisson_1d_residual.py) | [✓](scripts/poisson_1d_robustness.py) | — |
| 2D Poisson | [✓](scripts/poisson_2d_solution.py) | [✓](scripts/poisson_2d_residual.py) | [✓](scripts/poisson_2d_robustness.py) | — |
| 1D Helmholtz | [✓](scripts/helmholtz_1d_solution.py) | [✓](scripts/helmholtz_1d_residual.py) | [✓](scripts/helmholtz_1d_robustness.py) | — |
| Burgers (space–time) | [✓](scripts/burgers_1d_solution.py) | [✓](scripts/burgers_1d_residual.py) | [✓](scripts/burgers_1d_robustness.py) | — |
| KS (long rollout) | — | [✓](scripts/ks_residual.py) | — | — |
| SFNO (weather) | — | — | — | [✓](scripts/sfno_temperature.py) |

For SFNO:

```bash
uv sync --frozen --group sfno
uv run --group sfno python -m scripts.sfno_temperature
```

## Small examples and checks


```bash
uv run python -m examples.helmholtz_1d
uv run python -m examples.sfno
```

The Helmholtz demo generates its model and constant forcing input (`a(x)=1`)
in code. It certifies the residual, solution error, and robustness to an input
perturbation of size `0.001` on a 64-cell domain. Expected output is
approximately:

```text
Residual Linf upper: 1.0000318934200696
Solution Linf upper: 1.0000318934200696
Robustness cell-range width: 6.336254349409578e-08
```

See [examples](examples/README.md) for other API demonstrations.

## License

This project's code is available under the [MIT License](LICENSE). Third-party
code, model weights, and datasets are subject to their respective licenses and terms.

## Layout

- `lean_no/`: certification library, model loading, integration, and PDE bounds.
- `scripts/`: the 14 checkmark experiments; `_shared/` has reused helpers.
- `models/`: PDE and KS checkpoints.
- `training/`: PDE training and data generation.
- `examples/` and `tests/`: small API demonstrations and correctness checks. 
