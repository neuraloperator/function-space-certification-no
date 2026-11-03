# Small API examples

Run from the repository root after `uv sync --frozen --group dev`.
For example: `uv run python -m examples.helmholtz_1d`.

| Module | Demonstrates |
| --- | --- |
| `examples.grid` | Discrete IBP and CROWN on the included Poisson checkpoint |
| `examples.quadrature` | Continuum forward, IBP, and CROWN on that checkpoint |
| `examples.poisson_2d` | Poisson residual, solution error, and robustness |
| `examples.helmholtz_1d` | Helmholtz residual and solution error |
| `examples.burgers_1d` | Space–time Burgers residual and solution error |
| `examples.ks` | KS residuals and rollout handoffs |
| `examples.sfno` | A small spherical operator and regional temperature maximum |
| `examples.arb` | Optional Arb pointwise bounds |
| `examples.arb_residual` | Arb residual and solution checks on trained 1D checkpoints |

The PDE, KS, and SFNO demonstrations use small deterministic models; SFNO needs
neither CUDA, weather downloads, nor the production SFNO dependencies. Some
models have zero spectral weights to keep demonstrations inexpensive. Trained
paper experiments are in `scripts/` and linked from the root README.

`certify(model, grid, inputs, *, method=Method(...), ...)` encloses a field and
requested derivatives. Pass a callable for a fixed input, or
`InputSet(source, epsilon=..., lipschitz=...)` for uncertainty. Use
`residual_bound` and `solution_bound` for PDE quantities.

Certificate widths are named by their meaning: `pointwise_width`,
`cell_range_width`, `node_width`, and `query_width`. Each has one value per
output channel, or is `None` when not computed. Use `.amax()` for the largest
channel width. Cell-range widths also include variation within each cell;
`certificate.query(points).width` gives widths at the specified points.
Taylor, Gaussian, rectangle, and Arb integration remain available where
supported. All bounds exclude floating-point roundoff.
