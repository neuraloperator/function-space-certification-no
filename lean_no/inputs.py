import itertools
from dataclasses import dataclass

import torch

from lean_no.adapters.fno import model_reference

@dataclass(frozen=True)
class InputSet:
    source: object
    epsilon: object = 0.0
    lipschitz: object = None
    curvature: object = None

    @property
    def nominal(self):
        from lean_no.function_space.analytic import describe

        return describe(self.source) if callable(self.source) else self.source

    @property
    def is_exact(self):
        if callable(self.source):
            return not callable(self.epsilon) and not bool(
                torch.as_tensor(self.epsilon).ne(0).any()
            )
        if hasattr(self.source, "is_exact"):
            return self.source.is_exact
        return getattr(self.source, "fixed_input", False) or not bool(
            torch.as_tensor(getattr(self.source, "epsilon", 0.0)).ne(0).any()
        )

    def prepare(self, model, grid, method):
        from lean_no.domain import DiscreteGrid, FourierGrid
        from lean_no.enclosures.grid import GridBounds
        from lean_no.function_space.analytic import Series, describe, evaluate
        from lean_no.function_space.input import FourierInput, InputBounds
        from lean_no.function_space.spherical import SphereGrid

        source, epsilon = self.source, self.epsilon
        lipschitz, curvature = self.lipschitz, self.curvature
        uncertain = callable(epsilon) or bool(torch.as_tensor(epsilon).ne(0).any())
        if not callable(source):
            if uncertain or lipschitz is not None or curvature is not None:
                raise ValueError(
                    "explicit input contracts already define their uncertainty"
                )
            if not isinstance(source, Series):
                return source
        else:
            if isinstance(grid, DiscreteGrid):
                if callable(epsilon):
                    raise ValueError(
                        "a radius function requires continuum interval robustness"
                    )
                reference = model_reference(model)
                axes = [
                    torch.arange(n, device=reference.device, dtype=reference.dtype) / n
                    for n in grid.resolution
                ]
                center = evaluate(source, torch.meshgrid(*axes, indexing="ij"))[
                    None, None
                ]
                return GridBounds(center - epsilon, center + epsilon)
            if method.strategy == "linearized":
                if callable(epsilon):
                    raise ValueError(
                        "a radius function requires continuum interval robustness"
                    )
                return _bilinear_band(
                    model,
                    grid,
                    source,
                    describe(source, channels=True),
                    epsilon,
                    not uncertain,
                )

        series = describe(source) if callable(source) else source
        if isinstance(grid, SphereGrid):
            from lean_no.adapters.sfno import SFNOInput

            if callable(epsilon):
                raise ValueError(
                    "a radius function requires continuum interval robustness"
                )
            if len(series.slopes) != 2 or len(model.channels) != 1:
                raise ValueError(
                    "spherical callable inputs require one scalar function of colatitude and longitude"
                )
            if any(series.slopes) or any(k[1] for k, _ in series.terms):
                raise ValueError(
                    "longitude-dependent spherical inputs require explicit SFNOInput regularity bounds"
                )
            if uncertain and lipschitz is None:
                raise ValueError(
                    "spherical robustness requires a full-input lipschitz bound"
                )
            sizes = grid.latitude.new_tensor(
                [[series.magnitude((0, 0))], [series.magnitude((1, 0))]]
            )

            def reference(points):
                return evaluate(source, (points[:, 0], points[:, 1]))[:, None]

            return SFNOInput(
                None,
                model.channels,
                epsilon,
                lipschitz,
                reference=reference,
                sizes=sizes,
            )
        if not isinstance(grid, FourierGrid):
            raise TypeError("analytic inputs require Fourier geometry")
        if callable(epsilon):
            radius = describe(epsilon)
            if len(radius.slopes) != 1 or any(radius.slopes):
                raise ValueError(
                    "radius must be a one-dimensional trigonometric function"
                )
            terms = {}
            for frequency, value in radius.terms:
                terms[frequency] = terms.get(frequency, 0j) + value
            minimum = terms.get((0.0,), 0j).real - sum(
                abs(value) for frequency, value in terms.items() if any(frequency)
            )
            if minimum < 0:
                raise ValueError(
                    "radius function must have a certified nonnegative lower bound"
                )
            if lipschitz is None:
                raise ValueError(
                    "continuum robustness requires a full-input lipschitz bound"
                )
            exact = series.bounds()
            value, spread = series.expression(), radius.expression()
            return InputBounds(
                (value - spread,),
                (value + spread,),
                exact.supremum + radius.magnitude((0,)),
                lipschitz,
                curvature,
            )
        if len(series.slopes) != len(grid.resolution):
            if len(series.slopes) != 1 or len(grid.resolution) != 2:
                raise ValueError(
                    "input function coordinates must match the Fourier grid"
                )
            if uncertain:
                raise ValueError(
                    "profile derivative fields currently require fixed inputs"
                )
            return series
        if (
            method.propagation == "ibp"
            and method.strategy != "linearized"
            and method.integration != "taylor_quadrature"
            and not uncertain
            and not any(series.slopes)
        ):
            reference = model_reference(model)
            return series.field(grid.lengths, reference.device, grid.dtype)
        exact = series.bounds()
        if not uncertain:
            return exact
        if lipschitz is None:
            raise ValueError(
                "continuum robustness requires an explicit full-input lipschitz bound; epsilon alone does not bound derivatives"
            )
        bounds = FourierInput(exact, epsilon, lipschitz).bounds()
        return InputBounds(
            bounds.lower_expr,
            bounds.upper_expr,
            bounds.supremum,
            bounds.lipschitz,
            curvature,
        )

def _bilinear_band(model, grid, function, series, epsilon, fixed):
    from lean_no.adapters.fno import build_graph
    from lean_no.enclosures.bilinear import BilinearBand, BilinearBounds

    vector = isinstance(series, tuple)
    series = series if vector else (series,)
    dimension = len(series[0].slopes)
    spatial = dimension == 1 and len(grid.resolution) == 2
    if dimension != len(grid.resolution) and not spatial:
        raise ValueError("input function coordinates must match the Fourier grid")
    graph = build_graph(model, grid.dtype)
    reference = model_reference(model)
    axes = [
        torch.arange(n, device=reference.device, dtype=grid.dtype) * length / n
        for n, length in zip(grid.resolution, grid.lengths)
    ]
    corners = []
    for sides in itertools.product((0, 1), repeat=len(grid.resolution)):
        coordinates = torch.meshgrid(
            *(
                axis + side * length / n
                for axis, side, length, n in zip(
                    axes, sides, grid.lengths, grid.resolution
                )
            ),
            indexing="ij",
        )
        values = function(*coordinates[:dimension])
        values = values if vector else (values,)
        channels = [
            torch.as_tensor(v, device=reference.device, dtype=grid.dtype).expand_as(
                coordinates[0]
            )
            for v in values
        ]
        if spatial:
            channels.append(coordinates[1])
        elif graph.grid_embedding:
            channels.extend(coordinates)
        corners.append(torch.stack(channels))
    center = torch.stack(corners)
    interpolation = center.new_zeros(center.shape[1])
    for channel, description in enumerate(series):
        interpolation[channel] = sum(
            description.magnitude(tuple(2 if a == b else 0 for a in range(dimension)))
            * (grid.lengths[b] / grid.resolution[b]) ** 2
            / 8
            for b in range(dimension)
        )
    radius = torch.zeros_like(center)
    if not fixed:
        radius = interpolation.reshape(1, -1, *([1] * len(grid.resolution))).expand_as(
            center
        )
    if fixed:
        epsilon = interpolation if vector else float(interpolation[0])
    elif isinstance(epsilon, (torch.Tensor, tuple, list)):
        epsilon = torch.as_tensor(epsilon, device=center.device, dtype=center.dtype)
        if epsilon.shape == (len(series),):
            epsilon = torch.cat(
                (epsilon, center.new_zeros(center.shape[1] - len(series)))
            )
    return BilinearBand(
        BilinearBounds(center - radius, center + radius),
        epsilon,
        constant_in_time=spatial,
        fixed_input=fixed,
    )
