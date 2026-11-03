import math

import torch

from lean_no.certificate import Certificate
from lean_no.certify import certify
from lean_no.domain import FourierGrid, grid_dimension
from lean_no.graph import Field
from lean_no.methods import Method
from lean_no.propagation.derivative_cells import DerivativeField
from lean_no.quantities import Residual
from lean_no.quantities.norms import ResidualResult
from lean_no.quantities.residual import residual_bound

def validate_points(points, dimension, batch_size):
    if (
        not isinstance(points, torch.Tensor)
        or points.ndim != 2
        or points.shape[1] != dimension
        or not len(points)
        or points.is_complex()
        or not torch.isfinite(points).all()
    ):
        raise ValueError("points must be finite real [N, dimension] coordinates")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

@torch.no_grad()
def query_bounds(certificate, points, batch_size):
    batches = (
        (points,)
        if getattr(certificate, "query_width", None) is not None
        else points.split(batch_size)
    )
    rows = []
    for part in batches:
        lower, upper = certificate.query(part)
        if (
            not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all()
            or (lower > upper).any()
        ):
            raise ArithmeticError("invalid sampled enclosure")
        rows.append((lower.detach().cpu(), upper.detach().cpu()))
    return tuple(torch.cat(parts) for parts in zip(*rows))

@torch.no_grad()
def sample_points(certificate, points, batch_size, residual=False):
    validate_points(points, grid_dimension(certificate.grid), batch_size)
    field = (
        certificate.field
        if isinstance(certificate, ResidualResult)
        else certificate.enclosure
    )
    jets = isinstance(field, DerivativeField) or (
        isinstance(field, Field) and field.metadata.get("basis") != "spherical"
    )
    if not jets:
        target = (
            field
            if isinstance(certificate, ResidualResult) and not residual
            else certificate
        )
        lower, upper = query_bounds(target, points, batch_size)
        return dict(
            values=(lower + upper) / 2,
            lower=lower,
            upper=upper,
            kind="enclosure_midpoint",
        )
    rows = []
    for part in points.split(batch_size):
        reference = (
            field.forcing.error if isinstance(field, DerivativeField) else field.error
        )
        centers = part.to(reference) / reference.new_tensor(field.periods)
        widths = torch.zeros_like(centers)
        if residual:
            from lean_no.quantities.residual import residual_form

            bounds = certificate.bounds
            coefficient = (
                bounds.burgers.viscosity if bounds.burgers is not None else bounds.kappa
            )
            form = residual_form(
                field,
                centers,
                widths,
                Residual(bounds.pde, coefficient),
                certificate.forcing,
            )
        else:
            form = field.enclose(centers, widths, values_only=True)[0]
        interval = form.actual
        rows.append(
            tuple(
                value.detach().cpu()
                for value in (form.center, interval.lower, interval.upper)
            )
        )
    values, lower, upper = (torch.cat(parts) for parts in zip(*rows))
    return dict(
        values=values,
        lower=lower,
        upper=upper,
        kind="nominal" if field.arithmetic == "affine" else "enclosure_midpoint",
    )

@torch.no_grad()
def witness(
    model, grid, inputs, points, method, chunk_size=64, batch_size=16384, **options
):
    if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
        raise ValueError("witness requires exactly two fixed inputs")
    validate_points(points, grid_dimension(grid), batch_size)
    pairs = []
    for candidate in inputs:
        certificate = certify(
            model=model,
            grid=grid,
            inputs=candidate,
            chunk_size=chunk_size,
            query=points
            if isinstance(grid, FourierGrid)
            and method.propagation in ("ibp", "crown")
            and method.strategy == "general"
            and method.integration in ("rectangle", "gaussian", "clenshaw_curtis")
            else None,
            **options,
            method=method,
        )
        pairs.append(
            tuple(
                value.double()
                for value in query_bounds(certificate, points, batch_size)
            )
        )
        del certificate
    (a, b), (c, d) = pairs
    estimate = ((a + b) / 2 - (c + d) / 2).abs()
    separation = torch.maximum(a - d, c - b).clamp_min(0)
    index, channel = divmod(int(separation.argmax()), separation.shape[1])
    best, best_channel = divmod(int(estimate.argmax()), estimate.shape[1])
    intervals = [
        [float(low[index, channel]), float(high[index, channel])] for low, high in pairs
    ]
    outputs = [
        dict(
            lower=float(low[best, best_channel]),
            upper=float(high[best, best_channel]),
            value=float((low[best, best_channel] + high[best, best_channel]) / 2),
            uncertainty=float(high[best, best_channel] - low[best, best_channel]),
        )
        for low, high in pairs
    ]
    return dict(
        estimate=float(estimate.max()),
        lower_bound=float(separation.max()),
        point=points[index].tolist(),
        channel=channel,
        intervals=intervals,
        estimate_point=points[best].tolist(),
        outputs=outputs,
    )

def sample_residual(
    model,
    grid=None,
    inputs=None,
    quantity=None,
    points=None,
    method=Method("ibp", "gaussian"),
    batch_size=16384,
    **options,
):
    if isinstance(model, (Certificate, ResidualResult)):
        certificate = model
    else:
        if not isinstance(quantity, Residual):
            raise ValueError("sample_residual requires a Residual quantity")
        validate_points(points, grid_dimension(grid), batch_size)
        certificate = residual_bound(
            certify(
                model=model,
                grid=grid,
                inputs=inputs,
                derivatives=2,
                **options,
                method=method,
            ),
            quantity,
        )
    if not certificate.quantity.endswith("_residual"):
        raise ValueError("sample_residual requires a residual certificate")
    return sample_points(certificate, points, batch_size, residual=True)

def reference_error(
    model,
    grid=None,
    inputs=None,
    points=None,
    reference=None,
    method=Method("ibp", "gaussian"),
    batch_size=16384,
    **options,
):
    if reference is None:
        raise ValueError("reference is required")
    if isinstance(model, (Certificate, ResidualResult)):
        certificate = model
    else:
        validate_points(points, grid_dimension(grid), batch_size)
        certificate = certify(
            model=model, grid=grid, inputs=inputs, **options, method=method
        )
    if not isinstance(certificate, ResidualResult) and certificate.quantity != "output":
        raise ValueError("reference_error requires an output field")
    samples = sample_points(certificate, points, batch_size)
    values = reference(*points.T) if callable(reference) else reference
    values = torch.as_tensor(values, dtype=samples["values"].dtype, device="cpu")
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != samples["values"].shape or not torch.isfinite(values).all():
        raise ValueError("reference must provide finite [N, channels] values")
    errors = samples["values"] - values
    uncertainty = torch.maximum(
        samples["values"] - samples["lower"], samples["upper"] - samples["values"]
    )
    return dict(
        errors=errors,
        sampled_linf_error=float(errors.abs().max()),
        maximum_surrogate_point_uncertainty=float(uncertainty.max()),
        certified=False,
    )

@torch.no_grad()
def sample_temperature(
    model,
    grid=None,
    inputs=None,
    points=None,
    channel=0,
    scale=1.0,
    offset=0.0,
    batch_size=64,
    **options,
):
    from lean_no.propagation.spherical import Field, evaluate, linear_field

    validate_points(points, 2, batch_size)
    if isinstance(model, Field):
        field = model
    elif isinstance(model, (Certificate, ResidualResult)):
        field = model.enclosure
    else:
        field = certify(
            model=model,
            grid=grid,
            inputs=inputs,
            **options,
            method=Method("ibp", "clenshaw_curtis"),
        ).enclosure
    if not isinstance(field, Field):
        raise TypeError("sample_temperature requires a spherical output field")
    if (
        not math.isfinite(scale)
        or not math.isfinite(offset)
        or type(channel) is not int
        or not 0 <= channel < len(field.error)
    ):
        raise ValueError(
            "channel, scale and offset must define a finite temperature conversion"
        )
    if ((points[:, 0] < 0) | (points[:, 0] > math.pi)).any():
        raise ValueError("colatitude must lie in [0, pi]")
    if len(field.error) != 1 or scale != 1 or offset != 0:
        weight = field.error.new_zeros(1, len(field.error))
        weight[0, channel] = scale
        field = linear_field(field, weight, field.error.new_tensor([offset]))
    return torch.cat(
        [
            evaluate(field, part[:, 0], part[:, 1], False, {})[:, 0].cpu()
            for part in points.split(batch_size)
        ]
    )
