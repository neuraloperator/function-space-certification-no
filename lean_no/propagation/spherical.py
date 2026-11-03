import math

import torch

from lean_no.bounds.normalization import instance_radius
from lean_no.bounds.positive import positive_map
from lean_no.bounds.spherical import spectral_radii
from lean_no.graph import execute, make_field, walk
from lean_no.integration import Integration
from lean_no.spectral.spherical import (
    gelu_first,
    gelu_sizes,
    harmonic_eval,
    harmonic_sizes,
    sfno_grid,
)

def harmonic_field(coefficients, error=None, chunk_size=16):
    # The imaginary zonal component contributes nothing to a real ISHT.
    coefficients = coefficients.clone()
    coefficients[:, :, 0] = coefficients[:, :, 0].real.clone()
    degree = torch.arange(coefficients.shape[1], device=coefficients.device)[:, None]
    order = torch.arange(coefficients.shape[2], device=coefficients.device)[None, :]
    coefficients[:, order > degree] = 0
    sizes = harmonic_sizes(coefficients)
    error = torch.zeros_like(sizes[0]) if error is None else error
    return make_field(
        "spectral",
        {"coefficients": coefficients},
        (),
        error,
        sizes,
        dict(basis="spherical", chunk_size=chunk_size),
    )

def constant_field(value, error=None, chunk_size=16):
    coefficients = torch.complex(value, torch.zeros_like(value))[:, None, None]
    return harmonic_field(coefficients * math.sqrt(4 * math.pi), error, chunk_size)

def constant_value(source):
    if source.kind == "spectral" and not torch.count_nonzero(source.sizes[1]):
        return source.data["coefficients"][:, 0, 0].real / math.sqrt(4 * math.pi)
    return None

def add_coefficients(a, b):
    values = a.new_zeros(
        (a.shape[0], max(a.shape[1], b.shape[1]), max(a.shape[2], b.shape[2]))
    )
    values[:, : a.shape[1], : a.shape[2]] += a
    values[:, : b.shape[1], : b.shape[2]] += b
    return values

def evaluate(source, latitude, longitude, cartesian, cache):
    latitude, longitude = latitude.to(source.sizes), longitude.to(source.sizes)

    def inputs(node, visit):
        points = (
            torch.stack(torch.meshgrid(latitude, longitude, indexing="ij"), -1)
            if cartesian
            else torch.stack((latitude, longitude), -1)
        )
        return (
            node.data["input"]
            .evaluate(points.reshape(-1, 2))
            .reshape(*points.shape[:-1], -1)
        )

    operations = {
        "input": inputs,
        "spectral": lambda n, v: harmonic_eval(
            n.data["coefficients"], latitude, longitude, cartesian, cache
        ),
        "affine": lambda n, v: v(n.parents[0]) @ n.data["weight"].T + n.data["bias"],
        "activation": lambda n, v: torch.nn.functional.gelu(v(n.parents[0])),
        "add": lambda n, v: v(n.parents[0]) + v(n.parents[1]),
        "concat": lambda n, v: torch.cat([v(p) for p in n.parents], dim=-1),
    }
    return walk(operations, cache)(source)

def linear_field(source, weight, bias):
    weight, bias = weight.to(source.sizes), bias.to(source.sizes)
    # Fuse consecutive maps before taking absolute values: shared dependencies survive.
    if source.kind in ("affine", "spectral") and "weight" in source.data:
        bias = bias + weight @ source.data["bias"]
        weight = weight @ source.data["weight"]
        source = source.parents[0]
    absolute = weight.abs()
    error = positive_map(source.error[None], absolute.T)[0]
    if source.kind == "spectral":
        coefficients = source.data["coefficients"]
        values = torch.einsum(
            "oi,ilm->olm", weight.to(coefficients.dtype), coefficients
        )
        values[:, 0, 0] += bias * math.sqrt(4 * math.pi)
        result = harmonic_field(values, error, source.chunk_size)
        result = result.with_parents((source,))
        result.data.update(weight=weight, bias=bias)
    else:
        sizes = positive_map(source.sizes, absolute.T)
        sizes[0] += bias.abs()
        result = make_field(
            "affine", dict(weight=weight, bias=bias), (source,), error, sizes
        )
    return result

def sum_field(left, right):
    if (
        left.kind in ("affine", "spectral")
        and right.kind in ("affine", "spectral")
        and "weight" in left.data
        and "weight" in right.data
        and left.parents[0] == right.parents[0]
    ):
        return linear_field(
            left.parents[0],
            left.data["weight"] + right.data["weight"],
            left.data["bias"] + right.data["bias"],
        )
    for source, mapped in ((left, right), (right, left)):
        if (
            mapped.kind in ("affine", "spectral")
            and "weight" in mapped.data
            and mapped.parents[0] == source
        ):
            identity = torch.eye(
                len(source.error), device=source.error.device, dtype=source.error.dtype
            )
            return linear_field(
                source, mapped.data["weight"] + identity, mapped.data["bias"]
            )
    if left.kind == right.kind == "spectral":
        a, b = left.data["coefficients"], right.data["coefficients"]
        values = add_coefficients(a, b)
        result = harmonic_field(values, left.error + right.error, left.chunk_size)
    else:
        order = min(len(left.sizes), len(right.sizes))
        result = make_field(
            "add",
            {},
            (left, right),
            left.error + right.error,
            left.sizes[:order] + right.sizes[:order],
        )
    return result

def gelu_field(source):
    error = gelu_first * source.error
    value = constant_value(source)
    if value is not None:
        result = constant_field(
            torch.nn.functional.gelu(value), error, source.chunk_size
        )
    else:
        result = make_field(
            "activation",
            {"activation": "gelu"},
            (source,),
            error,
            gelu_sizes(source.sizes),
        )
    return result

def field_stats(source, grid):
    value = constant_value(source)
    if value is not None:
        return value, torch.zeros_like(value)
    mean = torch.zeros_like(source.sizes[0])
    for latitude in grid.latitude.split(source.chunk_size):
        values = evaluate(source, latitude, grid.longitude, True, {})
        mean += values.sum((0, 1)) / (grid.shape[0] * grid.shape[1])
    variance = torch.zeros_like(mean)
    for latitude in grid.latitude.split(source.chunk_size):
        values = evaluate(source, latitude, grid.longitude, True, {})
        variance += (values - mean).square().sum((0, 1)) / (
            grid.shape[0] * grid.shape[1]
        )
    return mean, variance

def normalize_field(source, data):
    mean, variance = field_stats(source, data["grid"])
    weight, bias = data["weight"].to(mean), data["bias"].to(mean)
    centered = source.sizes[0] + mean.abs()
    coefficients = None
    if source.kind == "spectral":
        coefficients = source.data["coefficients"].clone()
        coefficients[:, 0, 0] -= mean * math.sqrt(4 * math.pi)
        centered = harmonic_sizes(coefficients)[0]
    base = torch.sqrt(variance + data["epsilon"])
    # Stabilized standard deviations are 1-Lipschitz in native RMS.
    error = instance_radius(source.error, centered, base, weight, data["epsilon"])
    lower, upper = (
        (variance.sqrt() - source.error).clamp(min=0).square(),
        (variance.sqrt() + source.error).square(),
    )
    scale = weight / base
    shift = bias - mean * scale
    if coefficients is not None:
        coefficients *= scale[:, None, None]
        coefficients[:, 0, 0] += bias * math.sqrt(4 * math.pi)
        result = harmonic_field(coefficients, error, source.chunk_size)
    else:
        sizes = source.sizes * scale.abs()
        sizes[0] = centered * scale.abs() + bias.abs()
        result = make_field(
            "affine",
            dict(weight=torch.diag(scale), bias=shift),
            (source,),
            error,
            sizes,
        )
    result.data["normalization"] = True
    return result, dict(
        mean=mean,
        variance=variance,
        variance_lower=lower,
        variance_upper=upper,
        centered_bound=centered,
        grid=list(data["grid"].shape),
    )

def spectral_field(
    source,
    data,
    grid,
    projection=False,
    moment_cache=None,
    rule=Integration("clenshaw_curtis"),
):
    degrees, orders = data["degrees"], data["orders"]
    value = constant_value(source)
    if value is not None and not torch.count_nonzero(source.error):
        if not projection:
            value = data["weight"][:, :, 0].real.to(value) @ value
        result = constant_field(value, chunk_size=source.chunk_size)
        return result, dict(analytic_integral=True, integration_radius=result.error)
    moment_cache = {} if moment_cache is None else moment_cache
    key = (source, "projection")
    geometry = (id(grid), degrees, orders)
    previous = moment_cache.pop(key, None)
    if previous is not None and previous[0] == geometry:
        _, moments, exact, remainder, _ = previous
    else:
        moments, exact, remainder = rule(
            source, (degrees, orders), grid, options=moment_cache
        )
    if projection:
        # Grid-changing residual and spectral branches integrate the same
        # nominal source. Reuse its moments without retaining a spatial field.
        # Keep the grid alive too: its Python id must not be recycled while
        # these moments are retained by a standalone projection call.
        moment_cache[key] = (geometry, moments, exact, remainder, grid)
    if projection:
        coefficients = moments
    else:
        weight = data["weight"]
        coefficients = moments.new_zeros((weight.shape[0], degrees, orders))
        for start in range(0, degrees, source.chunk_size):
            stop = min(degrees, start + source.chunk_size)
            for channel in range(0, weight.shape[0], source.chunk_size):
                end = min(weight.shape[0], channel + source.chunk_size)
                block = weight[channel:end, :, start:stop].to(moments)
                coefficients[channel:end, start:stop] = torch.einsum(
                    "oil,ilm->olm", block, moments[:, start:stop]
                )
    radius_data = {**data, "projection": projection}
    if not projection:
        radius_data["weight"] = data["weight"].to(moments)
    inherited, integration = spectral_radii(
        source.error, radius_data, remainder, source.chunk_size
    )
    error = inherited + integration
    result = harmonic_field(coefficients, error, source.chunk_size)
    return result, dict(analytic_integral=exact, integration_radius=integration)

def spherical_bounds(
    model,
    grid,
    inputs,
    chunk_size=16,
    progress=None,
    strip_radii=None,
    strip_shape=(256, 512),
    rule=Integration("clenshaw_curtis"),
):
    if grid.kind != "clenshaw-curtis":
        raise ValueError("Clenshaw-Curtis quadrature requires a Clenshaw-Curtis grid")
    expected = sfno_grid(
        len(grid.latitude),
        grid.latitude.device,
        grid.latitude.dtype,
        grid.gauss_nodes,
        rule="clenshaw-curtis",
        nlon=grid.nlon,
    )
    if (
        grid.kind != expected.kind
        or grid.shape != expected.shape
        or not torch.equal(grid.latitude, expected.latitude)
        or not torch.equal(grid.quadrature, expected.quadrature)
    ):
        raise ValueError("SFNO bounds require an unmodified sfno_grid(Q) quadrature")
    if (
        inputs.sizes.device != grid.latitude.device
        or inputs.sizes.dtype != grid.latitude.dtype
    ):
        raise ValueError("input and quadrature must share a device and dtype")
    radius = inputs.sizes.new_full(
        (len(inputs.channels),),
        inputs.epsilon,
    )
    source = (
        harmonic_field(inputs.coefficients, radius, chunk_size=chunk_size)
        if inputs.coefficients is not None
        else make_field(
            "input",
            {"input": inputs},
            (),
            radius,
            inputs.sizes[:2],
            dict(basis="spherical", chunk_size=chunk_size),
        )
    )
    diagnostics = []
    strip_geometry = {}
    moment_cache = {}

    def apply(op, parents):
        source, detail = parents[0], {}
        if op.kind == "affine":
            result = linear_field(source, op.data["weight"], op.data["bias"])
        elif op.kind == "activation":
            result = gelu_field(source)
            from lean_no.spectral.spherical_strip import attach_signed_strip

            attach_signed_strip(
                result,
                op.name,
                strip_geometry,
                radii=strip_radii,
                shape=strip_shape,
            )
        elif op.kind == "add":
            result = sum_field(*parents)
        elif op.kind == "concat":
            order = min(len(p.sizes) for p in parents)
            result = make_field(
                "concat",
                {},
                tuple(parents),
                torch.cat([p.error for p in parents]),
                torch.cat([p.sizes[:order] for p in parents], dim=1),
            )
        elif op.kind == "normalize":
            result, detail = normalize_field(source, op.data)
        elif op.kind == "spectral":
            result, detail = spectral_field(
                source,
                op.data,
                grid,
                op.data.get("projection", False),
                moment_cache=moment_cache,
                rule=rule,
            )
        else:
            raise NotImplementedError(op.kind)
        # Preserve overflowing radii as inf. Only undefined operations involving
        # an already infinite parent radius may be replaced by an unbounded radius.
        if torch.isnan(result.error).any():
            if not any(torch.isinf(parent.error).any() for parent in parents):
                raise FloatingPointError(f"invalid radius at {op.name}")
            result = result.with_error(
                torch.where(torch.isnan(result.error), math.inf, result.error)
            )
        if (result.error < 0).any():
            raise FloatingPointError(f"negative radius at {op.name}")
        if (
            not torch.isfinite(result.sizes[:2]).all()
            or torch.isnan(result.sizes).any()
        ):
            raise FloatingPointError(
                f"nonfinite certificate at {op.name}; no certificate reported"
            )
        diagnostics.append(
            dict(
                name=op.name,
                radius=result.error,
                **detail,
            )
        )
        if progress is not None:
            progress(diagnostics[-1])
        return result

    return execute(model.graph, source, apply), diagnostics
