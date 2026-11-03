import math
import time

import torch

from lean_no.bounds.operators import is_identity
from lean_no.graph import Graph, execute, fourier
from lean_no.integration import integrate
from lean_no.integration.gaussian import AnalyticSource, GaussianOptions, gaussian_error
from lean_no.spectral.fourier import spectral_disk

def tanh_disk(center, radius):
    minimum = (center.abs() - radius).clamp_min(0)
    lower = torch.maximum(
        minimum.sinh(),
        torch.where(radius < math.pi / 2, minimum.cosh() * radius.cos(), 0.0),
    )
    if bool((lower <= 0).any()):
        raise ArithmeticError("tanh pole not excluded by disk cover")
    result = radius.sinh() / (lower * center.cosh())
    if not torch.isfinite(result).all():
        raise ArithmeticError("nonfinite analytic tanh disk")
    return center.tanh(), result

def source_disk(field, centers, radii, stop, layer_coefficients):
    center, radius = field.input_disks(centers, radii)
    if field.network.grid_embedding:
        center = torch.cat((center, centers), -1)
        radius = torch.cat((radius, radii[None].expand(len(centers), -1)), -1)
    real_only = bool((radii == 0).all())

    def apply(op, parents):
        c, r = parents[0]
        if op.kind == "spectral":
            return spectral_disk(layer_coefficients[op.name], centers, radii)
        if op.kind == "add":
            return tuple(a + b for a, b in zip(*parents))
        if op.kind == "affine":
            return c @ op.weight.T + op.bias, r @ op.weight.abs().T
        if is_identity(op.data["activation"]):
            return c, r
        if op.data["activation"] == "gelu":
            if not real_only:
                raise ValueError(
                    "Gaussian GELU coefficients require the strip source enclosure"
                )
            from lean_no.bounds.activations import gelu_first

            return torch.nn.functional.gelu(c), gelu_first * r
        if real_only:
            low, high = c - r, c + r
            center = (low.tanh() + high.tanh()) / 2
            stable = (2 * r.clamp(max=1)).sinh() / (2 * high.cosh() * low.cosh())
            return center, torch.where(r <= 1, stable, (high.tanh() - low.tanh()) / 2)
        return tanh_disk(c, r)

    return execute(Graph(field.network.ops[:stop]), (center, radius), apply)

@torch.no_grad()
def prepare_coefficients(
    field,
    order=16,
    batch=8192,
    progress=None,
    panels=None,
    method="gaussian",
    series=None,
    tolerance="1e-8",
    digits=30,
):
    analytic = hasattr(field.forcing, "data") and all(
        name in ("tanh", "identity") for name in field.network.activations
    )
    if method == "arb" and analytic:
        from lean_no.integration.arb import prepare_coefficients as arb_coefficients

        return arb_coefficients(
            field, series=series, tolerance=tolerance, digits=digits, progress=progress
        )
    if (
        type(order) is not int
        or order < (1 if method == "rectangle" else 2)
        or type(batch) is not int
        or batch < 1
    ):
        raise ValueError("Gaussian order >= 2 and positive batch required")
    if panels is not None and (type(panels) is not int or panels < 1):
        raise ValueError("positive panel count required")
    candidates = (16, 32, 64, 128, 256, 512) if panels is None else (panels,)
    ref = field.forcing.error
    dimension = len(field.periods)
    layer_coefficients, records = {}, []
    for op in field.network.ops:
        if op.kind != "spectral":
            continue
        started = time.perf_counter()
        weight, axes = op.weight, op.data["modes"]
        shape = tuple(len(axis) for axis in axes)
        if not analytic:
            proof = prepare_intervals(
                field, op, method, order, batch, layer_coefficients
            )
            layer_coefficients[op.name] = proof
            records.append(
                dict(
                    operator=op.name,
                    order=order,
                    panels=field.resolution,
                    seconds=time.perf_counter() - started,
                )
            )
            continue
        for count in candidates:
            rho = 2 / count
            radius = ref.new_full((dimension,), rho)
            cap = ref.new_zeros(weight.shape[1])
            try:
                for start in range(0, count**dimension, batch):
                    ids = torch.arange(
                        start, min(start + batch, count**dimension), device=ref.device
                    )
                    centers = torch.stack(
                        [
                            ((ids // count ** (dimension - a - 1)) % count + 0.5)
                            / count
                            for a in range(dimension)
                        ],
                        -1,
                    ).to(ref)
                    center, error = source_disk(
                        field, centers, radius, op.parents[0], layer_coefficients
                    )
                    cap = torch.maximum(cap, (center.abs() + error).amax(0))
                    if not torch.isfinite(cap).all():
                        raise ArithmeticError("nonfinite analytic cap")
                break
            except ArithmeticError as exc:
                if progress is not None:
                    progress(dict(operator=op.name, panels=count, refinement=str(exc)))
        else:
            raise ArithmeticError("analytic panel budget exhausted")
        phase = ref.new_zeros(shape)
        for axis, frequency in enumerate(axes):
            layout = [1] * dimension
            layout[axis] = len(frequency)
            phase += frequency.abs().reshape(layout)
        bound = cap.reshape(-1, *([1] * dimension)) * (2 * math.pi * rho * phase).exp()
        caps = (bound,) * dimension
        width = max(1, int(batch ** (1 / dimension)))
        chunks = (
            (min(batch, count * order),) if dimension == 1 else (width,) * dimension
        )

        def evaluate(*points):
            centers = torch.stack(torch.meshgrid(*points, indexing="ij"), -1).reshape(
                -1, dimension
            )
            center, radius = source_disk(
                field,
                centers,
                ref.new_zeros(dimension),
                op.parents[0],
                layer_coefficients,
            )
            layout = tuple(len(axis) for axis in points) + (-1,)
            return center.reshape(layout), radius.reshape(layout)

        value, radius = integrate(
            AnalyticSource(evaluate, caps, cap),
            axes,
            ((0.0, 1.0),) * dimension,
            method,
            GaussianOptions(order, (count,) * dimension, tuple(chunks)),
        )
        remainder = gaussian_error(caps, 1.0, order, method)
        inherited = (radius - remainder).flatten(1).amax(-1)
        coefficients = (weight * value[None]).sum(1)
        errors = (weight.abs() * radius[None]).sum(1)
        zero = tuple(int((axis == 0).nonzero()[0]) for axis in axes)
        coefficients[(slice(None), *zero)] += op.bias
        symmetry = torch.where(axes[-1] == 0, 1.0, 2.0)
        node = fourier(coefficients, axes, (errors * symmetry).flatten(1).sum(-1))
        node.data["coefficient_error"] = errors
        layer_coefficients[op.name] = node
        record = dict(
            operator=op.name,
            order=1 if method == "rectangle" else order,
            panels=count,
            nodes=count**dimension * (1 if method == "rectangle" else order**dimension),
            cap=float(cap.max()),
            gauss_remainder_max=float(remainder.max()),
            inherited_node_error_max=float(inherited.max()),
            coefficient_error_max=float(errors.max()),
            seconds=time.perf_counter() - started,
        )
        records.append(record)
        if progress is not None:
            progress(dict(phase="Fourier coefficients", **record))
    return layer_coefficients, records

def prepare_intervals(field, op, method, order, batch, coefficients):
    from dataclasses import replace

    from lean_no.integration import Integration
    from lean_no.propagation.derivative_cells import derivative_orders
    from lean_no.propagation.linear import Bounds

    dimension = len(field.periods)
    ref, shape = field.forcing.error, field.resolution
    units = tuple(
        tuple(int(a == b) for a in range(dimension)) for b in range(dimension)
    )
    orders = derivative_orders(dimension, 1)
    positions = [orders.index(unit) for unit in units]
    values, partials = [[], []], [[], []]
    for start in range(0, math.prod(shape), batch):
        ids = torch.arange(
            start, min(start + batch, math.prod(shape)), device=ref.device
        )
        left = torch.stack(
            [((ids // math.prod(shape[a + 1 :])) % n) / n for a, n in enumerate(shape)],
            -1,
        ).to(ref)
        halfwidth = left.new_tensor([0.5 / n for n in shape]).expand_as(left)
        node = field.enclose(
            left,
            torch.zeros_like(left),
            stop=op.parents[0],
            values_only=True,
            coefficients=coefficients,
        )[0]
        cell = field.enclose(
            left + halfwidth,
            halfwidth,
            stop=op.parents[0],
            orders=orders,
            coefficients=coefficients,
        )
        for j, endpoint in enumerate(("lower", "upper")):
            values[j].append(getattr(node, endpoint).T)
            partials[j].append(
                torch.stack(
                    [getattr(cell[index], endpoint).T for index in positions], 1
                )
            )
    nodes = [torch.cat(parts, -1).reshape(op.in_channels, *shape) for parts in values]
    derivatives = [
        torch.cat(parts, -1).reshape(op.in_channels, dimension, *shape)
        for parts in partials
    ]
    axes = tuple(
        torch.arange(n, device=ref.device, dtype=ref.dtype) * length / n
        for n, length in zip(shape, field.periods)
    )
    source = Bounds(axes, field.periods, *nodes, *derivatives)
    result = {}
    degree = max(1, max(map(sum, field.requested), default=0))
    rule = Integration(method, dict(order=order))
    for derivative in derivative_orders(dimension, degree):
        weight = op.weight
        for axis, power in enumerate(derivative):
            layout = [1] * dimension
            layout[axis] = -1
            weight = weight * (
                2j * math.pi * op.data["modes"][axis] / field.periods[axis]
            ).pow(power).reshape(layout)
        basis = replace(
            op,
            data={
                **op.data,
                "weight": weight,
                "bias": op.bias if not any(derivative) else torch.zeros_like(op.bias),
            },
        )
        result[derivative] = rule(source, basis, source.lengths)
    return result
