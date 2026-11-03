import itertools
import math
import time
from dataclasses import dataclass

import torch

from lean_no.integration.rules import interval_rule

@dataclass(frozen=True)
class LinearizedKernel:
    kernels: object
    rows: torch.Tensor
    epsilon: torch.Tensor

def output_patches(rows, panels, discrete):
    shape = rows.shape[1:]
    dimension = len(shape)
    counts = (panels,) * dimension if isinstance(panels, int) else tuple(panels)
    if len(counts) != dimension or any(n % p or p < 1 for n, p in zip(shape, counts)):
        raise ValueError("output panels must divide each nominal grid axis")
    axes = [torch.arange(n, device=rows.device) for n in shape]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(
        -1, dimension
    )
    tile = sum(
        (coordinates[:, a] // (n // p)) * math.prod(counts[a + 1 :])
        for a, (n, p) in enumerate(zip(shape, counts))
    )
    if discrete:
        keys = torch.cat((rows.flatten(1).T, tile[:, None].to(rows)), 1)
        _, inverse = keys.unique(dim=0, return_inverse=True)
    else:
        inverse = tile
    groups = int(inverse.max()) + 1
    first = torch.full((groups,), len(tile), device=rows.device, dtype=torch.long)
    first.scatter_reduce_(
        0, inverse, torch.arange(len(tile), device=rows.device), reduce="amin"
    )
    lo = torch.full(
        (groups, dimension), max(shape), device=rows.device, dtype=torch.long
    )
    hi = torch.zeros_like(lo)
    lo.scatter_reduce_(
        0, inverse[:, None].expand_as(coordinates), coordinates, reduce="amin"
    )
    hi.scatter_reduce_(
        0, inverse[:, None].expand_as(coordinates), coordinates + 1, reduce="amax"
    )
    if discrete:
        values = rows.flatten(1)[:, first].T
        error = torch.zeros_like(values)
    else:
        low = rows.new_full((groups, len(rows)), math.inf)
        high = -low
        index = inverse[:, None].expand(len(tile), len(rows))
        low.scatter_reduce_(0, index, rows.flatten(1).T, reduce="amin")
        high.scatter_reduce_(0, index, rows.flatten(1).T, reduce="amax")
        values, error = (low + high) / 2, (high - low) / 2
    scale = rows.new_tensor(shape)
    return (
        (lo + hi).to(rows) / (2 * scale),
        (hi - lo).to(rows) / (2 * scale),
        values,
        error,
        inverse,
    )

def source_panels(basis, panels, shared):
    amplitude, frequency = basis.amplitude, basis.frequencies
    shape = amplitude.shape[2:]
    if shared:
        nt = shape[-1]
        times = frequency[:, -1]
        amplitude = torch.fft.fft(amplitude, dim=-1, norm="forward")[
            ..., times.remainder(nt)
        ]
        phase = times.to(amplitude.real) / nt
        amplitude *= torch.exp(-1j * math.pi * phase) * torch.sinc(phase)
        frequency = frequency[:, :1]
        shape = shape[:1]
    counts = (panels,) * len(shape) if isinstance(panels, int) else tuple(panels)
    if len(counts) != len(shape) or any(p < 1 or n % p for n, p in zip(shape, counts)):
        raise ValueError("source panels must divide each nominal grid axis")
    layout = (
        *amplitude.shape[:2],
        *(item for n, p in zip(shape, counts) for item in (p, n // p)),
    )
    if shared:
        layout = (*layout, amplitude.shape[-1])
    block = amplitude.reshape(layout)
    axes = tuple(3 + 2 * a for a in range(len(shape)))
    if shared:
        low, high = block.real.amin(axes), block.real.amax(axes)
        ilo, ihi = block.imag.amin(axes), block.imag.amax(axes)
        center = (low + high) / 2 + 0.5j * (ilo + ihi)
        radius = ((high - low).square() + (ihi - ilo).square()).sqrt() / 2
    else:
        low, high = block.amin(axes), block.amax(axes)
        center, radius = (low + high) / 2, (high - low) / 2
    center = (
        center.reshape(*center.shape[:2], -1, len(frequency))
        if shared
        else center.flatten(2)
    )
    radius = radius.reshape_as(center)
    axes = [
        (torch.arange(p, device=amplitude.device, dtype=amplitude.real.dtype) + 0.5) / p
        for p in counts
    ]
    points = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).flatten(0, -2)
    return center, radius, points, frequency, counts

def coefficients_at(kernels, centers, rows):
    result = kernels.maps[-1].new_zeros(len(centers), kernels.gram.shape[-1])
    offset = 0
    for mapping, frequencies, count in zip(
        kernels.maps, kernels.frequencies, kernels.channels
    ):
        phase = torch.exp(2j * math.pi * (centers @ frequencies.to(centers).T))
        coefficients = (
            rows[:, offset : offset + count, None] * phase[:, None]
        ).flatten(1)
        result[:, : mapping.shape[-1]] += coefficients.to(mapping) @ mapping.flatten(
            0, 1
        )
        offset += count
    return result

def patch_energy(coefficients, centers, radii, rows, uncertain=False):
    dimension = centers.shape[1]
    axes = [
        torch.fft.fftfreq(n, d=1 / n, device=centers.device, dtype=centers.dtype)
        for n in coefficients.shape[-dimension:]
    ]
    frequencies = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(
        -1, dimension
    )
    coefficients = coefficients.flatten(-dimension)
    phase = torch.exp(2j * math.pi * (centers @ frequencies.T))
    distance = 2 * math.pi * (radii @ frequencies.abs().T)
    if uncertain:
        matrix = torch.einsum("rsk,bk->brs", coefficients, phase).real.abs()
        for axis in range(dimension):
            matrix += (
                radii[:, axis, None, None]
                * torch.einsum(
                    "rsk,bk->brs",
                    coefficients,
                    phase * (2j * math.pi * frequencies[:, axis]),
                ).real.abs()
            )
        matrix += 0.5 * torch.einsum(
            "rsk,bk->brs", coefficients.abs(), distance.square()
        )
        return torch.einsum("br,brs,bs->b", rows, matrix, rows).clamp_min(0).sqrt()
    energy = torch.einsum(
        "br,rsk,bs->bk", rows.to(coefficients), coefficients, rows.to(coefficients)
    )
    value = (energy * phase).sum(-1).real
    for axis in range(dimension):
        value += (
            radii[:, axis]
            * (energy * phase * (2j * math.pi * frequencies[:, axis]))
            .sum(-1)
            .real.abs()
        )
    value += 0.5 * (energy.abs() * distance.square()).sum(-1)
    return value.clamp_min(0).sqrt()

def panel_rule(frequencies, centers, width, method, order):
    nodes, weights, distance = interval_rule(method, order)
    ref = centers
    ids = list(itertools.product(range(len(nodes)), repeat=centers.shape[1]))
    offsets = ref.new_tensor(
        [[nodes[i] - 0.5 for i in item] for item in ids]
    ) * torch.as_tensor(width, device=ref.device, dtype=ref.dtype)
    weights = ref.new_tensor(
        [math.prod(float(weights[i]) for i in item) for item in ids]
    )
    points = centers[:, None] + offsets[None]
    phase = torch.exp(
        -2j * math.pi * torch.einsum("pqd,kd->pqk", points, frequencies.to(ref))
    )
    center_phase = torch.exp(-2j * math.pi * (centers @ frequencies.to(ref).T))
    return phase, center_phase, weights, distance

def polynomial_integral(
    coefficients, frequencies, centers, width, method, order, analytic, rule=None
):
    phase, center_phase, weights, distance = (
        panel_rule(frequencies, centers, width, method, order) if rule is None else rule
    )
    values = torch.einsum("bpk,pqk->bpq", coefficients, phase).real
    quadrature = torch.einsum("bpq,q->bp", values.abs(), weights)
    frequency = 2 * math.pi * frequencies.to(centers).abs()
    lipschitz = torch.einsum("bpk,kd->bpd", coefficients.abs(), frequency)
    middle = torch.einsum("bpk,pk->bp", coefficients, center_phase).real
    width = centers.new_tensor(width).expand(centers.shape[1])
    sign = middle.abs() > (width / 2 * lipschitz).sum(-1)
    zero = (frequencies == 0).all(-1)
    constant = coefficients[..., zero].sum(-1).real.abs()
    sign |= constant > coefficients[..., ~zero].abs().sum(-1)
    sign &= analytic
    first_order = distance * (width * lipschitz).sum(-1)
    degree = (
        2 * order
        if method == "gaussian"
        else order
        if method == "clenshaw_curtis"
        else 1
    )
    caps = torch.einsum(
        "bpk,kd->bpd", coefficients.abs(), torch.exp(2 * width * frequency)
    )
    cauchy = 2 / 0.75 * 0.25**degree * caps.sum(-1)
    error = torch.where(sign, torch.minimum(first_order, cauchy), first_order)
    volume = float(width.prod())
    return volume * (quadrature + error), dict(
        sign_certified=int(sign.sum()),
        cauchy_used=int((sign & (cauchy < first_order)).sum()),
        quadrature_error=float(volume * error.sum(-1).max()),
    )

@torch.no_grad()
def kernel_norm(source, method, options=None):
    options = dict(options or {})
    panels = options.get(
        "panels",
        tuple(
            min(16, n)
            for n in source.kernels.shape[
                : 1 if source.kernels.shared else len(source.kernels.shape)
            ]
        ),
    )
    output_panels = options.get(
        "output_panels", tuple(min(16, n) for n in source.kernels.shape)
    )
    order, batch = options.get("order", 4), options.get("batch", 32)
    if type(batch) is not int or batch < 1:
        raise ValueError("positive kernel batch required")
    started = time.perf_counter()
    kernels, shape = source.kernels, source.kernels.shape
    dimension = len(shape)
    units = tuple(
        tuple(int(i == j) for i in range(dimension)) for j in range(dimension)
    )
    grams = kernels.series(units)
    sources = [source_panels(basis, panels, kernels.shared) for basis in kernels.bases]
    all_frequencies = torch.cat([data[3] for data in sources])
    frequencies, inverse = all_frequencies.unique(dim=0, return_inverse=True)
    points, counts = sources[0][2], sources[0][4]
    widths = tuple(1 / n for n in counts)
    rule = panel_rule(frequencies, points, widths, method, order)
    batch = min(batch, max(1, 2**22 // (len(points) * len(frequencies))))
    from lean_no.propagation.perturbation import Response

    channels = (
        source.rows.channels if isinstance(source.rows, Response) else len(source.rows)
    )
    output = source.epsilon.new_empty(channels, *shape)
    stats = dict(
        sign_certified=0,
        cauchy_used=0,
        amplitude_uncertainty=0.0,
        output_remainder=0.0,
        row_uncertainty=0.0,
    )
    records = []
    for channel in range(channels):
        row_field = (
            source.rows.row(channel, sum(kernels.channels), shape, source.epsilon)
            if isinstance(source.rows, Response)
            else source.rows[channel]
        )
        centers, radii, rows, error, groups = output_patches(
            row_field, output_panels, getattr(kernels, "discrete", False)
        )
        result = rows.new_zeros(len(rows))
        for start in range(0, len(rows), batch):
            stop = min(start + batch, len(rows))
            coefficients = coefficients_at(
                kernels, centers[start:stop], rows[start:stop]
            )
            value = rows.new_zeros(stop - start)
            for input_channel, epsilon in enumerate(source.epsilon):
                polynomial = coefficients.new_zeros(
                    stop - start, len(points), len(frequencies)
                )
                inherited = rows.new_zeros(stop - start, len(points))
                offset = mode_offset = 0
                for basis, data in zip(kernels.bases, sources):
                    amplitude, uncertainty, _, modes, _ = data
                    count, size = len(basis.amplitude), len(basis.frequencies)
                    terms = coefficients[:, offset : offset + count * size].reshape(
                        stop - start, count, size
                    )
                    if kernels.shared:
                        part = torch.einsum(
                            "bik,ipk->bpk", terms, amplitude[:, input_channel]
                        )
                        inherited += torch.einsum(
                            "bik,ipk->bp", terms.abs(), uncertainty[:, input_channel]
                        )
                    else:
                        part = torch.einsum(
                            "bik,ip->bpk", terms, amplitude[:, input_channel].to(terms)
                        )
                        inherited += torch.einsum(
                            "bik,ip->bp", terms.abs(), uncertainty[:, input_channel]
                        )
                    polynomial.index_add_(
                        -1, inverse[mode_offset : mode_offset + size], part
                    )
                    offset += count * size
                    mode_offset += size
                integrals, record = polynomial_integral(
                    polynomial,
                    frequencies,
                    points,
                    widths,
                    method,
                    order,
                    inherited == 0,
                    rule,
                )
                norm = integrals.sum(-1) + inherited.mean(-1)
                variation = torch.zeros_like(norm)
                for axis in range(dimension):
                    variation += radii[start:stop, axis] * patch_energy(
                        grams[axis + 1][input_channel],
                        centers[start:stop],
                        radii[start:stop],
                        rows[start:stop],
                    )
                row_error = (
                    patch_energy(
                        grams[0][input_channel],
                        centers[start:stop],
                        radii[start:stop],
                        error[start:stop],
                        True,
                    )
                    if error[start:stop].any()
                    else 0
                )
                value += epsilon * (norm + variation + row_error)
                stats["sign_certified"] += record["sign_certified"]
                stats["cauchy_used"] += record["cauchy_used"]
                stats["amplitude_uncertainty"] = max(
                    stats["amplitude_uncertainty"], float(inherited.mean(-1).max())
                )
                stats["output_remainder"] = max(
                    stats["output_remainder"], float(variation.max())
                )
            result[start:stop] = value
            if options.get("progress"):
                options["progress"](
                    dict(
                        stage="linearized",
                        channel=channel,
                        target=channel + 1,
                        targets=channels,
                        completed=stop,
                        total=len(rows),
                    )
                )
        output[channel] = result[groups].reshape(shape)
        records.append(dict(channel=channel, output_patches=len(rows)))
    stats.update(
        method=method,
        order=order,
        panels=panels,
        output_panels=output_panels,
        output_patches=sum(record["output_patches"] for record in records),
        targets=records,
        source_nodes=len(points)
        * len(interval_rule(method, order)[0]) ** points.shape[1],
        seconds=time.perf_counter() - started,
    )
    return output, stats
