import itertools
import math
import time
from dataclasses import dataclass, field

import torch

from lean_no.bounds.fusion import compose_affine
from lean_no.enclosures.bilinear import BilinearBounds
from lean_no.graph import execute
from lean_no.integration import integrate
from lean_no.propagation import bilinear

def full_symbol(weight, modes):
    extents = tuple(int(axis.abs().max()) for axis in modes)
    axes = [torch.arange(-m, m + 1, device=weight.device) for m in extents]
    frequencies = torch.stack(torch.meshgrid(*axes, indexing="ij")).flatten(1).T
    symbol = weight.new_zeros(*weight.shape[:2], len(frequencies))
    shape = tuple(2 * extent + 1 for extent in extents)
    strides = [math.prod(shape[a + 1 :]) for a in range(len(shape))]
    for indices in itertools.product(*(range(len(axis)) for axis in modes)):
        frequency = tuple(int(modes[a][i]) for a, i in enumerate(indices))
        value = weight[(slice(None), slice(None), *indices)]
        value = value * (0.5 if frequency[-1] == 0 else 1)
        positive = sum(
            (k + m) * stride for k, m, stride in zip(frequency, extents, strides)
        )
        negative = sum(
            (-k + m) * stride for k, m, stride in zip(frequency, extents, strides)
        )
        symbol[:, :, positive] += value
        symbol[:, :, negative] += value.conj()
    return symbol, frequencies

def cell_moments(values, frequencies):
    dimension = frequencies.shape[-1]
    shape = values.shape[-dimension:]
    spectrum = torch.fft.fftn(values, dim=tuple(range(-dimension, 0)), norm="forward")
    phase = frequencies.to(values) / values.new_tensor(shape)
    factor = torch.exp(-1j * math.pi * phase.sum(-1)) * torch.sinc(phase).prod(-1)
    indices = tuple(
        frequencies[..., axis].long().remainder(n) for axis, n in enumerate(shape)
    )
    return spectrum[(..., *indices)] * factor

@dataclass
class Response:

    kind: str
    channels: int
    parents: tuple = ()
    data: object = None

    def __post_init__(self):
        if self.kind in ("affine", "scale") and self.parents[0].kind == "zero":
            self.kind, self.parents, self.data = "zero", (), None
        if self.kind == "add":
            self.parents = tuple(
                parent for parent in self.parents if parent.kind != "zero"
            )
            if not self.parents:
                self.kind = "zero"
            elif len(self.parents) == 1:
                parent = self.parents[0]
                self.kind, self.parents, self.data = (
                    parent.kind,
                    parent.parents,
                    parent.data,
                )
        if self.kind == "affine" and self.parents[0].kind == "affine":
            self.data = self.data @ self.parents[0].data
            self.parents = self.parents[0].parents
        if self.kind == "scale" and self.parents[0].kind == "scale":
            self.data = self.data * self.parents[0].data
            self.parents = self.parents[0].parents

    def row(self, channel, count, shape, reference):
        result = reference.new_zeros(count, *shape)

        def backward(node, weights):
            if node.kind == "basis":
                result[node.data : node.data + node.channels] += weights
            elif node.kind == "affine":
                backward(
                    node.parents[0], torch.einsum("oi,o...->i...", node.data, weights)
                )
            elif node.kind == "scale":
                backward(node.parents[0], weights * node.data)
            elif node.kind == "add":
                for parent in node.parents:
                    backward(parent, weights)

        seed = reference.new_zeros(self.channels, *([1] * len(shape)))
        seed[channel] = 1
        backward(self, seed)
        # Break the recursive closure cycle so its dense GPU row is freed
        # immediately when the caller releases it, rather than waiting for GC.
        backward = None
        return result

    def moments(self, frequencies, count, shape, reference):
        layout = (*frequencies.shape[:-1],)
        if self.kind == "zero":
            return torch.zeros(
                self.channels,
                count,
                *layout,
                device=reference.device,
                dtype=torch.complex128,
            )
        if self.kind == "affine":
            values = self.parents[0].moments(frequencies, count, shape, reference)
            return torch.einsum("oi,ij...->oj...", self.data.to(values), values)
        if self.kind == "add":
            return sum(
                parent.moments(frequencies, count, shape, reference)
                for parent in self.parents
            )
        if self.kind == "basis":
            values = torch.zeros(
                self.channels,
                count,
                *layout,
                device=reference.device,
                dtype=torch.complex128,
            )
            ids = torch.arange(self.channels, device=reference.device)
            values[ids, ids + self.data] = (frequencies == 0).all(-1).to(values)
            return values
        if self.kind == "scale" and self.parents[0].kind == "basis":
            leaf = self.parents[0]
            values = torch.zeros(
                self.channels,
                count,
                *layout,
                device=reference.device,
                dtype=torch.complex128,
            )
            ids = torch.arange(self.channels, device=reference.device)
            values[ids, ids + leaf.data] = cell_moments(self.data, frequencies)
            return values
        return torch.stack(
            [
                cell_moments(self.row(i, count, shape, reference), frequencies)
                for i in range(self.channels)
            ]
        )

@dataclass
class FourierBasis:
    amplitude: torch.Tensor  # [source channel, independent input, *cells]
    frequencies: torch.Tensor

def basis_gram(left, right, shared=False):
    a, b = left.amplitude, right.amplitude
    k, l = left.frequencies, right.frequencies
    result = torch.empty(
        a.shape[1],
        len(a),
        len(k),
        len(b),
        len(l),
        device=a.device,
        dtype=torch.complex128,
    )
    difference = k[:, None] - l[None]
    for channel in range(a.shape[1]):
        if not shared:
            for i in range(len(a)):
                values = cell_moments(a[i, channel] * b[:, channel], difference)
                result[channel, i] = values.permute(1, 0, 2)
        else:
            nx, nt = a.shape[-2:]
            times_a, inverse_a = k[:, 1].unique(return_inverse=True)
            times_b, inverse_b = l[:, 1].unique(return_inverse=True)

            def temporal(value, times):
                v = torch.fft.fft(value, dim=-1, norm="forward")[
                    ..., times.remainder(nt)
                ]
                phase = times.to(value.real) / nt
                return v * torch.exp(-1j * math.pi * phase) * torch.sinc(phase)

            aa, bb = temporal(a[:, channel], times_a), temporal(b[:, channel], times_b)
            spatial = difference[..., 0]
            phase = spatial.to(a) / nx
            factor = torch.exp(-1j * math.pi * phase) * torch.sinc(phase)
            for i in range(len(a)):
                for t in range(len(times_a)):
                    rows = (inverse_a == t).nonzero().flatten()
                    products = aa[i, :, t][None, :, None] * bb.conj()
                    moments = torch.fft.fft(products, dim=1, norm="forward")
                    values = moments[:, spatial[rows].remainder(nx), inverse_b[None]]
                    result[channel, i, rows] = (values * factor[rows]).permute(1, 0, 2)
    return result.flatten(1, 2).flatten(2, 3)

def covariance_series(value, left, right, shape):
    difference = left[:, None] - right[None]
    index = sum(
        difference[..., a].remainder(n) * math.prod(shape[a + 1 :])
        for a, n in enumerate(shape)
    ).flatten()
    # [input, left channel, right channel, left mode, right mode]
    value = value.permute(0, 1, 3, 2, 4).flatten(-2)
    result = value.new_zeros(*value.shape[:-1], math.prod(shape))
    result.scatter_add_(-1, index.expand_as(value), value)
    return result.reshape(*result.shape[:-1], *shape)

@dataclass
class KernelState:

    shape: tuple
    inputs: int
    shared: bool = False
    bases: list = field(default_factory=list)
    maps: list = field(default_factory=list)
    frequencies: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    gram: torch.Tensor | None = None
    covariance: dict = field(default_factory=dict)

    def append(self, op, local, rows):
        weight, frequency = full_symbol(op.weight, op.data["modes"])
        basis = FourierBasis(local, frequency)
        blocks = [basis_gram(basis, previous, self.shared) for previous in self.bases]
        diagonal = basis_gram(basis, basis, self.shared)
        if self.gram is None:
            self.gram = diagonal
        else:
            cross = torch.cat(blocks, -1)
            self.gram = torch.cat(
                (
                    torch.cat((self.gram, cross.conj().transpose(-1, -2)), -1),
                    torch.cat((cross, diagonal), -1),
                ),
                -2,
            )
        previous = self.gram.shape[-1] - diagonal.shape[-1]
        mapping = weight.new_zeros(op.out_channels, len(frequency), self.gram.shape[-1])
        mapping[..., previous:] = torch.einsum(
            "oik,kl->okil",
            weight,
            torch.eye(len(frequency), device=weight.device, dtype=weight.dtype),
        ).flatten(-2)
        offset = 0
        for count, frequencies, old in zip(self.channels, self.frequencies, self.maps):
            difference = frequency[:, None] - frequencies[None]
            if isinstance(rows, Response):
                gates = rows.moments(difference, sum(self.channels), self.shape, local)[
                    :, offset : offset + count
                ]
            else:
                gates = cell_moments(rows[:, offset : offset + count], difference)
            response = torch.einsum("oik,ijkl->okjl", weight, gates).flatten(-2)
            mapping[..., : old.shape[-1]] += response @ old.flatten(0, 1)
            offset += count
        self.bases.append(basis)
        self.maps.append(mapping)
        self.frequencies.append(frequency)
        self.channels.append(op.out_channels)
        for j, old in enumerate(self.maps):
            cov = torch.einsum(
                "am,imn,bn->iab",
                mapping.flatten(0, 1),
                self.gram[..., : old.shape[-1]],
                old.flatten(0, 1).conj(),
            )
            cov = cov.reshape(
                self.inputs, op.out_channels, len(frequency), old.shape[0], old.shape[1]
            )
            self.covariance[len(self.maps) - 1, j] = cov

    def series(self, derivatives=()):
        dimension = len(self.shape)
        extent = torch.stack([f.abs().amax(0) for f in self.frequencies]).amax(0)
        shape = tuple(int(4 * n + 1) for n in extent)
        total = sum(self.channels)
        reference = self.maps[0]
        results = []
        for order in ((0,) * dimension, *derivatives):
            result = reference.new_zeros(self.inputs, total, total, *shape)
            starts = [0, *itertools.accumulate(self.channels)]
            for (i, j), value in self.covariance.items():
                left, right = self.frequencies[i], self.frequencies[j]
                multiplier = lambda f: (
                    torch.ones(len(f), device=f.device, dtype=torch.complex128)
                    * math.prod(
                        (2j * math.pi * f[:, a].double()) ** power
                        for a, power in enumerate(order)
                    )
                )
                value = (
                    value
                    * multiplier(left)[None, None, :, None, None]
                    * multiplier(right).conj()[None, None, None, None]
                )
                block = covariance_series(value, left, right, shape)
                result[:, starts[i] : starts[i + 1], starts[j] : starts[j + 1]] = block
                if i != j:
                    reverse = covariance_series(
                        value.conj().permute(0, 3, 4, 1, 2), right, left, shape
                    )
                    result[:, starts[j] : starts[j + 1], starts[i] : starts[i + 1]] = (
                        reverse
                    )
            results.append(result)
        return results

def energy_bound(coefficients, rows, shape):
    dimension = len(shape)
    spatial = tuple(range(-dimension, 0))
    reference = coefficients.real
    channels = rows.channels if isinstance(rows, Response) else len(rows)
    features = coefficients.shape[0]

    def row(channel):
        return (
            rows.row(channel, features, shape, reference)
            if isinstance(rows, Response)
            else rows[channel]
        )

    axes = [
        torch.fft.fftfreq(n, d=1 / n, device=reference.device, dtype=reference.dtype)
        for n in coefficients.shape[-dimension:]
    ]
    frequencies = torch.meshgrid(*axes, indexing="ij")
    distance = sum(math.pi * axis.abs() / n for axis, n in zip(frequencies, shape))
    remainder = (coefficients.abs() * distance.square()).sum(spatial) / 2
    phase = torch.exp(
        1j * sum(math.pi * axis / n for axis, n in zip(frequencies, shape))
    )
    indices = torch.meshgrid(
        *(axis.long().remainder(n) for axis, n in zip(axes, shape)), indexing="ij"
    )
    flat_index = sum(
        axis * math.prod(shape[j + 1 :]) for j, axis in enumerate(indices)
    ).flatten()
    result = reference.new_zeros(channels, *shape)
    for channel in range(channels):
        weights = row(channel).abs()
        result[channel] = (
            weights * torch.einsum("rs,s...->r...", remainder, weights)
        ).sum(0)
    for order in range(dimension + 1):
        quadratic = torch.zeros_like(result)
        multiplier = 1 if order == 0 else 2j * math.pi * frequencies[order - 1]
        for i in range(features):
            spectrum = coefficients.new_zeros(features, *shape)
            spectrum.flatten(1).scatter_add_(
                1,
                flat_index.expand(features, -1),
                (coefficients[i] * phase * multiplier).flatten(1),
            )
            values = torch.fft.ifftn(spectrum, dim=spatial, norm="forward").real
            del spectrum
            for channel in range(channels):
                weights = row(channel)
                quadratic[channel] += (values * weights).sum(0) * weights[i]
        result += quadratic if order == 0 else quadratic.abs() / (2 * shape[order - 1])
    return result.clamp_min(0).sqrt()

@dataclass
class LinearState:
    nominal: BilinearBounds
    local: torch.Tensor
    rows: Response
    remainder: torch.Tensor

def spectral_radius(radius, op, chunk):
    dimension = len(op.data["modes"])
    shape = radius.shape[-dimension:]
    spatial = tuple(range(-dimension, 0))
    spectrum = torch.fft.rfftn(radius, dim=spatial)
    result = radius.new_empty(op.out_channels, *shape)
    for start in range(0, op.out_channels, chunk):
        stop = min(start + chunk, op.out_channels)
        kernel = bilinear.series_bound(
            op.weight[start:stop], op.data["modes"], shape, tuple(1 / n for n in shape)
        )
        product = torch.einsum(
            "oi...,i...->o...", torch.fft.rfftn(kernel, dim=spatial), spectrum
        )
        result[start:stop] = (
            torch.fft.irfftn(product, shape, dim=spatial) / math.prod(shape)
        ).clamp_min(0)
    return result

def activation_bounds(name, lower, upper, radius, repetitions=1):
    from lean_no.bounds.activations import (
        gelu_derivative,
        gelu_range,
        relu_remainder,
    )

    midpoint = (lower + upper) / 2
    if name == "relu":
        slope = torch.where((lower < 0) & (upper > 0), 0.5, (midpoint > 0).to(lower))
        error = relu_remainder(lower, upper, radius, slope)
    elif name == "tanh":

        def slope_at(value):
            derivative = torch.ones_like(value)
            for _ in range(repetitions):
                value = value.tanh()
                derivative *= 1 - value.square()
            return derivative

        slope = slope_at(midpoint)
        left, right = lower - radius, upper + radius
        minimum = slope_at(torch.maximum(left.abs(), right.abs()))
        maximum = slope_at(torch.zeros_like(left).clamp(left, right))
        error = radius * torch.maximum((minimum - slope).abs(), (maximum - slope).abs())
    elif name == "gelu":
        slope = gelu_derivative(midpoint, 1)
        from lean_no.bounds.interval import Interval

        derivative = gelu_range(Interval(lower - radius, upper + radius), 1)
        low, high = derivative.lower, derivative.upper
        error = radius * torch.maximum((low - slope).abs(), (high - slope).abs())
    elif name == "identity":
        slope, error = torch.ones_like(lower), torch.zeros_like(lower)
    else:
        raise ValueError(f"no perturbation rule for {name}")
    return slope, error

@torch.no_grad()
def perturbation_bounds(
    graph,
    nominal,
    epsilon,
    channel_chunk=1,
    constant_in_time=False,
    integration=None,
    integration_options=None,
):
    graph = compose_affine(graph)
    shape = tuple(nominal.lower.shape[2:])
    dimension = len(shape)
    if dimension not in (1, 2, 3) or nominal.lower.shape[0] != 2**dimension:
        raise ValueError("one corner per side of each rectangular cell is required")
    if nominal.lower.dtype != torch.float64:
        raise ValueError("perturbation bounds require float64 cell envelopes")
    epsilon = torch.as_tensor(
        epsilon, device=nominal.lower.device, dtype=nominal.lower.dtype
    )
    inputs = nominal.lower.shape[1]
    if epsilon.ndim == 0:
        epsilon = torch.cat((epsilon[None], epsilon.new_zeros(inputs - 1)))
    if (
        epsilon.shape != (inputs,)
        or not torch.isfinite(epsilon).all()
        or (epsilon < 0).any()
    ):
        raise ValueError(
            "epsilon must be nonnegative, scalar or one bound per input channel"
        )
    active = epsilon > 0
    # A zero perturbation still uses one zero-weight input to retain nominal semantics.
    if not active.any():
        active[0] = True
    epsilon = epsilon[active]
    if constant_in_time and dimension != 2:
        raise ValueError("shared-time perturbations require a space-time domain")
    kernels = KernelState(shape, int(active.sum()), constant_in_time)
    local = torch.eye(inputs, device=epsilon.device, dtype=epsilon.dtype)[:, active]
    local = local.reshape(inputs, -1, *([1] * dimension)).expand(inputs, -1, *shape)
    source = LinearState(
        nominal, local, Response("zero", inputs), local.new_zeros(inputs, *shape)
    )
    options = dict(integration_options or {})
    if options.get("stages", "final") not in ("final", "all"):
        raise ValueError("kernel stages must be final or all")
    radii, records = [], []
    elapsed = 0.0
    kernels.discrete = True

    def linear_radius(state, method=None):
        nonlocal elapsed
        started = time.perf_counter()
        rows = state.rows
        result = (state.local.abs() * epsilon.reshape(1, -1, *([1] * dimension))).sum(1)
        if kernels.maps:
            if method is None:
                for channel, gram in enumerate(kernels.series()[0]):
                    result = result + epsilon[channel] * energy_bound(gram, rows, shape)
            else:
                from lean_no.integration.kernel_norm import LinearizedKernel

                source = LinearizedKernel(kernels, rows, epsilon)
                norm, record = integrate(source, method=method, options=options)
                records.append(record)
                result = result + norm
        elapsed += time.perf_counter() - started
        return result

    def apply(op, parents):
        parent = parents[0]
        nominal, local, rows, remainder = (
            parent.nominal,
            parent.local,
            parent.rows,
            parent.remainder,
        )
        if op.kind == "affine":
            nominal = bilinear.affine(nominal, op.weight, op.bias)
            local = torch.einsum("oi,ij...->oj...", op.weight, local)
            rows = Response("affine", op.out_channels, (rows,), op.weight)
            remainder = torch.einsum("oi,i...->o...", op.weight.abs(), remainder)
        elif op.kind == "spectral":
            nominal = bilinear.spectral(
                nominal, op.weight, op.data["modes"], op.bias, channel_chunk
            )
            kernels.append(op, local, rows)
            local = local.new_zeros(op.out_channels, len(epsilon), *shape)
            rows = Response(
                "basis", op.out_channels, data=sum(kernels.channels) - op.out_channels
            )
            remainder = spectral_radius(remainder, op, channel_chunk)
        elif op.kind == "add":
            nominal = BilinearBounds(
                sum(p.nominal.lower for p in parents),
                sum(p.nominal.upper for p in parents),
            )
            local = sum(p.local for p in parents)
            rows = Response("add", len(local), tuple(p.rows for p in parents))
            remainder = sum(p.remainder for p in parents)
        elif op.kind == "activation":
            kernels.discrete &= op.data["activation"] in ("relu", "identity")
            linear = linear_radius(
                parent, integration if options.get("stages") == "all" else None
            )
            radius = linear + remainder
            radii.append(float(radius.max()))
            slope, error = activation_bounds(
                op.data["activation"],
                *nominal.cell_bounds,
                radius,
                op.data.get("repetitions", 1),
            )
            for _ in range(op.data.get("repetitions", 1)):
                nominal = bilinear.activation(nominal, op.data["activation"])
            local = local * slope[:, None]
            rows = Response("scale", len(slope), (rows,), slope)
            remainder = slope.abs() * remainder + error
        else:
            raise NotImplementedError(f"no perturbation rule for {op.kind}")
        return LinearState(nominal, local, rows, remainder)

    result = execute(graph, source, apply)
    linear = linear_radius(result, integration)
    radius = linear + result.remainder
    output = BilinearBounds(
        result.nominal.lower - radius[None], result.nominal.upper + radius[None]
    )
    diagnostics = dict(
        implementation="graph",
        nominal_width=float(result.nominal.functional_width),
        linear_width=2 * float(linear.max()),
        remainder_width=2 * float(result.remainder.max()),
        preactivation_radii=radii,
        kernel_l2_seconds=elapsed if integration is None else 0.0,
        kernel_integration={
            "method": integration,
            "seconds": elapsed,
            "targets": records,
        }
        if records
        else {},
    )
    return output, diagnostics
