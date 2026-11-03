import math

import torch

from lean_no.function_space.spherical import sphere_grid
from lean_no.spectral.spherical import (
    gelu_linear,
    gelu_second,
    gelu_third,
    legendre_basis,
)

def nominal_affine_terms(source):
    terms = {}
    bias = torch.zeros_like(source.error)

    def visit(node, weight):
        nonlocal bias
        if node.kind == "affine":
            bias = bias + weight @ node.data["bias"]
            visit(node.parents[0], weight @ node.data["weight"])
        elif node.kind == "add":
            for parent in node.parents:
                visit(parent, weight)
        elif node.kind == "concat":
            for parent, block in zip(
                node.parents, weight.split([len(p.error) for p in node.parents], dim=1)
            ):
                visit(parent, block)
        else:
            terms[node] = weight if node not in terms else terms[node] + weight

    visit(
        source,
        torch.eye(
            len(source.error), dtype=source.sizes.dtype, device=source.sizes.device
        ),
    )
    return terms, bias

def complex_derivative(order, height, real, real_radius):
    # G'''=(z^3-4z) phi(z), G''''=(-z^4+7z^2-4) phi(z).
    # On a real interval, |x|^j exp(-x^2/2) attains its maximum at
    # |x|=sqrt(j), clipped to the interval of possible absolute values.
    # Keep the Gaussian's real decay when bounding each polynomial term.
    powers = {3: {3: 1, 1: 4}, 4: {4: 1, 2: 7, 0: 4}}[order]
    lower = (real.abs() - real_radius).clamp(min=0)
    upper = real.abs() + real_radius
    result = torch.zeros_like(height)
    for power, coefficient in powers.items():
        for j in range(power + 1):
            witness = torch.clamp(lower.new_tensor(math.sqrt(j)), min=lower, max=upper)
            moment = witness**j * ((height.square() - witness.square()) / 2).exp()
            result += coefficient * math.comb(power, j) * moment * height ** (power - j)
    return result / math.sqrt(2 * math.pi)

def gelu_rectangle(real, imaginary, real_radius, imaginary_radius):
    gaussian = (-real.square() / 2).exp() / math.sqrt(2 * math.pi)
    derivative = 0.5 * (1 + torch.erf(real / math.sqrt(2))) + real * gaussian
    second = (2 - real.square()) * gaussian
    slope, _, spread = gelu_linear(real, real_radius)
    gain = slope.abs() + spread
    variation = spread + (slope - derivative).abs()
    height = imaginary.abs() + imaginary_radius
    square_gap = imaginary_radius * (2 * imaginary.abs() + imaginary_radius)
    center_real = torch.nn.functional.gelu(real) - 0.5 * imaginary.square() * second
    center_imaginary = imaginary * derivative
    error_real = gain * real_radius + 0.5 * gelu_second * square_gap
    error_real += (
        0.5
        * imaginary.square()
        * torch.minimum(
            gelu_third * real_radius, real_radius.new_tensor(2 * gelu_second)
        )
    )
    error_real += complex_derivative(4, height, real, real_radius) * height**4 / 24
    error_imaginary = gain * imaginary_radius + variation * imaginary.abs()
    error_imaginary += complex_derivative(3, height, real, real_radius) * height**3 / 6
    return center_real, center_imaginary, error_real, error_imaginary

def theta_derivative(basis):
    degrees, orders, _ = basis.shape
    degree = torch.arange(degrees, device=basis.device, dtype=basis.real.dtype)[:, None]
    mode = torch.arange(orders, device=basis.device, dtype=basis.real.dtype)[None]
    up = ((degree - mode) * (degree + mode + 1)).clamp(min=0).sqrt()
    down = ((degree + mode) * (degree - mode + 1)).clamp(min=0).sqrt()
    higher = torch.cat((basis[:, 1:], torch.zeros_like(basis[:, :1])), dim=1)
    lower = torch.cat((-basis[:, 1:2], basis[:, :-1]), dim=1)
    return 0.5 * (up[..., None] * higher - down[..., None] * lower)

def complex_legendre(theta, degrees, orders):
    sine, cosine = theta.sin(), theta.cos()
    basis = theta.new_zeros((degrees, orders, theta.numel()))
    basis[0, 0] = 1 / math.sqrt(4 * math.pi)
    for l in range(1, degrees):
        if l < orders:
            basis[l, l] = -math.sqrt((2 * l + 1) / (2 * l)) * sine * basis[l - 1, l - 1]
        if l - 1 < orders:
            basis[l, l - 1] = math.sqrt(2 * l + 1) * cosine * basis[l - 1, l - 1]
        m = torch.arange(
            min(l - 1, orders), device=theta.device, dtype=theta.real.dtype
        )
        first = ((4 * l * l - 1) / (l * l - m * m)).sqrt()
        second = (
            ((2 * l + 1) * ((l - 1) ** 2 - m * m)) / ((2 * l - 3) * (l * l - m * m))
        ).sqrt()
        basis[l, : len(m)] = (
            first[:, None] * cosine * basis[l - 1, : len(m)]
            - second[:, None] * basis[l - 2, : len(m)]
        )
    return basis

def harmonic_strip_boxes(
    latent, radii, shape, order, coordinate_only=None, cache=None, latitude_batch=None
):
    cache = {} if cache is None else cache
    if latitude_batch is not None and latitude_batch < 1:
        raise ValueError("strip latitude batch must be positive")
    coefficients = latent.data["coefficients"]
    degrees, orders = coefficients.shape[-2:]
    if degrees < 2 or orders != degrees:
        raise ValueError(
            "signed strip boxes require complete harmonic orders and degree >= 1"
        )
    if shape[1] < 2 * orders - 1:
        raise ValueError("signed strip longitude grid aliases retained orders")
    grid = sphere_grid(shape, dtype=latent.error.dtype, device=latent.error.device)
    edges = grid.edges
    ht = torch.maximum(grid.latitude - edges[:-1], edges[1:] - grid.latitude)
    hp = math.pi / grid.nlon
    h = ht + hp
    modes = torch.arange(orders, device=coefficients.device, dtype=latent.error.dtype)
    factors = latent.error.new_full((orders,), 2.0)
    factors[0] = 1
    l = torch.arange(degrees, device=coefficients.device, dtype=latent.error.dtype)
    amplitude = (coefficients.abs().square() * factors).sum(-1).sqrt()
    amplitude *= ((2 * l + 1) / (4 * math.pi)).sqrt()
    tail_real = (torch.cosh(radii[:, None] * l) * l ** (order + 1)) @ amplitude.T
    tail_imag = (torch.sinh(radii[:, None] * l) * l ** (order + 1)) @ amplitude.T
    for coordinate in (0, 1) if coordinate_only is None else (coordinate_only,):
        if coordinate == 1:
            weighted = coefficients.abs().square() * factors
            scale = ((2 * l + 1) / (4 * math.pi)).sqrt() * l ** (order + 1)
            re = (
                (
                    weighted[None]
                    * torch.cosh(radii[:, None] * modes)[:, None, None] ** 2
                )
                .sum(-1)
                .sqrt()
            )
            im = (
                (
                    weighted[None]
                    * torch.sinh(radii[:, None] * modes)[:, None, None] ** 2
                )
                .sum(-1)
                .sqrt()
            )
            tail_real = (re * scale).sum(-1)
            tail_imag = (im * scale).sum(-1)
        step = shape[0] if latitude_batch is None else latitude_batch
        for start in range(0, shape[0], step):
            stop = min(start + step, shape[0])
            # Streamed rows do not retain bases for previous latitude chunks.
            row_cache = cache if latitude_batch is None else {}
            key = (
                degrees,
                orders,
                shape,
                start,
                stop,
                tuple(radii.cpu().tolist()),
                coordinate,
                order,
                coefficients.device,
                coefficients.dtype,
            )
            if key not in row_cache:
                theta = grid.latitude[start:stop]
                if coordinate == 0:
                    theta = (theta[:, None] + 1j * radii).flatten()
                    basis = complex_legendre(theta, degrees, orders)
                else:
                    basis = legendre_basis(theta, degrees, orders)
                bases = [basis]
                for a in range(order):
                    bases.append(theta_derivative(bases[-1]))
                row_cache[key] = bases
            bases = row_cache[key]
            error_real = latent.error.new_zeros(
                ((stop - start), shape[1], len(radii), len(latent.error))
            )
            error_imag = torch.zeros_like(error_real)
            center = None
            for a in range(order + 1):
                basis = bases[a].to(coefficients)
                positive = torch.einsum("clm,lmt->ctm", coefficients, basis)
                negative = torch.einsum("clm,lmt->ctm", coefficients.conj(), basis)
                if coordinate == 0:
                    positive = positive.reshape(
                        len(latent.error), (stop - start), len(radii), orders
                    )
                    negative = negative.reshape(
                        len(latent.error), (stop - start), len(radii), orders
                    )
                else:
                    phase = (radii[:, None] * modes).exp()
                    positive = positive[:, :, None] / phase[None, None]
                    negative = negative[:, :, None] * phase[None, None]
                for b in range(order + 1 - a):
                    plus = positive * (1j * modes) ** b
                    minus = negative * (-1j * modes) ** b
                    re_modes = (plus + minus.conj()) / 2
                    im_modes = (plus - minus.conj()) / (2j)
                    real = torch.fft.irfft(
                        re_modes, n=grid.nlon, norm="forward"
                    ).permute(1, 3, 2, 0)
                    imaginary = torch.fft.irfft(
                        im_modes, n=grid.nlon, norm="forward"
                    ).permute(1, 3, 2, 0)
                    if a == b == 0:
                        center = real, imaginary
                    else:
                        factor = (
                            ht[start:stop, None, None, None] ** a
                            * hp**b
                            / (math.factorial(a) * math.factorial(b))
                        )
                        error_real += real.abs() * factor
                        error_imag += imaginary.abs() * factor
            remainder = h[start:stop, None, None, None] ** (order + 1) / math.factorial(
                order + 1
            )
            error_real += remainder * tail_real[None, None]
            error_imag += remainder * tail_imag[None, None]
            yield tuple(
                value.reshape(-1, len(radii), len(latent.error))
                for value in (*center, error_real, error_imag)
            )

def signed_pointwise_strip(
    source,
    radii,
    shape=(256, 512),
    order=12,
    coordinate_only=None,
    cache=None,
    point_batch=16384,
    radius_batch=None,
    latitude_batch=None,
):
    if radius_batch is not None:
        if radius_batch < 1:
            raise ValueError("strip radius batch must be positive")
        if len(radii) > radius_batch:
            # Retain only the small caps between batches. Caching the full
            # harmonic Taylor bases would keep every radius batch alive.
            return torch.cat(
                [
                    signed_pointwise_strip(
                        source,
                        batch,
                        shape,
                        order,
                        coordinate_only,
                        point_batch=point_batch,
                        latitude_batch=latitude_batch,
                    )
                    for batch in radii.split(radius_batch)
                ]
            )
    path, latent = [], source
    while latent.kind != "spectral":
        if latent.kind not in ("activation", "affine") or len(latent.parents) != 1:
            raise ValueError("strip path must be affine/GELU after a harmonic field")
        path.append(latent)
        latent = latent.parents[0]
    peak = source.error.new_zeros((len(radii), len(source.error)))
    for box in harmonic_strip_boxes(
        latent, radii, shape, order, coordinate_only, cache, latitude_batch
    ):
        for start in range(0, len(box[0]), point_batch):
            local = tuple(value[start : start + point_batch] for value in box)
            for node in reversed(path):
                if node.kind == "activation":
                    local = gelu_rectangle(*local)
                else:
                    real, imaginary, er, ei = local
                    weight, bias = node.data["weight"], node.data["bias"]
                    local = (
                        real @ weight.to(real).T + bias.to(real),
                        imaginary @ weight.to(imaginary).T,
                        er @ weight.abs().to(er).T,
                        ei @ weight.abs().to(ei).T,
                    )
            real, imaginary, er, ei = local
            cap = torch.hypot(real.abs() + er, imaginary.abs() + ei)
            cap = torch.nan_to_num(cap, nan=math.inf, posinf=math.inf)
            peak = torch.maximum(peak, cap.amax(0))
    return peak

def attach_signed_strip(source, name, cache, radii=None, shape=(256, 512)):
    if radii is None or source.kind != "activation":
        return
    if name != "encoder.1":
        parts = name.split(".")
        if (
            len(parts) != 4
            or parts[0] != "blocks"
            or parts[2:] != ["mlp", "1"]
            or not 0 <= int(parts[1]) <= 6
        ):
            return
    if len(radii) != 2:
        raise ValueError("strip radii must specify theta and phi")
    candidates = [
        source.error.new_tensor(radius).reshape(-1).unique(sorted=True)
        for radius in radii
    ]
    if any(
        not len(values) or not torch.isfinite(values).all() or (values <= 0).any()
        for values in candidates
    ):
        raise ValueError("strip radii must be finite positive numbers")
    latent = source
    while latent.kind in ("activation", "affine"):
        latent = latent.parents[0]
    if latent.kind != "spectral":
        return
    degrees, orders = latent.data["coefficients"].shape[-2:]
    if degrees < 2 or orders != degrees:
        return
    source.data["signed_strip"] = {
        label: (
            candidates[coordinate],
            signed_pointwise_strip(
                source,
                candidates[coordinate],
                shape,
                12,
                coordinate,
                cache,
                radius_batch=1 if math.prod(shape) > 512 * 1024 else None,
            ),
        )
        for coordinate, label in enumerate(("theta", "phi"))
    }
