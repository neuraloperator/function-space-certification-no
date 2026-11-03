import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from lean_no.bounds.activations import (
    complex_gelu,
    gelu_first,
    gelu_remainder_log_coefficient,
    gelu_series,
    gelu_third,
    local_gelu_log_coefficient,
)
from lean_no.graph import walk

def frozen_in_time(node, seen=None):
    seen = set() if seen is None else seen
    if node in seen:
        return True
    seen.add(node)
    if node.kind == "spectral":
        return bool((node.data["modes"][0] == 0).all())
    if node.kind == "input":
        return node.data["tp"] == 0 and frozen_in_time(node.data["u"], seen)
    if node.kind == "profile":
        return frozen_in_time(
            node.data.get("derivative_field", node.data["nominal"]), seen
        )
    if node.kind == "initial_blend":
        return False
    return all(frozen_in_time(p, seen) for p in node.parents)

def polynomial_product(left, right):
    degree = left.shape[-1] - 1
    windows = F.pad(left, (degree, degree)).unfold(-1, degree + 1, 1)
    return (windows * right.flip(-1).unsqueeze(-2)).sum(-1)

def polynomial_composition(series, argument):
    degree = argument.shape[-1] - 1
    width = max(2, math.isqrt(degree + 1))
    zero = torch.zeros_like(argument[..., 0].real)

    def multiply(left, lr, right, rr):
        full = polynomial_product(left, right)
        error = (
            full[..., degree + 1 :].abs().sum(-1)
            + lr * (right.abs().sum(-1) + rr)
            + rr * left.abs().sum(-1)
        )
        return full[..., : degree + 1].clone(), error

    one = torch.zeros_like(argument)
    one[..., 0] = 1
    powers, errors = [one, argument], [zero, zero]
    for _ in range(2, width + 1):
        p, r = multiply(powers[-1], errors[-1], argument, zero)
        powers.append(p)
        errors.append(r)
    basis = torch.stack(powers[:width], -2)
    radii = torch.stack(errors[:width], -1)
    result, remainder = torch.zeros_like(argument), zero
    for begin in reversed(range(0, degree + 1, width)):
        if begin != (degree // width) * width:
            result, remainder = multiply(
                result, remainder, powers[width], errors[width]
            )
        weights = series[..., begin : begin + width]
        result = result + torch.einsum(
            "nci,ncik->nck", weights, basis[..., : weights.shape[-1], :]
        )
        remainder = remainder + (weights.abs() * radii[..., : weights.shape[-1]]).sum(
            -1
        )
    return result, remainder

@dataclass
class Taylor:
    coefficients: torch.Tensor  # [query, channel, degree+1]
    remainder: torch.Tensor
    complex_domain: bool = False

    def magnitude(self):
        return self.coefficients.abs().sum(-1) + self.remainder

    def affine(self, weight, bias):
        c = torch.einsum(
            "nck,oc->nok", self.coefficients, weight.to(self.coefficients.dtype)
        )
        c[..., 0] += bias
        return Taylor(c, self.remainder @ weight.abs().T, self.complex_domain)

    def __add__(self, other):
        return Taylor(
            self.coefficients + other.coefficients,
            self.remainder + other.remainder,
            self.complex_domain or other.complex_domain,
        )

    def scale(self, value):
        return Taylor(
            self.coefficients * value[..., None],
            self.remainder * value.abs(),
            self.complex_domain,
        )

    def gelu(self):
        c = self.coefficients
        degree = c.shape[-1] - 1
        derivatives, e0, e1 = gelu_series(c[..., 0], degree)
        q = c.clone()
        q[..., 0] = 0
        variation = q.abs().sum(-1)
        polynomial, truncation = polynomial_composition(derivatives, q)
        complex_domain = self.complex_domain or c.is_complex()
        height = (
            ((c[..., 0].imag.abs() if c.is_complex() else 0) + variation)
            if complex_domain
            else None
        )
        coefficient = torch.minimum(
            torch.as_tensor(
                gelu_remainder_log_coefficient(degree + 1, height),
                device=c.device,
                dtype=variation.dtype,
            ),
            local_gelu_log_coefficient(
                c[..., 0], variation, degree + 1, complex_domain=complex_domain
            ),
        )
        log_tail = coefficient + (degree + 1) * variation.log()
        tail = torch.where(variation == 0, 0.0, log_tail.exp())
        if height is None:
            propagated = self.remainder * gelu_first
        else:
            total_height = height + self.remainder
            near = (c[..., 0].real.abs() - variation - self.remainder).clamp(min=0)
            modulus = c[..., 0].abs() + variation + self.remainder
            gain = 1 + (total_height + modulus) * (
                (total_height.square() - near.square()) / 2
            ).exp() / math.sqrt(2 * math.pi)
            gain = torch.nan_to_num(gain, nan=torch.inf, posinf=torch.inf)
            propagated = torch.where(self.remainder == 0, 0.0, gain * self.remainder)
        return Taylor(
            polynomial,
            truncation + tail + propagated + e0 + e1 * variation,
            complex_domain,
        )

class ProfileTaylorCache:

    def __init__(self, capacity=131072, batch=None):
        if capacity < 1 or (batch is not None and batch < 1):
            raise ValueError("positive cache capacity and polynomial batch required")
        self.capacity, self.batch = capacity, batch
        self.tables = {}
        self.hits = self.misses = 0

    def quadratic(self, source, space, width, degree):
        if degree < 2:
            raise ValueError("quadratic profile reduction requires degree at least two")
        key = (source, degree, space.dtype)
        if key not in self.tables:
            if not frozen_in_time(source):
                raise ValueError("polynomial profile cache requires a frozen graph")
            self.tables[key] = (
                {},
                space.new_empty(self.capacity, len(source.error), 4),
            )
        lookup, table = self.tables[key]
        keys = list(zip(space.cpu().tolist(), width.cpu().tolist()))
        missing = list(dict.fromkeys(k for k in keys if k not in lookup))
        if len(lookup) + len(missing) > self.capacity:
            lookup.clear()
            missing = list(dict.fromkeys(keys))
        if len(missing) > self.capacity:
            raise ValueError("profile polynomial query exceeds cache capacity")
        batch = self.batch if self.batch is not None else (128 if space.is_cuda else 32)
        for begin in range(0, len(missing), batch):
            selected = missing[begin : begin + batch]
            x = space.new_tensor([p[0] for p in selected])
            h = width.new_tensor([p[1] for p in selected])
            model = spatial_taylor(source, x.real * 0, x, h, degree)
            divisor = torch.where(h == 0, torch.ones_like(h), h)[:, None]
            c = model.coefficients
            gx = c[..., 1] / divisor
            qxx = 2 * c[..., 2] / divisor.square()
            remainder = c[..., 3:].abs().sum(-1) + model.remainder
            packed = torch.stack((c[..., 0], gx, qxx, remainder), -1)
            start = len(lookup)
            table[start : start + len(selected)] = packed
            lookup.update((k, start + i) for i, k in enumerate(selected))
        ids = torch.tensor([lookup[k] for k in keys], device=space.device)
        self.misses += len(missing)
        self.hits += len(keys) - len(missing)
        selected = table[ids]
        return (
            selected[..., 0],
            selected[..., 1],
            selected[..., 2],
            selected[..., 3].real,
        )

@torch.no_grad()
def spatial_taylor(
    field,
    time,
    space,
    halfwidth,
    degree=32,
    trace=None,
    uncertain_graph=False,
    frozen_profiles=None,
):
    split = field.metadata.get("split", 1.0)
    if degree < 1 or time.ndim != 1 or space.shape != time.shape:
        raise ValueError("positive degree and paired time/space vectors required")
    if time.is_complex() or not bool(
        torch.isfinite(time).all() & torch.isfinite(space).all()
    ):
        raise ValueError("finite real query times and finite spatial centers required")
    if isinstance(halfwidth, complex) or (
        isinstance(halfwidth, torch.Tensor) and halfwidth.is_complex()
    ):
        raise ValueError("spatial halfwidth must be real")
    width = torch.as_tensor(
        halfwidth, device=space.device, dtype=space.real.dtype
    ).expand_as(space)
    if bool((width < 0).any()):
        raise ValueError("negative spatial halfwidth")
    zero_time = torch.zeros_like(time)
    complex_domain = space.is_complex()
    real_center = not complex_domain or not bool((space.imag != 0).any())
    center_space = space.real if real_center else space
    cache = {}

    def constant(value):
        c = value.new_zeros(*value.shape, degree + 1)
        c[..., 0] = value
        return Taylor(c, torch.zeros_like(value.real), complex_domain)

    def apply(node, visit, times, active):
        d = node.data
        if node.kind == "spectral":
            kt, kx = d["modes"][0], d["modes"][1]
            bt = torch.exp(2j * math.pi * times[:, None] * kt)
            bx = torch.exp(2j * math.pi * center_space[:, None] * kx) * torch.where(
                kx == 0, 1.0, 2.0
            )
            tmp = torch.einsum("nk,ckl->ncl", bt, d["coefficients"])
            phase = 2j * math.pi * width[:, None] * kx
            if not real_center:
                minus = torch.exp(-2j * math.pi * space[:, None] * kx) * torch.where(
                    kx == 0, 1.0, 2.0
                )
                tmp_minus = torch.einsum(
                    "nk,ckl->ncl", bt.conj(), d["coefficients"].conj()
                )
            terms = []
            for n in range(degree + 1):
                value = torch.einsum("ncl,nl->nc", tmp, bx)
                if not real_center:
                    value = (value + torch.einsum("ncl,nl->nc", tmp_minus, minus)) / 2
                    minus = minus * (-phase) / (n + 1)
                else:
                    value = value.real
                terms.append(value)
                bx = bx * phase / (n + 1)
            amp = (d["coefficients"].abs() * torch.where(kx == 0, 1.0, 2.0)).sum(1)
            omega = 2 * math.pi * width[:, None] * kx.abs()
            factor = ((degree + 1) * omega.log() - math.lgamma(degree + 2)).exp()
            if complex_domain:
                factor = (
                    factor
                    * (omega + 2 * math.pi * space.imag.abs()[:, None] * kx.abs()).exp()
                )
            result = Taylor(
                torch.stack(terms, -1),
                torch.einsum("cl,nl->nc", amp, factor),
                complex_domain,
            )
        elif node.kind == "profile":
            source = d.get("derivative_field") if uncertain_graph else None
            if uncertain_graph and source is None and bool((node.error != 0).any()):
                raise ValueError(
                    "uncertainty trace requires a derivative-preserving handoff"
                )
            source = d["nominal"] if source is None else source
            result = (
                visit(source, zero_time, True)
                if frozen_profiles is None
                else frozen_profiles(source, space, width, degree, trace)
            )
        elif node.kind == "input":
            u = visit(d["u"], times, active)
            tc = constant(times[:, None] * d["tp"])
            xc = constant(center_space[:, None] * d["lx"])
            xc.coefficients[..., 1] = width[:, None] * d["lx"]
            result = Taylor(
                torch.cat((u.coefficients, tc.coefficients, xc.coefficients), 1),
                torch.cat((u.remainder, tc.remainder, xc.remainder), 1),
                complex_domain,
            )
        elif node.kind == "pad" and not active:
            result = constant(space.new_zeros(len(space), len(node.error)))
        else:
            parents = [visit(p, times, active) for p in node.parents]
            if node.kind == "affine":
                result = parents[0].affine(d["weight"], d["bias"])
            elif node.kind == "add":
                result = parents[0] + parents[1]
            elif node.kind == "activation":
                result = parents[0].gelu()
            elif node.kind == "pad":
                result = parents[0]
            elif node.kind == "initial_blend":
                tau = times[:, None] / split
                result = parents[0].scale(1 - tau) + parents[1].scale(tau)
            else:
                raise ValueError(node.kind)
        if trace is not None:
            trace(node, result)
        return result

    visit = walk(
        apply, cache, key=lambda node, times, active: (node, times.data_ptr(), active)
    )

    # Fixed query times may span both padding branches, so split their batches.
    active = time < split
    if bool(active.all()) or bool((~active).all()):
        return visit(field, time, bool(active.all()))
    pieces = []
    for selected in (active, ~active):
        pieces.append(
            (
                selected,
                spatial_taylor(
                    field,
                    time[selected],
                    space[selected],
                    width[selected],
                    degree,
                    trace=trace,
                    uncertain_graph=uncertain_graph,
                    frozen_profiles=frozen_profiles,
                ),
            )
        )
    coefficient_dtype = torch.promote_types(
        pieces[0][1].coefficients.dtype, pieces[1][1].coefficients.dtype
    )
    c = torch.empty(
        len(time),
        len(field.error),
        degree + 1,
        device=space.device,
        dtype=coefficient_dtype,
    )
    r = c.real.new_empty(len(time), len(field.error))
    for selected, part in pieces:
        c[selected], r[selected] = part.coefficients, part.remainder
    return Taylor(c, r, complex_domain)

def taylor_disk(
    node,
    centers,
    radii,
    active,
    cache=None,
    real_only=False,
    gelu_order=8,
    real_spatial=False,
    profile_degree=0,
):
    split = node.metadata.get("split", 1.0)
    cache = {} if cache is None else cache
    if real_only and centers.is_complex():
        raise ValueError("real Taylor boxes require real centers")

    def apply(node, visit):
        d = node.data
        if node.kind == "spectral":
            c, kt, kx = d["coefficients"], d["modes"][0], d["modes"][1]
            bt = torch.exp(2j * math.pi * centers[:, :1] * kt)
            bx = torch.exp(2j * math.pi * centers[:, 1:] * kx) * torch.where(
                kx == 0, 1.0, 2.0
            )

            def synth(coeff):
                pos = torch.einsum("bk,ckl,bl->bc", bt, coeff, bx)
                if not centers.is_complex():
                    return pos.real
                neg_t = torch.exp(-2j * math.pi * centers[:, :1] * kt)
                neg_x = torch.exp(-2j * math.pi * centers[:, 1:] * kx) * torch.where(
                    kx == 0, 1.0, 2.0
                )
                return (
                    pos + torch.einsum("bk,ckl,bl->bc", neg_t, coeff.conj(), neg_x)
                ) / 2

            v = synth(c)
            g = torch.stack(
                (
                    synth(c * (2j * math.pi * kt[:, None])),
                    synth(c * (2j * math.pi * kx)),
                ),
                -1,
            )
            omega = (
                2
                * math.pi
                * (
                    radii[:, 0, None, None] * kt.abs()[None, :, None]
                    + radii[:, 1, None, None] * kx.abs()[None, None, :]
                )
            )
            amp = c.abs() * torch.where(kx == 0, 1.0, 2.0)
            h = torch.stack(
                (
                    torch.stack(
                        (
                            synth(c * (2j * math.pi * kt[:, None]) ** 2),
                            synth(
                                c * (2j * math.pi * kt[:, None]) * (2j * math.pi * kx)
                            ),
                        ),
                        -1,
                    ),
                    torch.stack(
                        (
                            synth(
                                c * (2j * math.pi * kt[:, None]) * (2j * math.pi * kx)
                            ),
                            synth(c * (2j * math.pi * kx) ** 2),
                        ),
                        -1,
                    ),
                ),
                -2,
            )
            height = (
                (
                    2
                    * math.pi
                    * (
                        centers[:, :1].imag.abs()[:, :, None] * kt.abs()[None, :, None]
                        + centers[:, 1:].imag.abs()[:, :, None]
                        * kx.abs()[None, None, :]
                    )
                )
                if centers.is_complex()
                else 0
            )
            factor = 1 if real_only else (omega + height).exp()
            r = torch.einsum("bkl,ckl->bc", omega**3 * factor / 6, amp)
        elif node.kind == "profile":
            if real_spatial and not real_only:
                # A time contour leaves x real. The complete real magnitude cover
                # of the frozen profile therefore supplies a constant enclosure,
                # uniformly in x, without analytically continuing any old slab.
                # These are polynomial enclosure coefficients, not point jets.
                if centers.is_complex() and bool((centers[:, 1].imag != 0).any()):
                    raise ValueError(
                        "real spatial profile requires real spatial centers"
                    )
                if bool(
                    (
                        (centers[:, 1].real - radii[:, 1] < 0)
                        | (centers[:, 1].real + radii[:, 1] > 1)
                    ).any()
                ):
                    raise ValueError(
                        "real spatial profile range is certified only on [0,1]"
                    )
                v = centers.new_zeros(len(centers), len(node.error))
                g = v.new_zeros(*v.shape, 2)
                h = v.new_zeros(*v.shape, 2, 2)
                r = node.size.expand(len(centers), -1)
                if profile_degree:
                    model_key = ("profile_models", profile_degree)
                    if model_key not in cache:
                        cache[model_key] = ProfileTaylorCache()
                    pv, gx, qxx, pr = cache[model_key].quadratic(
                        d["nominal"], centers[:, 1].real, radii[:, 1], profile_degree
                    )
                    variation = (
                        pr
                        + gx.abs() * radii[:, 1, None]
                        + qxx.abs() * radii[:, 1, None].square() / 2
                    )
                    use = torch.isfinite(variation) & (variation <= r)
                    v = torch.where(use, pv.to(v.dtype), v)
                    g[..., 1] = torch.where(use, gx.to(g.dtype), 0.0)
                    h[..., 1, 1] = torch.where(use, qxx.to(h.dtype), 0.0)
                    r = torch.where(use, pr, r)
                return v, g, h, r
            if profile_degree:
                model_key = ("profile_models", profile_degree)
                if model_key not in cache:
                    cache[model_key] = ProfileTaylorCache()
                spatial_centers = centers[:, 1]
                if not real_only:
                    # A real center can still represent a complex analytic disk.
                    # Real-axis derivative caps alone do not enclose that disk.
                    spatial_centers = spatial_centers.to(torch.complex128)
                v, gx, qxx, r = cache[model_key].quadratic(
                    d["nominal"], spatial_centers, radii[:, 1], profile_degree
                )
                g = v.new_zeros(*v.shape, 2)
                h = v.new_zeros(*v.shape, 2, 2)
                g[..., 1], h[..., 1, 1] = gx, qxx
                return v, g, h, r
            cc, rr = centers.clone(), radii.clone()
            cc[:, 0] = 0
            rr[:, 0] = 0
            profile_real = real_only or real_spatial
            if profile_real:
                if cc.is_complex() and bool((cc[:, 1].imag != 0).any()):
                    raise ValueError(
                        "real spatial profile requires real spatial centers"
                    )
                cc = cc.real
            v, g, h, r = taylor_disk(
                d.get("derivative_field", d["nominal"]),
                cc,
                rr,
                True,
                cache,
                real_only=profile_real,
                gelu_order=gelu_order,
                real_spatial=real_spatial,
                profile_degree=profile_degree,
            )
        elif node.kind == "input":
            uv, ug, uh, ur = visit(d["u"])
            v = torch.cat((uv, centers[:, :1] * d["tp"], centers[:, 1:] * d["lx"]), -1)
            g = v.new_zeros(*v.shape, 2)
            g[:, :1] = ug
            g[:, 1, 0], g[:, 2, 1] = d["tp"], d["lx"]
            h = v.new_zeros(*v.shape, 2, 2)
            h[:, :1] = uh
            r = torch.cat((ur, centers.real * 0), -1)
        elif node.kind == "pad" and not active:
            v = centers.new_zeros(len(centers), len(node.error))
            g, h, r = (
                v.new_zeros(*v.shape, 2),
                v.new_zeros(*v.shape, 2, 2),
                v.real.clone(),
            )
        else:
            parts = [visit(p) for p in node.parents]
            if node.kind == "affine":
                v, g, h, r = parts[0]
                v = v @ d["weight"].T.to(v.dtype) + d["bias"]
                g = torch.einsum("bid,oi->bod", g, d["weight"].to(g.dtype))
                h = torch.einsum("bide,oi->bode", h, d["weight"].to(h.dtype))
                r = r @ d["weight"].abs().T
            elif node.kind == "add":
                v, g, h, r = (sum(p[j] for p in parts) for j in range(4))
            elif node.kind == "initial_blend":
                av, ag, ah, ar = parts[0]
                bv, bg, bh, br = parts[1]
                tau = centers[:, :1] / split
                dtau = radii[:, :1] / split
                direction = centers.new_tensor([1 / split, 0.0])
                dv, dg, dh = bv - av, bg - ag, bh - ah
                v = (1 - tau) * av + tau * bv
                g = (
                    (1 - tau)[..., None] * ag
                    + tau[..., None] * bg
                    + dv[..., None] * direction
                )
                h = (
                    (1 - tau)[..., None, None] * ah
                    + tau[..., None, None] * bh
                    + direction[:, None] * dg[..., None, :]
                    + dg[..., :, None] * direction[None, :]
                )
                quadratic = (
                    dh.abs() * radii[:, None, :, None] * radii[:, None, None, :]
                ).sum((-1, -2)) / 2
                r = (1 - tau).abs() * ar + tau.abs() * br + dtau * (quadratic + ar + br)
            elif node.kind == "pad":
                v, g, h, r = parts[0]
            elif node.kind == "activation":
                v, g, h, r = parts[0]
                linear = (g.abs() * radii[:, None]).sum(-1)
                quadratic = (
                    h.abs() * radii[:, None, :, None] * radii[:, None, None, :]
                ).sum((-1, -2)) / 2
                tail = quadratic + r
                variation = linear + tail
                density = torch.exp(-v.square() / 2) / math.sqrt(2 * math.pi)
                if v.is_complex():
                    activated, slope, value_error, slope_error = complex_gelu(
                        v, order=gelu_order
                    )
                    height = v.imag.abs() + variation
                else:
                    activated = F.gelu(v)
                    slope = 0.5 * torch.erfc(-v / math.sqrt(2)) + v * density
                    value_error = slope_error = 0
                    height = variation
                lo = (v.real.abs() - variation).clamp(min=0)
                z = v.abs() + variation
                c3 = (
                    (z**3 + 4 * z)
                    * torch.exp((height.square() - lo.square()) / 2)
                    / math.sqrt(2 * math.pi)
                )
                if real_only:
                    c3 = torch.full_like(variation, gelu_third)
                second = (2 - v.square()) * density
                r = (
                    slope.abs() * r
                    + second.abs() * (linear * tail + tail.square() / 2)
                    + c3 * variation**3 / 6
                    + value_error
                    + slope_error * variation
                )
                h = (
                    slope[..., None, None] * h
                    + second[..., None, None] * g[..., :, None] * g[..., None, :]
                )
                v, g = activated, slope[..., None] * g
            else:
                raise ValueError(node.kind)
        return v, g, h, r

    return walk(apply, cache, key=lambda node: (node, real_only))(node)

def polydisk(
    node,
    centers,
    radii,
    active,
    gelu_order=8,
    real_spatial=False,
    profile_degree=0,
    profile_models=None,
):
    cache = (
        {}
        if profile_models is None
        else {("profile_models", profile_degree): profile_models}
    )
    v, g, h, r = taylor_disk(
        node,
        centers,
        radii,
        active,
        cache,
        gelu_order=gelu_order,
        real_spatial=real_spatial,
        profile_degree=profile_degree,
    )
    quadratic = (h.abs() * radii[:, None, :, None] * radii[:, None, None, :]).sum(
        (-1, -2)
    ) / 2
    return v, g, quadratic + r
