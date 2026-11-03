import math

import torch
import torch.nn.functional as F

from lean_no.adapters.rollout_fno import build_graph
from lean_no.bounds.activations import gelu_first
from lean_no.bounds.fourier import spectral_kernel_l2
from lean_no.bounds.normalization import stable_rms
from lean_no.graph import affine, execute, fourier, make_field, walk

def initial_state_handoff(output, retain_derivatives=False):
    split = output.metadata.get("split", 1.0)
    cache = {}

    def apply(node, visit, actual=False):
        d = node.data
        error = node.error if actual else node.error * 0
        if node.kind == "spectral":
            phase = torch.exp(2j * math.pi * split * d["modes"][0])
            c = (d["coefficients"] * phase[None, :, None]).sum(1, keepdim=True)
            out = fourier(
                c, (d["modes"][0].new_zeros(1), d["modes"][1]), error, node.metadata
            )
            if actual and "coefficient_error" in d:
                out.data["coefficient_error"] = d["coefficient_error"].sum(
                    1, keepdim=True
                )
            elif actual and bool((error != 0).any()):
                raise ValueError("terminal derivatives require per-coefficient errors")
        elif node.kind == "pad":
            out = visit(node.parents[0], actual)
        elif node.kind == "initial_blend":
            # The terminal trace has t/h=1 exactly, including uncertain
            # inputs and derivatives; the direct initial branch vanishes.
            out = visit(node.parents[1], actual)
        elif node.kind == "input":
            base = make_field(
                "input", {**d, "tp": 0.0}, (), error, node.size, node.metadata
            )
            w = torch.eye(3, device=node.error.device, dtype=node.error.dtype)
            out = affine(base, w, node.error.new_tensor([0, split * d["tp"], 0]))
        else:
            out = make_field(
                node.kind,
                dict(d),
                tuple(visit(p, actual) for p in node.parents),
                error,
                node.size,
                node.metadata,
            )
        return out

    visit = walk(apply, cache, key=lambda node, actual=False: (node, actual))

    nominal = visit(output)
    data = {"nominal": nominal}
    if retain_derivatives:
        data["derivative_field"] = visit(output, True)
    result = make_field(
        "profile",
        data,
        (),
        output.error.clone(),
        output.size.clone(),
        output.metadata,
    )
    return result

class FrozenProfileValues:

    def __init__(self, space):
        self.space = space.clone()
        self.table = {}
        self.hits = self.misses = 0

    def evaluate(self, node, t, x):
        if not torch.equal(x, self.space):
            raise ValueError("profile value cache spatial grid changed")
        if node not in self.table:
            self.table[node] = values(
                node.data["nominal"], t[:, :1] * 0, x, active=True, profile_cache=self
            )
            self.misses += 1
        else:
            self.hits += 1
        result = self.table[node]
        if result.shape[0] != t.shape[0]:
            raise ValueError("profile value cache batch changed")
        return result.expand(-1, t.shape[1], -1, -1)

def values(node, t, x, active=None, cache=None, profile_cache=None):

    split = node.metadata.get("split", 1.0)

    def spectral(node, visit):
        d = node.data
        bt = torch.exp(2j * math.pi * t[..., None] * d["modes"][0])
        bx = torch.exp(2j * math.pi * x[..., None] * d["modes"][1])
        bx = bx * torch.where(d["modes"][1] == 0, 1.0, 2.0)
        tmp = torch.einsum("btk,ckl->btcl", bt, d["coefficients"])
        return torch.einsum("btcl,bxl->btxc", tmp, bx).real

    def profile(node, visit):
        if profile_cache is not None:
            return profile_cache.evaluate(node, t, x)
        return values(node.data["nominal"], t[:, :1] * 0, x, active=True).expand(
            -1, t.shape[1], -1, -1
        )

    def inputs(node, visit):
        d = node.data
        u = visit(d["u"])[..., 0]
        return torch.stack(
            (
                u,
                t[..., None].expand_as(u) * d["tp"],
                x[:, None, :].expand_as(u) * d["lx"],
            ),
            -1,
        )

    def pad(node, visit):
        v = visit(node.parents[0])
        if active is False:
            return v * 0
        return (
            torch.where((t < split)[..., None, None], v, 0.0) if active is None else v
        )

    tau = t[..., None, None] / split
    operations = {
        "spectral": spectral,
        "profile": profile,
        "input": inputs,
        "pad": pad,
        "affine": lambda n, v: v(n.parents[0]) @ n.data["weight"].T + n.data["bias"],
        "add": lambda n, v: sum(v(p) for p in n.parents),
        "initial_blend": lambda n, v: (
            (1 - tau) * v(n.parents[0]) + tau * v(n.parents[1])
        ),
        "activation": lambda n, v: F.gelu(v(n.parents[0])),
    }
    return walk(operations, cache)(node)

@torch.no_grad()
def rollout(state, profile, integrate, lengths, split, native_shape, orders=((0, 0),)):
    graph = build_graph(state, lengths, split, native_shape, orders)
    time_period, space_period = lengths
    if callable(profile):
        from lean_no.function_space.analytic import Series, describe

        reference = state["fno_blocks.convs.bias"]
        series = describe(profile)
        profile = Series(
            tuple(((0.0, k[0]), c) for k, c in series.terms), (0.0, 0.0)
        ).field(lengths, reference.device, reference.dtype)
    ref = profile.error
    budget = {}
    inputs = make_field(
        "input",
        {"u": profile, "tp": time_period, "lx": space_period},
        (),
        torch.cat((ref, ref.new_zeros(2))),
        torch.cat((profile.size, ref.new_tensor([split * time_period, space_period]))),
        graph.metadata,
    )
    inputs.run.metadata["input_node"] = inputs.node
    b = ref.new_zeros(2, 3)
    b[0, 0] = ref[0]
    budget[inputs] = b
    trace = []
    native_t = (
        torch.arange(native_shape[0], device=ref.device, dtype=ref.dtype)[None]
        / native_shape[0]
    )
    native_x = (
        torch.arange(native_shape[1], device=ref.device, dtype=ref.dtype)[None]
        / native_shape[1]
    )
    native_profiles = FrozenProfileValues(native_x)

    def apply(op, parents):
        p, d = parents[0], op.data
        bp = budget[p]
        if op.kind == "affine":
            out = affine(p, d["weight"], d["bias"])
            bo = bp @ d["weight"].abs().T
        elif op.kind == "activation":
            out = make_field(
                "activation", {"activation": "gelu"}, (p,), gelu_first * p.error, p.size
            )
            bo = gelu_first * bp
        elif op.kind == "add":
            q = parents[1]
            bo = bp + budget[q]
            out = make_field("add", {}, parents, bo.sum(0), p.size + q.size)
        elif op.kind == "initial_blend":
            q = parents[1]
            bo = torch.maximum(bp, budget[q])
            out = make_field(
                "initial_blend", {}, parents, bo.sum(0), torch.maximum(p.size, q.size)
            )
        elif op.kind == "pad":
            bo = bp.clone()
            out = make_field("pad", {}, (p,), p.error, p.size)
        elif op.kind == "normalize":
            val = values(p, native_t, native_x, profile_cache=native_profiles)
            mean = val.mean()
            std = (val.var(unbiased=False) + d["epsilon"]).sqrt()
            rms = stable_rms(bp)
            E = rms.sum()
            lower = (std - E).clamp(min=math.sqrt(d["epsilon"]))
            centered = p.size + mean.abs()
            bo = d["weight"].abs() * (
                (bp + rms[:, None]) / lower + centered * rms[:, None] / (std * lower)
            )
            out = affine(
                p,
                torch.diag(d["weight"] / std),
                d["bias"] - d["weight"] * mean / std,
            )
            out.data["scale_error"] = d["weight"].abs() * E / (std * lower)
            out.data["_bound_bias_error"] = d["weight"].abs() * (
                E / lower + mean.abs() * E / (std * lower)
            )
            trace.append(
                dict(
                    normalization=d["index"],
                    std=float(std),
                    uncertainty=float(E),
                    denominator_lower=float(lower),
                    budget=bo.amax(-1).tolist(),
                )
            )
        elif op.kind == "spectral":
            c, error = integrate(p, d["layer"])
            w = d["weight"]
            bc = bp.clone()
            bc[1] += error
            modes = torch.arange(w.shape[-1], device=ref.device, dtype=ref.dtype)
            sym = torch.where(modes == 0, 1.0, 2.0).to(ref)
            # Incoming radii bound one common source function. Quadrature errors
            # are independent coefficient enclosures and retain the triangle bound.
            bo = bp @ spectral_kernel_l2(w).T
            bo[1] += torch.einsum("oikl,i,l->o", w.abs(), error, sym)
            spec = fourier(
                torch.einsum("ikl,oikl->okl", c, w),
                (
                    torch.arange(
                        -(w.shape[-2] // 2),
                        w.shape[-2] - w.shape[-2] // 2,
                        device=ref.device,
                        dtype=ref.dtype,
                    ),
                    modes,
                ),
                bo.sum(0),
                graph.metadata,
            )
            spec.data["coefficient_error"] = torch.einsum(
                "oikl,i->okl", w.abs(), bc.sum(0)
            )
            out = affine(
                spec,
                torch.eye(w.shape[0], device=ref.device, dtype=ref.dtype),
                d["bias"],
            )
        else:
            raise ValueError(op.kind)
        out = out.with_error(bo.sum(0))
        budget[out] = bo
        return out

    output = execute(graph, inputs, apply)
    if getattr(integrate, "rule", None) == "gaussian":
        output.data["integration"] = "gaussian"
    return output, budget[output], trace

def output_bounds(field, points):
    split = field.metadata.get("split", 1.0)
    time_period, space_period = field.metadata.get("lengths", (1.0, 1.0))
    points = torch.as_tensor(points, device=field.error.device, dtype=field.error.dtype)
    if points.ndim != 2 or points.shape[1] != 2 or not torch.isfinite(points).all():
        raise ValueError("expected finite N by 2 physical coordinates")
    if (
        (points < 0).any()
        or (points[:, 0] > split * time_period).any()
        or (points[:, 1] > space_period).any()
    ):
        raise ValueError("points must lie in the active physical slab")
    centers = []
    for chunk in points.split(1024):
        if len(chunk):
            centers.append(
                values(
                    field,
                    chunk[:, :1] / time_period,
                    chunk[:, 1:] / space_period,
                    active=True,
                )[:, 0, 0]
            )
    nominal = torch.cat(centers) if centers else points.new_empty((0, len(field.error)))
    from lean_no.certificate import QueryBounds

    return QueryBounds(nominal, field.error.expand_as(nominal))
