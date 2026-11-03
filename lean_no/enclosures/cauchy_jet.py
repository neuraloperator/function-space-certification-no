import math

import torch

from lean_no.bounds.affine import DualForm
from lean_no.enclosures.taylor import Taylor, frozen_in_time, spatial_taylor

class ComplexErrorTrace:

    def __init__(self, radius):
        self.radius = radius
        self.errors = {}
        self.models = {}

    def __call__(self, node, model):
        d = node.data
        zero = torch.zeros_like(model.remainder)
        if bool((node.error == 0).all()):
            error = zero
        elif node.kind == "spectral":
            if "coefficient_error" not in d:
                raise ValueError("uncertain profile needs spectral coefficient errors")
            symmetry = torch.where(d["modes"][1] == 0, 1.0, 2.0)
            height = 2 * math.pi * self.radius[:, None] * d["modes"][1].abs()
            error = (
                d["coefficient_error"][None] * symmetry * height.exp()[:, None, None]
            ).sum((-1, -2))
        elif node.kind == "profile":
            source = d.get("derivative_field", d["nominal"])
            error = self.errors[source]
        elif node.kind == "input":
            error = torch.cat((self.errors[d["u"]], zero[:, :2]), -1)
        elif node.kind == "affine":
            p = node.parents[0]
            ep = self.errors[p]
            error = ep @ d["weight"].abs().T
            if "scale_error" in d:
                scale = d["scale_error"]
                magnitude = self.models[p].magnitude()
                error = error + scale * (magnitude + ep)
                if "_bound_bias_error" in d:
                    bias = d["_bound_bias_error"]
                else:
                    # Historical handoffs omit the constant bias perturbation.
                    # Bound it with the affine identity at a real reference point.
                    bias = node.error + p.error @ d["weight"].abs().T
                    bias = bias + scale * (p.size + p.error)
                error = error + bias
        elif node.kind == "add":
            error = self.errors[node.parents[0]] + self.errors[node.parents[1]]
        elif node.kind == "pad":
            error = self.errors.get(node.parents[0], zero)
        elif node.kind == "activation":
            p = node.parents[0]
            parent = self.models[p]
            ep = self.errors[p]
            center = parent.coefficients[..., 0]
            variation = parent.coefficients[..., 1:].abs().sum(-1) + parent.remainder
            height = (center.imag.abs() if center.is_complex() else 0) + variation + ep
            near = (center.real.abs() - variation - ep).clamp(min=0)
            modulus = center.abs() + variation + ep
            gain = 1 + (height + modulus) * (
                (height.square() - near.square()) / 2
            ).exp() / math.sqrt(2 * math.pi)
            error = torch.where(ep == 0, 0.0, ep * gain)
        else:
            raise ValueError(f"unsupported frozen uncertainty operation: {node.kind}")
        error = torch.where(node.error[None] == 0, 0.0, error)
        self.errors[node] = torch.nan_to_num(error, nan=torch.inf, posinf=torch.inf)
        self.models[node] = model

class FrozenPolynomialCache:

    def __init__(self, capacity=131072):
        if capacity < 1:
            raise ValueError("positive polynomial cache capacity required")
        self.capacity = capacity
        self.tables = {}
        self.hits = self.misses = 0

    @torch.no_grad()
    def __call__(self, source, space, radius, degree, trace):
        if (
            trace is None
            or not isinstance(trace, ComplexErrorTrace)
            or not frozen_in_time(source)
        ):
            raise ValueError("frozen complex-error trace required for polynomial reuse")
        real_center = not space.is_complex() or not bool((space.imag != 0).any())
        key = (source, degree, space.dtype, real_center)
        if key not in self.tables:
            dtype = space.real.dtype if real_center else space.dtype
            coefficients = torch.empty(
                self.capacity,
                len(source.error),
                degree + 1,
                device=space.device,
                dtype=dtype,
            )
            self.tables[key] = (
                {},
                coefficients,
                space.real.new_empty(self.capacity, len(source.error), 2),
            )
        lookup, coefficients, errors = self.tables[key]
        keys = list(zip(space.cpu().tolist(), radius.cpu().tolist()))
        missing = list(dict.fromkeys(k for k in keys if k not in lookup))
        if len(missing) + len(lookup) > self.capacity:
            lookup.clear()
            missing = list(dict.fromkeys(keys))
        if len(missing) > self.capacity:
            raise ValueError("polynomial query exceeds cache capacity")
        if missing:
            x = space.new_tensor([p[0] for p in missing])
            r = radius.new_tensor([p[1] for p in missing])
            inner = ComplexErrorTrace(r)
            model = spatial_taylor(
                source,
                x.real * 0,
                x,
                r,
                degree,
                trace=inner,
                uncertain_graph=True,
                frozen_profiles=self,
            )
            begin = len(lookup)
            coefficients[begin : begin + len(missing)] = model.coefficients
            errors[begin : begin + len(missing), :, 0] = model.remainder
            errors[begin : begin + len(missing), :, 1] = inner.errors[source]
            lookup.update((k, begin + i) for i, k in enumerate(missing))
        ids = torch.tensor([lookup[k] for k in keys], device=space.device)
        model = Taylor(coefficients[ids], errors[ids, :, 0], space.is_complex())
        trace.models[source] = model
        trace.errors[source] = errors[ids, :, 1]
        self.misses += len(missing)
        self.hits += len(keys) - len(missing)
        return model

@torch.no_grad()
def profile_forms(
    source, space, halfwidth, degree=32, frozen_profiles=None, orders=None, periods=None
):
    orders = tuple(
        orders if orders is not None else source.metadata.get("orders", ((0, 0),))
    )
    periods = (
        periods if periods is not None else source.metadata.get("lengths", (1.0, 1.0))
    )
    space_period = periods[1]
    if degree < max(dx for dt, dx in orders) + 2 or not frozen_in_time(source):
        raise ValueError(
            "degree must exceed spatial order by two; profile must be frozen"
        )
    # Positive query cells must keep shrinking their analytic disk. A fixed
    # radius floor would stop refinement from resolving sharp late profiles.
    radius = torch.where(
        halfwidth > 0, 2 * halfwidth, halfwidth.new_full(halfwidth.shape, 2.0**-20)
    )
    gap = radius - halfwidth
    trace = ComplexErrorTrace(radius)
    if frozen_profiles is None:
        model = spatial_taylor(
            source,
            space * 0,
            space.to(torch.complex128),
            radius,
            degree,
            trace=trace,
            uncertain_graph=True,
        )
    else:
        model = frozen_profiles(
            source, space.to(torch.complex128), radius, degree, trace
        )
    c = model.coefficients.real
    h = torch.stack((halfwidth * 0, halfwidth), -1)
    jets = []
    error = trace.errors[source]
    for dt, order in orders:
        if dt:
            z = torch.zeros_like(c[..., 0])
            jets.append(DualForm.constant(z, h))
            continue
        divisor = radius[:, None] ** order * space_period**order
        value = c[..., order] * math.factorial(order) / divisor
        gradient = value.new_zeros(*value.shape, 2)
        gradient[..., 1] = (
            c[..., order + 1] * math.factorial(order + 1) / (divisor * radius[:, None])
        )
        tail = torch.zeros_like(value)
        ratio = halfwidth / radius
        for k in range(order + 2, degree + 1):
            factor = math.factorial(k) // math.factorial(k - order)
            tail += c[..., k].abs() * factor * ratio[:, None] ** (k - order) / divisor
        nominal_error = (
            model.remainder
            * math.factorial(order)
            / (gap[:, None] * space_period) ** order
        )
        remainder = torch.nan_to_num(
            tail + nominal_error, nan=torch.inf, posinf=torch.inf
        )
        delta = error * math.factorial(order) / (gap[:, None] * space_period) ** order
        if order == 0:
            delta = torch.minimum(delta, source.error[None])
        jets.append(
            DualForm(
                value,
                gradient,
                torch.stack((remainder, remainder + delta), -1),
                h,
            )
        )
    return jets

class ProfileJetCache:
    def __init__(self, capacity=131072):
        if capacity < 1:
            raise ValueError("positive profile cache capacity required")
        self.capacity = capacity
        self.tables = {}
        self.hits = self.misses = 0

    @torch.no_grad()
    def evaluate(self, source, space, h, compute, orders=None, periods=None):
        orders = tuple(
            orders if orders is not None else source.metadata.get("orders", ((0, 0),))
        )
        periods = tuple(
            periods
            if periods is not None
            else source.metadata.get("lengths", (1.0, 1.0))
        )
        key = source, orders, periods
        if key not in self.tables:
            if not frozen_in_time(source):
                raise ValueError("profile cache requires a graph frozen in time")
            channels = len(source.error)
            self.tables[key] = (
                {},
                {
                    "c": space.new_empty(self.capacity, len(orders), channels),
                    "g": space.new_empty(self.capacity, len(orders), channels, 2),
                    "r": space.new_empty(self.capacity, len(orders), channels, 2),
                },
            )
        lookup, table = self.tables[key]
        keys = list(zip(space.cpu().tolist(), h[:, 1].cpu().tolist()))
        missing = list(dict.fromkeys(key for key in keys if key not in lookup))
        if len(missing) + len(lookup) > self.capacity:
            lookup.clear()
            missing = list(dict.fromkeys(keys))
        if len(missing) > self.capacity:
            raise ValueError("profile batch exceeds cache capacity")
        if missing:
            points = space.new_tensor(missing)
            jets = compute(points[:, 0], points[:, 1])
            start = len(lookup)
            for name, storage in table.items():
                storage[start : start + len(missing)] = torch.stack(
                    [getattr(p, name) for p in jets], 1
                )
            lookup.update((key, start + i) for i, key in enumerate(missing))
        ids = torch.tensor([lookup[key] for key in keys], device=space.device)
        selected = {name: storage[ids] for name, storage in table.items()}
        self.misses += len(missing)
        self.hits += len(keys) - len(missing)
        return [
            DualForm(
                selected["c"][:, j],
                selected["g"][:, j],
                selected["r"][:, j],
                h,
            )
            for j in range(len(orders))
        ]

class CauchyProfileJetCache(ProfileJetCache):
    def __init__(self, capacity=131072, degree=32, batch=64):
        super().__init__(capacity)
        if degree < 6 or batch < 1:
            raise ValueError("degree >=6 and positive polynomial batch required")
        self.degree, self.batch = degree, batch
        self.polynomials = FrozenPolynomialCache(capacity)

    def evaluate(self, source, space, h, compute, orders=None, periods=None):
        orders = tuple(
            orders if orders is not None else source.metadata.get("orders", ((0, 0),))
        )
        periods = tuple(
            periods
            if periods is not None
            else source.metadata.get("lengths", (1.0, 1.0))
        )

        def calculate(x, width):
            parts = [
                profile_forms(
                    source,
                    x[i : i + self.batch],
                    width[i : i + self.batch],
                    self.degree,
                    frozen_profiles=self.polynomials,
                    orders=orders,
                    periods=periods,
                )
                for i in range(0, len(x), self.batch)
            ]
            hh = torch.stack((width * 0, width), -1)
            return [
                DualForm(
                    torch.cat([p[j].c for p in parts]),
                    torch.cat([p[j].g for p in parts]),
                    torch.cat([p[j].r for p in parts]),
                    hh,
                )
                for j in range(len(orders))
            ]

        return super().evaluate(source, space, h, calculate, orders, periods)
