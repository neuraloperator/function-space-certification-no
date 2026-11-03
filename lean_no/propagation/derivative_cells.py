import itertools
import math
from dataclasses import dataclass, field

import torch

from lean_no.adapters.fno import build_graph
from lean_no.bounds.activations import gelu_range
from lean_no.bounds.affine import DualForm, compose, tanh_compose, tanh_polynomial
from lean_no.bounds.derivatives import compose_jet
from lean_no.bounds.interval import Interval, interval_linear
from lean_no.graph import Graph, execute, walk
from lean_no.spectral.fourier import spectral_disk, spectral_forms

def derivative_orders(dimension, degree):
    return tuple(
        sorted(
            (
                a
                for a in itertools.product(range(degree + 1), repeat=dimension)
                if sum(a) <= degree
            ),
            key=lambda a: (sum(a), tuple(reversed(a))),
        )
    )

orders_2d = derivative_orders(2, 2)

def tanh_range(value, order):
    t = Interval(value.lower.tanh(), value.upper.tanh())
    result = Interval.around(torch.zeros_like(value.lower))
    for coefficient in reversed(tanh_polynomial(order)):
        result = result * t + coefficient
    return result

def relu_range(value, order):
    if order == 0:
        return Interval(value.lower.relu(), value.upper.relu())
    if order == 1:
        return Interval(
            (value.lower > 0).to(value.lower), (value.upper >= 0).to(value.upper)
        )
    raise ValueError(
        "higher ReLU derivatives do not exist across activation boundaries"
    )

@dataclass
class InputField:

    bounds: object
    periods: tuple
    error: torch.Tensor

    @property
    def metadata(self):
        return {"lengths": self.periods}

    def enclose(self, centers, halfwidth, orders):
        points = centers * centers.new_tensor(self.periods)
        physical = halfwidth * halfwidth.new_tensor(self.periods)
        query = points[:, 0] if len(self.periods) == 1 else points.T
        lower, upper = (
            self.bounds.lower(query).T.to(centers),
            self.bounds.upper(query).T.to(centers),
        )
        if lower.shape != (len(centers), len(self.error)) or (lower > upper).any():
            raise ValueError("invalid input band enclosure")
        slopes = self.bounds.lipschitz.to(centers).reshape(len(self.error), -1)
        if slopes.shape[-1] != len(self.periods):
            raise ValueError("input Lipschitz bounds must match the physical axes")
        forms = []
        for order in orders:
            degree = sum(order)
            if degree == 0:
                radius = physical @ slopes.T
                value = Interval(lower - radius, upper + radius)
            elif degree == 1:
                magnitude = slopes[:, order.index(1)]
                value = Interval.around(
                    centers.new_zeros(len(centers), len(self.error)), magnitude
                )
            elif degree == 2 and self.bounds.curvature is not None:
                axes = [a for a, power in enumerate(order) for _ in range(power)]
                magnitude = self.bounds.curvature.to(centers).reshape(
                    len(self.error), len(self.periods), len(self.periods)
                )[:, axes[0], axes[1]]
                value = Interval.around(
                    centers.new_zeros(len(centers), len(self.error)), magnitude
                )
            elif (
                len(self.periods) == 1 and self.bounds.is_exact and self.bounds.symbolic
            ):
                from flint import arb, ctx

                rows = []
                with ctx.workprec(80):
                    for expression in self.bounds.lower_expr:
                        derivative = expression.diff(degree)
                        values = [
                            derivative.eval_arb(arb(float(x), float(r)))
                            for x, r in zip(points[:, 0], physical[:, 0])
                        ]
                        rows.append(
                            (
                                [float(v.lower()) for v in values],
                                [float(v.upper()) for v in values],
                            )
                        )
                value = Interval(
                    centers.new_tensor([row[0] for row in rows]).T,
                    centers.new_tensor([row[1] for row in rows]).T,
                )
            else:
                raise ValueError(f"input regularity for derivative {order} is required")
            forms.append(
                Interval(
                    value.lower[..., None].expand(-1, -1, 2),
                    value.upper[..., None].expand(-1, -1, 2),
                )
            )
        return forms

def constant_form(center, gradient, halfwidth, arithmetic):
    if arithmetic == "affine":
        return DualForm(center, gradient, center.new_zeros(*center.shape, 2), halfwidth)
    radius = (gradient.abs() * halfwidth[:, None]).sum(-1)
    return Interval.around(
        center[..., None].expand(*center.shape, 2), radius[..., None]
    )

def concatenate(parts, halfwidth, arithmetic):
    if arithmetic == "affine":
        return DualForm(
            torch.cat([p.c for p in parts], -1),
            torch.cat([p.g for p in parts], -2),
            torch.cat([p.r for p in parts], -2),
            halfwidth,
        )
    return Interval(
        torch.cat([p.lower for p in parts], -2), torch.cat([p.upper for p in parts], -2)
    )

def initial_blend_jet(initial, learned, tau, orders, duration):
    result = [a * (tau * -1 + 1) + b * tau for a, b in zip(initial, learned)]
    for j, order in enumerate(orders):
        if order[0]:
            lower = orders.index((order[0] - 1, *order[1:]))
            result[j] = result[j] + (learned[lower] + initial[lower] * -1) * (
                order[0] / duration
            )
    return result

def apply_jets(op, parents, orders, halfwidth, arithmetic, data=None):
    source = parents[0]
    data = op.data if data is None else data
    if op.kind == "add":
        return [a + b for a, b in zip(*parents)]
    if op.kind == "pad":
        return list(source)
    if op.kind == "affine":
        weight, bias = data["weight"], data["bias"]
        result = []
        for j, value in enumerate(source):
            shift = bias if j == 0 else 0
            if arithmetic == "affine":
                mapped = value.linear(weight, shift)
            else:
                mapped = Interval(
                    *interval_linear(
                        weight,
                        value.lower,
                        value.upper,
                        lambda w, v: torch.einsum("ncl,oc->nol", v, w),
                    )
                )
                mapped = mapped + (bias[None, :, None] if j == 0 else 0)
            if "scale_error" in data and j:
                if arithmetic == "affine":
                    mapped.r[..., 1] += (
                        data["scale_error"] * value.interval().magnitude()[..., 1]
                    )
                else:
                    radius = data["scale_error"] * value.magnitude()[..., 1]
                    mapped += Interval.around(
                        torch.zeros_like(mapped.lower),
                        torch.stack((radius * 0, radius), -1),
                    )
            result.append(mapped)
        return result
    if op.kind == "activation":
        name = data["activation"]
        if name == "identity":
            return source
        rule = (
            {"tanh": tanh_compose, "gelu": compose}
            if arithmetic == "affine"
            else {"tanh": tanh_range, "gelu": gelu_range, "relu": relu_range}
        )[name]
        return compose_jet(source, orders, rule)
    raise NotImplementedError(op.kind)

def field_jets(
    output,
    centers,
    halfwidth,
    orders,
    periods,
    arithmetic="affine",
    profile_cache=None,
    split=1.0,
):
    if (
        centers.shape != halfwidth.shape
        or centers.ndim != 2
        or centers.shape[1] != len(periods)
    ):
        raise ValueError("centers and half-widths must match the domain dimension")
    tolerance = 8 * torch.finfo(centers.dtype).eps
    edge = centers.new_ones(centers.shape[1])
    edge[0] = split
    if (
        not torch.isfinite(centers).all()
        or not torch.isfinite(halfwidth).all()
        or bool(
            (halfwidth < 0).any()
            | (centers - halfwidth < -tolerance).any()
            | (centers + halfwidth > edge + tolerance).any()
        )
    ):
        raise ValueError("cell outside active domain")

    def apply(node, visit):
        data = node.data
        if node.kind == "spectral":
            forms = spectral_forms(node, centers, halfwidth, orders, periods)
            return (
                forms
                if arithmetic == "affine"
                else [value.interval() for value in forms]
            )
        if node.kind == "profile":
            source = data.get("derivative_field")
            if source is None:
                if bool((node.error != 0).any()):
                    raise ValueError("profile requires derivative-preserving handoff")
                source = data["nominal"]
            if profile_cache is not None and arithmetic == "affine":
                return profile_cache.evaluate(
                    source,
                    centers[:, 1],
                    halfwidth,
                    lambda x, h: field_jets(
                        source,
                        torch.stack((x * 0, x), -1),
                        torch.stack((h * 0, h), -1),
                        orders,
                        periods,
                        arithmetic,
                        profile_cache,
                        split,
                    ),
                    orders=orders,
                    periods=periods,
                )
            return visit(source)
        if node.kind == "input":
            source = visit(data["u"])
            result = []
            for j, order in enumerate(orders):
                coordinates = []
                for axis, scale in enumerate((data["tp"], data["lx"])):
                    center = (
                        centers[:, axis, None] * scale
                        if not any(order)
                        else (
                            centers[:, axis, None] * 0
                            + (
                                order
                                == tuple(int(i == axis) for i in range(len(periods)))
                            )
                            * scale
                            / periods[axis]
                        )
                    )
                    gradient = centers.new_zeros(len(centers), 1, len(periods))
                    if not any(order):
                        gradient[..., axis] = scale
                    coordinates.append(
                        constant_form(center, gradient, halfwidth, arithmetic)
                    )
                result.append(
                    concatenate([source[j], *coordinates], halfwidth, arithmetic)
                )
            return result
        parents = [visit(parent) for parent in node.parents]
        if node.kind == "initial_blend":
            gradient = centers.new_zeros(len(centers), 1, len(periods))
            gradient[..., 0] = 1 / split
            tau = constant_form(centers[:, :1] / split, gradient, halfwidth, arithmetic)
            result = initial_blend_jet(*parents, tau, orders, split * periods[0])
        else:
            result = apply_jets(
                node.operation, parents, orders, halfwidth, arithmetic, node.data
            )
        value = result[0]
        if arithmetic == "affine":
            nominal = value.r[..., 0]
            actual = nominal + node.error
            if node.kind == "initial_blend":
                actual = torch.minimum(value.r[..., 1], actual)
            result[0] = DualForm(
                value.c, value.g, torch.stack((nominal, actual), -1), halfwidth
            )
        else:
            low, high = (
                value.lower[..., 0] - node.error,
                value.upper[..., 0] + node.error,
            )
            if node.kind == "initial_blend":
                low, high = (
                    torch.maximum(low, value.lower[..., 1]),
                    torch.minimum(high, value.upper[..., 1]),
                )
            result[0] = Interval(
                torch.stack((value.lower[..., 0], low), -1),
                torch.stack((value.upper[..., 0], high), -1),
            )
        return result

    return walk(apply)(output)

def enclose_cells(
    output,
    time,
    space,
    half_time=0.0,
    half_space=0.0,
    profile_cache=None,
    return_forms=False,
    orders=None,
    periods=None,
    split=None,
):
    orders = tuple(
        orders if orders is not None else output.metadata.get("orders", ((0, 0),))
    )
    periods = tuple(
        periods if periods is not None else output.metadata.get("lengths", (1.0, 1.0))
    )
    split = output.metadata.get("split", 1.0) if split is None else split
    if time.ndim != 1 or space.shape != time.shape:
        raise ValueError("paired coordinate vectors required")
    centers = torch.stack((time, space), -1)
    widths = torch.stack(
        (
            torch.as_tensor(half_time, device=time.device, dtype=time.dtype).expand_as(
                time
            ),
            torch.as_tensor(
                half_space, device=space.device, dtype=space.dtype
            ).expand_as(space),
        ),
        -1,
    )
    forms = field_jets(
        output, centers, widths, orders, periods, "affine", profile_cache, split
    )
    return (
        forms
        if return_forms
        else tuple([form.lane(i).interval() for form in forms] for i in (0, 1))
    )

def enclose_jet(
    output,
    time,
    space,
    half_time=0.0,
    half_space=0.0,
    paired=False,
    orders=None,
    periods=None,
    split=None,
):
    orders = tuple(
        orders if orders is not None else output.metadata.get("orders", ((0, 0),))
    )
    periods = tuple(
        periods if periods is not None else output.metadata.get("lengths", (1.0, 1.0))
    )
    split = output.metadata.get("split", 1.0) if split is None else split
    if paired:
        centers = torch.stack((time, space), -1)
        shape = (len(time), 1)
    else:
        centers = torch.stack(torch.meshgrid(time, space, indexing="ij"), -1).reshape(
            -1, 2
        )
        shape = (len(time), len(space))
    widths = torch.stack(
        (
            torch.as_tensor(half_time, device=time.device, dtype=time.dtype).expand(
                len(centers)
            ),
            torch.as_tensor(half_space, device=space.device, dtype=space.dtype).expand(
                len(centers)
            ),
        ),
        -1,
    )
    jets = field_jets(output, centers, widths, orders, periods, "interval", split=split)
    return tuple(
        [
            Interval(
                value.lower[..., lane].reshape(*shape, -1),
                value.upper[..., lane].reshape(*shape, -1),
            )
            for value in jets
        ]
        for lane in (0, 1)
    )

@dataclass
class DerivativeField:
    network: Graph
    forcing: object
    arithmetic: str = "affine"
    coefficients: dict = field(default_factory=dict)
    records: list = field(default_factory=list)
    time_period: float | None = None
    requested: tuple = ()
    resolution: tuple = ()
    integration: str = "gaussian"

    @property
    def domain(self):
        from lean_no.domain import PeriodicBox

        return PeriodicBox(
            self.periods, ("x", "t") if self.time_period is not None else ()
        )

    @property
    def periods(self):
        return self.forcing.metadata.get(
            "lengths", (1.0,) * self.network.metadata["dimension"]
        )

    @property
    def orders(self):
        degree = max(map(sum, self.requested), default=2)
        return derivative_orders(len(self.periods), degree)

    @classmethod
    def from_spatial(cls, model, forcing):
        return cls(build_graph(model, torch.float64), forcing)

    @classmethod
    def from_profile(cls, model, initial, time_period=17 / 16):
        from lean_no.function_space.analytic import Series

        model = build_graph(model, torch.float64)
        if (
            model.dimension != 2
            or model.in_channels != 2
            or model.out_channels != 1
            or model.grid_embedding
        ):
            raise ValueError("space-time fields require an eval-mode two-input FNO")
        if (
            not isinstance(initial, Series)
            or any(initial.slopes)
            or len(initial.slopes) != 1
        ):
            raise ValueError("an exact periodic trigonometric profile is required")
        if any(k[0] != round(k[0]) for k, value in initial.terms):
            raise ValueError(
                "Burgers requires a periodic profile on the unit spatial interval"
            )
        if not math.isfinite(time_period) or time_period <= 0:
            raise ValueError("positive finite time period required")
        network = build_graph(model, torch.float64)
        series = Series(tuple(((k[0], 0.0), c) for k, c in initial.terms), (0.0, 0.0))
        source = series.field((1.0, time_period), network.device, network.dtype)
        return cls(network, source, time_period=time_period)

    def input_forms(self, centers, halfwidth, orders):
        arithmetic = self.arithmetic
        if isinstance(self.forcing, InputField):
            jets = self.forcing.enclose(centers, halfwidth, orders)
        else:
            forms = spectral_forms(
                self.forcing, centers, halfwidth, orders, self.periods
            )
            jets = (
                forms
                if arithmetic == "affine"
                else [value.interval() for value in forms]
            )
        axes = [1] if self.time_period is not None else []
        scales = [self.time_period] if self.time_period is not None else []
        if self.network.grid_embedding:
            axes.extend(range(len(self.periods)))
            scales.extend([1.0] * len(self.periods))
        for j, order in enumerate(orders):
            parts = [jets[j]]
            for axis, scale in zip(axes, scales):
                center = (
                    centers[:, axis, None] * scale
                    if not any(order)
                    else (
                        centers[:, axis, None] * 0
                        + (
                            order
                            == tuple(int(i == axis) for i in range(len(self.periods)))
                        )
                        * scale
                        / self.periods[axis]
                    )
                )
                gradient = centers.new_zeros(len(centers), 1, len(self.periods))
                if not any(order):
                    gradient[..., axis] = scale
                parts.append(constant_form(center, gradient, halfwidth, arithmetic))
            jets[j] = concatenate(parts, halfwidth, arithmetic)
        return jets

    @torch.no_grad()
    def enclose(
        self,
        centers,
        halfwidth,
        stop=None,
        values_only=False,
        orders=None,
        coefficients=None,
    ):
        dimension = len(self.periods)
        if (
            centers.ndim != 2
            or centers.shape[1] != dimension
            or halfwidth.shape != centers.shape
        ):
            raise ValueError("cells must match the field dimension")
        tolerance = 8 * torch.finfo(centers.dtype).eps
        if (
            not torch.isfinite(centers).all()
            or not torch.isfinite(halfwidth).all()
            or bool(
                (halfwidth < 0).any()
                | (centers - halfwidth < -tolerance).any()
                | (centers + halfwidth > 1 + tolerance).any()
            )
        ):
            raise ValueError("cell outside the domain")
        orders = (
            tuple(orders)
            if orders is not None
            else self.orders[:1]
            if values_only
            else self.orders
        )
        jets = self.input_forms(centers, halfwidth, orders)

        def apply(op, parents):
            if op.kind == "spectral":
                proof = (self.coefficients if coefficients is None else coefficients)[
                    op.name
                ]
                if isinstance(proof, dict):
                    forms = []
                    points = centers * centers.new_tensor(self.periods)
                    query = points[:, 0] if dimension == 1 else points.T.contiguous()
                    for order in orders:
                        bounds = proof[order]
                        lower, upper = bounds.bounds(query)
                        slopes = bounds.partial_magnitude.flatten(2).amax(-1)
                        radius = (
                            halfwidth * halfwidth.new_tensor(self.periods)
                        ) @ slopes.T
                        lower, upper = lower.T - radius, upper.T + radius
                        forms.append(
                            Interval(
                                lower[..., None].expand(-1, -1, 2),
                                upper[..., None].expand(-1, -1, 2),
                            )
                        )
                    return forms
                forms = spectral_forms(proof, centers, halfwidth, orders, self.periods)
                return (
                    forms
                    if self.arithmetic == "affine"
                    else [value.interval() for value in forms]
                )
            return apply_jets(op, parents, orders, halfwidth, self.arithmetic)

        result = execute(
            self.network if stop is None else Graph(self.network.ops[:stop]),
            jets,
            apply,
        )
        if self.arithmetic == "interval":
            return [
                Interval(value.lower[..., 1], value.upper[..., 1]) for value in result
            ]
        return result

    def input_disks(self, centers, radii):
        if isinstance(self.forcing, InputField):
            value = self.forcing.enclose(
                centers, radii[None].expand_as(centers), ((0,) * len(self.periods),)
            )[0]
            center, radius = (
                value.center[..., 1],
                (value.upper[..., 1] - value.lower[..., 1]) / 2,
            )
        else:
            center, radius = spectral_disk(self.forcing, centers, radii)
        if self.time_period is not None:
            center = torch.cat((center, centers[:, 1, None] * self.time_period), -1)
            radius = torch.cat(
                (
                    radius,
                    torch.ones_like(centers[:, 1, None]) * radii[1] * self.time_period,
                ),
                -1,
            )
        return center, radius

    @torch.no_grad()
    def certify_coefficients(
        self,
        order=16,
        batch=8192,
        progress=None,
        panels=None,
        integration="gaussian",
        **kwargs,
    ):
        from lean_no.integration.coefficients import prepare_coefficients

        self.coefficients, self.records = prepare_coefficients(
            self, order, batch, progress, panels, integration, **kwargs
        )
        self.integration = integration
        return self

    @property
    def stats(self):
        return {"quadrature_order": self.records[-1]["order"] if self.records else 0}

    @property
    def node_bounds(self):
        ref = self.forcing.error
        axes = [
            torch.arange(n, device=ref.device, dtype=ref.dtype) / n
            for n in self.resolution
        ]
        centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).flatten(0, -2)
        parts = [[], []]
        for points in centers.split(8192):
            value = self.enclose(points, torch.zeros_like(points), values_only=True)[0]
            parts[0].append(value.lower.T)
            parts[1].append(value.upper.T)
        return tuple(
            torch.cat(values, 1).reshape(-1, *self.resolution) for values in parts
        )

    @property
    def node_lower(self):
        return self.node_bounds[0]

    @property
    def node_upper(self):
        return self.node_bounds[1]

    @property
    def cell_bounds(self):
        if not self.resolution:
            raise ValueError("no verification partition supplied")
        lower, upper = [], []
        ref = self.forcing.error
        for start in range(0, math.prod(self.resolution), 8192):
            ids = torch.arange(
                start, min(start + 8192, math.prod(self.resolution)), device=ref.device
            )
            centers = torch.stack(
                [
                    ((ids // math.prod(self.resolution[a + 1 :])) % n + 0.5) / n
                    for a, n in enumerate(self.resolution)
                ],
                -1,
            ).to(ref)
            halfwidth = centers.new_tensor(
                [0.5 / n for n in self.resolution]
            ).expand_as(centers)
            value = self.enclose(centers, halfwidth, values_only=True)[0].actual
            lower.append(value.lower.T)
            upper.append(value.upper.T)
        return tuple(
            torch.cat(parts, 1).reshape(-1, *self.resolution)
            for parts in (lower, upper)
        )

    @torch.no_grad()
    def query(self, points):
        from lean_no.certificate import QueryBounds
        from lean_no.integration.coefficients import source_disk

        centers = self.domain.normalize(points.to(self.forcing.error))
        if isinstance(self.forcing, InputField) or any(
            name not in ("tanh", "identity") for name in self.network.activations
        ):
            value = self.enclose(centers, torch.zeros_like(centers), values_only=True)[
                0
            ].actual
            return QueryBounds.from_bounds(value.lower, value.upper)
        center, radius = source_disk(
            self,
            centers,
            centers.new_zeros(len(self.periods)),
            len(self.network.ops),
            self.coefficients,
        )
        return QueryBounds(center, radius)
