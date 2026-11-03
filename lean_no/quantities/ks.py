import gc
import json
import math
import time

import torch

from lean_no.enclosures.cauchy_jet import CauchyProfileJetCache
from lean_no.graph import add, affine, fourier
from lean_no.propagation.derivative_cells import enclose_cells, enclose_jet
from lean_no.propagation.rollout import values
from lean_no.quantities import Residual
from lean_no.refine import audit_cover, refine

split = 32 / 34
time_period = 0.1 / split
space_period = 6 * math.pi
orders = ((0, 0), (1, 0), (0, 1), (0, 2), (0, 3), (0, 4))
geometry = dict(
    split=split,
    lengths=(time_period, space_period),
    orders=orders,
    native_shape=(34, 128),
)

def ks_residual(jet, viscosity=0.01):
    return Residual("ks", viscosity).evaluate(dict(zip(orders, jet)))

def point_witness(output, time, space):
    nominal, actual = enclose_jet(
        output,
        time,
        space,
        paired=True,
        orders=orders,
        periods=(time_period, space_period),
        split=split,
    )
    residual = ks_residual(actual)
    lower = torch.maximum(residual.lower, -residual.upper).clamp(min=0)[:, 0, 0]
    index = int(lower.argmax())
    return {
        "absolute_lower": float(lower[index]),
        "t": float(time[index] * time_period),
        "x": float(space[index] * space_period),
        "residual_lower": float(residual.lower[index, 0, 0]),
        "residual_upper": float(residual.upper[index, 0, 0]),
        "nominal_derivatives": {
            name: float(j.lower[index, 0, 0])
            for name, j in zip(("u", "u_t", "u_x", "u_xx", "u_xxx", "u_xxxx"), nominal)
        },
    }

@torch.no_grad()
def terminal_residual_l2(output, profile_cache=None):
    ref = output.error
    base = 2048

    def bound(level, indices):
        lows, highs = [], []
        for lv, ids in zip(level.split(2048), indices.split(2048)):
            count = base * torch.exp2(lv.to(ref))
            x = (ids.to(ref) + 0.5) / count
            forms = enclose_cells(
                output,
                torch.full_like(x, split),
                x,
                0.0,
                0.5 / count,
                profile_cache=profile_cache,
                orders=orders,
                periods=(time_period, space_period),
                split=split,
                return_forms=True,
            )
            square = ks_residual(forms).spatial_square_polynomial(spatial_axis=1)
            mean, error = square[:, 0], square[:, 3]
            lows.append(((mean - error).clamp_min(0) / count).double().cpu())
            highs.append(((mean + error) / count).double().cpu())
        return torch.cat(lows), torch.cat(highs)

    level = torch.zeros(base, dtype=torch.int64)
    indices = torch.arange(base)
    lows, highs = bound(level, indices)
    while True:
        lower = math.sqrt(space_period * math.fsum(lows.tolist()))
        upper = math.sqrt(space_period * math.fsum(highs.tolist()))
        gap = (upper - lower) / lower if lower else (0.0 if upper == 0 else math.inf)
        print(
            f"terminal L2: {len(level)} cells, [{lower:.6g}, {upper:.6g}], gap {gap:.2%}",
            flush=True,
        )
        if gap <= 0.099:
            break
        if len(level) >= 131072 or int(level.max()) >= 20:
            raise ArithmeticError("terminal KS L2 cover did not reach 9.9% gap")
        widths = highs - lows
        ranked = widths.argsort(descending=True)
        cumulative = widths[ranked].cumsum(0)
        count = min(1024, int(torch.searchsorted(cumulative, widths.sum() / 2)) + 1)
        selected = ranked[:count]
        keep = torch.ones(len(level), dtype=torch.bool)
        keep[selected] = False
        children = (2 * indices[selected, None] + torch.arange(2)).flatten()
        depths = (level[selected] + 1).repeat_interleave(2)
        child_low, child_high = bound(depths, children)
        level = torch.cat((level[keep], depths))
        indices = torch.cat((indices[keep], children))
        lows = torch.cat((lows[keep], child_low))
        highs = torch.cat((highs[keep], child_high))
    depth = int(level.max())
    complete = int(
        torch.bitwise_left_shift(torch.ones_like(level), depth - level).sum()
    )
    if complete != base * 2**depth:
        raise ArithmeticError("incomplete terminal KS spatial cover")
    return dict(
        l2_lower=lower,
        l2_upper=upper,
        relative_gap=gap,
        local_time=split * time_period,
        spatial_domain=[0.0, space_period],
        spatial_cells=len(level),
        maximum_depth=depth,
        complete_spatial_cover=True,
        trace="active-left",
        roundoff_enclosed=False,
    )

@torch.no_grad()
def initial_mismatch(output, initial, profile_cache=None):
    if output.error.numel() != 1 or initial.error.numel() != 1:
        raise ValueError("scalar KS fields required")
    ref = output.error
    difference = add(output, affine(initial, -ref.new_ones(1, 1), ref.new_zeros(1)))
    cells = 2048
    centers = (torch.arange(cells, device=ref.device, dtype=ref.dtype) + 0.5) / cells
    lower, upper, witness = 0.0, 0.0, None
    for x in centers.split(256):
        t = torch.zeros_like(x)
        _, actual = enclose_cells(
            difference,
            t,
            x,
            0.0,
            0.5 / cells,
            profile_cache=profile_cache,
            orders=orders,
            periods=(time_period, space_period),
            split=split,
        )
        cap = torch.nan_to_num(
            actual[0].magnitude(), nan=torch.inf, posinf=torch.inf, neginf=torch.inf
        )
        upper = max(upper, float(cap.max()))
        interval = actual[0]
        score = torch.maximum(interval.lower, -interval.upper).clamp_min(0).flatten()
        valid = (
            torch.isfinite(interval.lower) & torch.isfinite(interval.upper)
        ).flatten()
        if not valid.any():
            continue
        score = torch.where(valid, score, -torch.inf)
        index = int(score.argmax())
        if witness is None or float(score[index]) > lower:
            lower = float(score[index])
            witness = dict(
                x=float(x[index] * space_period),
                difference_lower=float(interval.lower.flatten()[index]),
                difference_upper=float(interval.upper.flatten()[index]),
            )
    return dict(
        linf_lower=lower,
        linf_upper=max(lower, upper),
        witness=witness,
        spatial_cells=cells,
        normalized_domain=[0, 1],
        time=0.0,
        complete_uniform_cover=True,
        roundoff_enclosed=False,
    )

def adaptive_cells(
    field,
    base,
    level,
    indices,
    parent_upper=None,
    profile_cache=None,
):
    ref = field.error
    scale = torch.exp2(level.to(ref))
    nt, nx = base[0] * scale, base[1] * scale
    t = (indices[:, 0].to(ref) + 0.5) * split / nt
    x = (indices[:, 1].to(ref) + 0.5) / nx
    from lean_no.quantities.cells import evaluate_cells

    centers = torch.stack((t, x), -1)
    widths = torch.stack((split / (2 * nt), 1 / (2 * nx)), -1)

    def evaluate(points, radii):
        forms = field.enclose(points, radii, orders=orders, profile_cache=profile_cache)
        return Residual("ks", 0.01).evaluate(dict(zip(orders, forms)))

    upper, candidate = evaluate_cells(centers, widths, evaluate, parent_upper)
    point = ref.new_tensor(candidate["point"])
    witness = point_witness(field, point[:1], point[1:])
    return upper, witness

def cell_bounds(
    field,
    base,
    level,
    indices,
    parent_upper=None,
    max_batch=65536,
    profile_cache=None,
):
    parts, witness = [], dict(absolute_lower=0.0)
    begin, chunk = 0, min(max_batch, len(indices))
    if chunk < 1:
        raise ValueError("cell batch must be positive and nonempty")
    while begin < len(indices):
        end = min(begin + chunk, len(indices))
        failed = False
        try:
            upper, candidate = adaptive_cells(
                field,
                base,
                level[begin:end],
                indices[begin:end],
                None if parent_upper is None else parent_upper[begin:end],
                profile_cache=profile_cache,
            )
        except torch.OutOfMemoryError:
            if chunk == 1:
                raise
            failed = True
        if failed:
            # Release exception frames before releasing their CUDA tensors.
            gc.collect()
            torch.cuda.empty_cache()
            chunk = max(1, chunk // 2)
            print(
                json.dumps(dict(phase="reducing enclosure batch", cells=chunk)),
                flush=True,
            )
            continue
        parts.append(upper)
        if candidate["absolute_lower"] >= witness["absolute_lower"]:
            witness = candidate
        begin = end
    return torch.cat(parts), witness

def validate_witness_autograd(field, witness):
    t = field.error.new_tensor([witness["t"] / time_period]).requires_grad_()
    x = field.error.new_tensor([witness["x"] / space_period]).requires_grad_()
    with torch.enable_grad():
        scalar = values(field, t[None], x[None], active=True)[0, 0, 0, 0]
        ut = torch.autograd.grad(scalar, t, create_graph=True)[0] / time_period
        exact = [scalar, ut[0]]
        derivative = scalar
        for order in range(1, 5):
            derivative = torch.autograd.grad(derivative.sum(), x, create_graph=True)[0]
            exact.append(derivative[0] / space_period**order)
    nominal, actual = enclose_jet(
        field,
        t.detach(),
        x.detach(),
        paired=True,
        orders=orders,
        periods=(time_period, space_period),
        split=split,
    )
    errors = []
    for reference, interval in zip(exact, nominal):
        torch.testing.assert_close(
            interval.lower.reshape(()), reference.detach(), rtol=2e-7, atol=2e-7
        )
        errors.append(float((interval.lower.reshape(()) - reference.detach()).abs()))
    residual = exact[1] + exact[0] * exact[2] + exact[3] + 0.01 * exact[5]
    enclosed = ks_residual(actual)
    tolerance = 2e-7 * (1 + abs(float(residual.detach())))
    # This is a nominal AD diagnostic. Actual interval membership additionally
    # carries all coefficient and normalization uncertainty from earlier windows.
    return dict(
        nominal_residual=float(residual.detach()),
        nominal_derivative_max_difference=max(errors),
        nominal_residual_in_actual_interval=(
            float(enclosed.lower.detach()) - tolerance
            <= float(residual.detach())
            <= float(enclosed.upper.detach()) + tolerance
        ),
    )

@torch.no_grad()
def ks_bounds(
    field,
    quantity,
    index=1,
    batch=16384,
    enclosure_batch=4096,
    profile_cache=None,
    upper_target=None,
    progress=None,
    max_seconds=None,
):
    if not isinstance(quantity, Residual) or quantity.pde != "ks":
        raise ValueError("KS residual certification requires Residual for ks")
    if quantity.coefficient not in (None, 0.01):
        raise ValueError("KS residual certification supports viscosity 0.01 only")
    if upper_target is not None and (
        not math.isfinite(upper_target) or upper_target <= 0
    ):
        raise ValueError("residual upper target must be finite and positive")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds < 0):
        raise ValueError("residual time budget must be finite and nonnegative")
    if profile_cache is None:
        profile_cache = CauchyProfileJetCache(capacity=131072, degree=32)
    started = time.perf_counter()
    base = (64, 128)
    indices = torch.cartesian_prod(torch.arange(base[0]), torch.arange(base[1]))

    def bound(level, indices, parent_upper=None):
        return cell_bounds(
            field,
            base,
            level,
            indices,
            parent_upper,
            max_batch=enclosure_batch,
            profile_cache=profile_cache,
        )

    upper, witness = bound(torch.zeros(len(indices), dtype=torch.int64), indices)
    leaves = [dict(level=0, indices=indices, upper=upper)]
    while True:
        lower = witness["absolute_lower"]
        upper = max(lower, max(float(leaf["upper"].max()) for leaf in leaves))
        gap = (
            (upper - lower) / lower if lower else (0.0 if upper == 0 else float("inf"))
        )
        if progress is not None:
            progress(
                dict(
                    linf_lower=lower,
                    linf_upper=upper,
                    relative_gap=gap,
                    leaf_count=sum(len(leaf["indices"]) for leaf in leaves),
                )
            )
        if (
            gap <= 0.099
            or (upper_target is not None and upper < upper_target)
            or (
                max_seconds is not None and time.perf_counter() - started >= max_seconds
            )
        ):
            break
        leaves, witness, reason = refine(bound, leaves, witness, 10.0, batch)
        if reason == "integer coverage depth limit reached":
            raise ArithmeticError(reason)
    coverage = audit_cover(leaves, *base)
    validation = validate_witness_autograd(field, witness)
    if not validation["nominal_residual_in_actual_interval"]:
        raise ArithmeticError("nominal AD residual lies outside the residual enclosure")
    print(f"rollout {index}: [{lower:.6g}, {upper:.6g}], gap {gap:.2%}", flush=True)
    report = dict(
        rollout=index,
        linf_lower=lower,
        linf_upper=upper,
        relative_gap=gap,
        coverage=coverage,
        witness=witness,
        witness_autograd_validation=validation,
        roundoff_enclosed=False,
        seconds=time.perf_counter() - started,
        stopping_rule="upper target"
        if upper_target is not None and upper < upper_target
        else "relative gap"
        if gap <= 0.099
        else "time budget",
        upper_target=upper_target,
        max_seconds=max_seconds,
    )
    from lean_no.quantities.norms import ResidualBounds

    return ResidualBounds(
        "ks", (lower, upper), (0.0, 0.0), field.periods, dict(residual=coverage)
    ), report

def paper_profile(device="cpu", inputs=None):
    if inputs is not None:
        from lean_no.function_space.analytic import Series, describe
        from lean_no.graph import make_field

        series = describe(inputs)
        source = Series(
            tuple(((0.0, k[0]), c) for k, c in series.terms), (0.0, 0.0)
        ).field((time_period, space_period), device)
        return make_field(
            source.kind,
            dict(source.data),
            source.parents,
            source.error,
            source.sizes,
            geometry,
        )
    modes = torch.arange(3, device=device, dtype=torch.float64)
    c = torch.tensor([0, -0.5j, 0.25], device=device, dtype=torch.complex128)
    return fourier(c[None, None], (modes[:1], modes), modes.new_zeros(1), geometry)
