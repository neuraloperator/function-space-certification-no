import math
import time

import torch

from lean_no.enclosures.taylor import ProfileTaylorCache, polydisk
from lean_no.integration import integrate
from lean_no.integration.gaussian import AnalyticSource, GaussianOptions
from lean_no.propagation.rollout import FrozenProfileValues, values

class StripFailure(ArithmeticError):
    pass

def contour_cap(
    source,
    centers,
    radii,
    active,
    gelu_order=8,
    real_spatial=False,
    profile_degree=0,
    profile_models=None,
):
    v, g, r = polydisk(
        source,
        centers,
        radii,
        active,
        gelu_order=gelu_order,
        real_spatial=real_spatial,
        profile_degree=profile_degree,
        profile_models=profile_models,
    )
    return v, v.abs() + (g.abs() * radii[:, None]).sum(-1) + r

def adaptive_strip(
    source,
    start,
    end,
    rho,
    axis,
    cap_limit=1e20,
    max_depth=13,
    space_only=False,
    max_cells=262144,
    gelu_order=8,
    profile_degree=0,
    profile_models=None,
):
    ref = source.error
    active = start == 0
    bounds = [[start, end], [0.0, 1.0]]
    bounds[axis] = [bounds[axis][0] - rho, bounds[axis][1] + rho]
    maximum = torch.zeros_like(ref)
    records = []
    for side in ("horizontal", "left", "right"):
        left = ref.new_tensor([b[0] for b in bounds]).to(torch.complex128)
        direction = ref.new_tensor([b[1] - b[0] for b in bounds]).to(torch.complex128)
        if side == "horizontal":
            left[axis] += 1j * rho
        else:
            left[axis] = bounds[axis][0 if side == "left" else 1] - 1j * rho
            direction[axis] = 2j * rho
        it, ix = torch.meshgrid(
            torch.arange(4, device=ref.device),
            torch.arange(16, device=ref.device),
            indexing="ij",
        )
        pending = torch.stack((it.flatten(), ix.flatten()), -1)
        saved_cells = []
        saved_caps = []
        accepted_count = 0
        for depth in range(max_depth + 1):
            if len(pending) + accepted_count > max_cells:
                raise StripFailure(
                    "contour cell budget exhausted; refine Gaussian panels"
                )
            failures = []
            widths = ref.new_tensor(
                [1 / (4 * (1 if space_only else 2**depth)), 1 / (16 * 2**depth)]
            )
            cover = direction.abs() * widths / 2
            for ids in pending.split(8192):
                centers = left + (ids.to(ref.dtype) + 0.5) * widths * direction
                v, cap = contour_cap(
                    source,
                    centers,
                    cover[None].expand(len(ids), -1),
                    active,
                    gelu_order=gelu_order,
                    real_spatial=axis == 0,
                    profile_degree=profile_degree,
                    profile_models=profile_models,
                )
                # Reject a Gaussian disk whose center values already exceed the budget.
                if not torch.isfinite(v).all() or bool((v.abs() > cap_limit).any()):
                    raise StripFailure(
                        "complex center exceeds cap limit; refine Gaussian panels"
                    )
                good = torch.isfinite(cap).all(-1) & (cap.amax(-1) <= cap_limit)
                if good.any():
                    accepted_count += int(good.sum())
                    maximum = torch.maximum(maximum, cap[good].amax(0))
                    saved_cells.append(
                        torch.cat(
                            (torch.full_like(ids[good, :1], depth), ids[good]), -1
                        ).cpu()
                    )
                    saved_caps.append(cap[good].cpu())
                failures.append(ids[~good])
            failed = torch.cat(failures)
            if depth >= 4:
                print(
                    f"contour {axis} {side} depth {depth}: {len(pending) - len(failed)} accepted, {len(failed)} refine",
                    flush=True,
                )
            if not len(failed):
                break
            if depth == max_depth:
                raise StripFailure("contour cover depth exhausted")
            offsets = torch.tensor(
                [[0, 0], [0, 1]] if space_only else [[0, 0], [0, 1], [1, 0], [1, 1]],
                device=ref.device,
            )
            scale = torch.tensor([1 if space_only else 2, 2], device=ref.device)
            pending = (scale * failed[:, None] + offsets).reshape(-1, 2)
        records.append(
            dict(
                side=side,
                cells=torch.cat(saved_cells),
                caps=torch.cat(saved_caps),
                space_only=space_only,
                gelu_order=gelu_order,
                real_spatial=axis == 0,
                profile_degree=profile_degree,
            )
        )
    return maximum, records

class StripGauss:
    rule = "gaussian"

    def __init__(
        self,
        order=80,
        cells=((32, 64), (32, 64), (2048, 128), (4096, 128)),
        cap_limit=1e20,
        batch=8,
        progress=print,
        space_only=False,
        max_cover_cells=262144,
        gelu_order=8,
        profile_degree=0,
    ):
        self.order, self.cells, self.cap_limit, self.batch, self.progress = (
            order,
            cells,
            cap_limit,
            batch,
            progress,
        )
        self.records = []
        self.space_only = space_only
        self.gelu_order = gelu_order
        if profile_degree and profile_degree < 2:
            raise ValueError("profile degree must be zero or at least two")
        self.profile_degree = profile_degree
        self.profile_models = ProfileTaylorCache() if profile_degree else None
        if max_cover_cells < 64:
            raise ValueError("contour cell budget must allow the 64 initial cells")
        self.max_cover_cells = max_cover_cells

    @torch.no_grad()
    def __call__(self, source, layer):
        split = source.metadata.get("split", 1.0)
        order = (
            self.order[layer] if isinstance(self.order, (tuple, list)) else self.order
        )
        ref = source.error
        batch = self.batch
        if ref.device.type == "cuda" and batch > 64:
            seen = set()

            def count(node):
                if node in seen or node.kind == "profile":
                    return (0, 0)
                seen.add(node)
                channels = node.error.numel()
                total, maximum = (
                    channels * (2 if node.kind == "spectral" else 1),
                    channels,
                )
                dependencies = list(node.parents)
                if node.kind == "input":
                    dependencies.append(node.data["u"])
                for parent in dependencies:
                    n, m = count(parent)
                    total += n
                    maximum = max(maximum, m)
                return total, maximum

            channels, largest = count(source)
            free, total = torch.cuda.mem_get_info(ref.device)
            # DAG values plus matrix/activation temporaries; leave 30% headroom.
            bytes_per_point = 8 * (channels + 2 * largest + 128)
            safe = max(1, int(0.70 * free / (8192 * bytes_per_point)))
            batch = min(batch, max(1, (safe // 32) * 32))
            self.progress(
                f"block {layer}: time batch {batch}, free GPU GiB {free / 2**30:.1f}",
                flush=True,
            )
        started = time.perf_counter()
        nt, nx = source.metadata["modes"][layer]
        kt = torch.arange(-(nt // 2), nt - nt // 2, device=ref.device, dtype=ref.dtype)
        kx = torch.arange(nx, device=ref.device, dtype=ref.dtype)
        coefficients = torch.zeros(
            len(ref), nt, nx, device=ref.device, dtype=torch.complex128
        )
        error = torch.zeros_like(ref)
        segments = []
        for start, end in ((0.0, split), (split, 1.0)):
            if start == end:
                continue
            if source.kind == "pad" and start == split:
                continue
            nt, nx = self.cells[layer]
            if start:
                nt = max(1, math.ceil(nt * (1 - split) / split))
            counts = [nt, nx]
            caps, covers, radii = [], [], []
            for axis in (0, 1):
                for attempt in range(9):
                    rho = 2 * ((end - start) if axis == 0 else 1) / counts[axis]
                    self.progress(
                        f"block {layer} interval {start:.3g}: axis {axis} contour, panels {counts[axis]}",
                        flush=True,
                    )
                    try:
                        cap, cover = adaptive_strip(
                            source,
                            start,
                            end,
                            rho,
                            axis,
                            self.cap_limit,
                            max_depth=24 if self.space_only else 13,
                            space_only=self.space_only
                            or (layer == 0 and source.kind == "pad"),
                            max_cells=self.max_cover_cells,
                            gelu_order=self.gelu_order,
                            profile_degree=self.profile_degree,
                            profile_models=self.profile_models,
                        )
                        cap *= math.exp(
                            2
                            * math.pi
                            * float((kt if axis == 0 else kx).abs().max())
                            * rho
                        )
                        break
                    except StripFailure as exc:
                        self.progress(str(exc), flush=True)
                        counts[axis] *= 2
                else:
                    raise ArithmeticError("unable to certify analytic strip")
                caps.append(cap)
                covers.append(cover)
                radii.append(rho)
            profile_values = {}

            def evaluate(t, x):
                key = x.storage_offset()
                if key not in profile_values:
                    profile_values[key] = FrozenProfileValues(x[None])
                v = values(
                    source,
                    t[None],
                    x[None],
                    active=start == 0,
                    profile_cache=profile_values[key],
                )[0]
                return v, None

            self.progress(
                f"block {layer}: quadrature {counts}, {math.prod(counts) * order**2} nodes",
                flush=True,
            )
            moment, segment_error = integrate(
                AnalyticSource(evaluate, tuple(caps), ref),
                (kt, kx),
                ((start, end), (0.0, 1.0)),
                "gaussian",
                GaussianOptions(order, tuple(counts), (batch, 8192)),
            )
            coefficients += moment
            error += segment_error
            segments.append(
                dict(
                    interval=[start, end],
                    counts=counts,
                    caps=torch.stack(caps).cpu(),
                    covers=covers,
                    error=segment_error.cpu(),
                    radii=radii,
                    node_count=math.prod(counts) * order**2,
                )
            )
        record = dict(
            layer=layer,
            time_batch=batch,
            order=order,
            segments=segments,
            coefficients=coefficients.cpu(),
            error=error.cpu(),
            seconds=time.perf_counter() - started,
        )
        self.records.append(record)
        self.progress(
            f"block {layer} complete: {record['seconds']:.2f}s, moment radius {float(error.max()):.3g}",
            flush=True,
        )
        return coefficients, error
