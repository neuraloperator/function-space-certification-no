import math

import torch

from lean_no.adapters.fno import build_graph, native
from lean_no.domain import FourierGrid
from lean_no.evaluation import witness
from lean_no.methods import Method, linearized_method
from lean_no.runtime import measure
from scripts._shared.inputs import basis_values, center_values, witness_input

def sample_ball(count, dimension, reference, generator=None):
    magnitudes = torch.empty(
        count, dimension, device=reference.device, dtype=torch.float64
    )
    magnitudes.exponential_(generator=generator)
    magnitudes /= magnitudes.sum(dim=1, keepdim=True)
    signs = (
        torch.randint(
            0, 2, magnitudes.shape, device=reference.device, generator=generator
        )
        .mul(2)
        .sub(1)
    )
    radius = torch.rand(
        count, 1, device=reference.device, dtype=torch.float64, generator=generator
    )
    return radius.pow(1 / dimension) * signs * magnitudes

def sparse_samples(dimension, reference):
    eye = torch.eye(dimension, device=reference.device, dtype=torch.float64)
    return torch.cat((eye, -eye))

def native_inputs(case, model):
    reference = next(model.parameters())
    axis = (
        torch.arange(case.grid_resolution, device=reference.device, dtype=torch.float64)
        * case.length
        / case.grid_resolution
    )
    coordinates = (
        (axis,)
        if case.spatial_dimension == 1
        else torch.meshgrid(axis, axis, indexing="ij")
    )
    grid = coordinates[0] if case.spatial_dimension == 1 else coordinates
    center, basis = center_values(case, grid), basis_values(case, grid)
    if case.time_samples:
        times = (
            torch.arange(
                case.time_samples, device=reference.device, dtype=torch.float64
            )
            * case.time_period
            / case.time_samples
        )
        center = torch.stack(
            (
                center[:, None].expand(-1, case.time_samples),
                times[None].expand(case.grid_resolution, -1),
            )
        )
        zero = torch.zeros(
            len(basis),
            case.grid_resolution,
            case.time_samples,
            device=reference.device,
            dtype=torch.float64,
        )
        basis = torch.stack((basis[:, :, None].expand_as(zero), zero), dim=1)
        points = torch.cartesian_prod(axis, times)
    else:
        center, basis = center[None], basis[:, None]
        points = (
            axis[:, None]
            if case.spatial_dimension == 1
            else torch.stack(coordinates, -1).reshape(-1, 2)
        )
    return center, basis, points

@torch.no_grad()
def run_mc(case, model):
    if next(model.parameters()).dtype != torch.float64:
        raise ValueError("robustness search requires float64 model parameters")
    center, basis, points = native_inputs(case, model)
    graph = build_graph(model, torch.float64)
    generator = torch.Generator(device=center.device).manual_seed(case.mc_seed)
    vertices = sparse_samples(len(basis), center)
    coefficients = torch.cat(
        (
            vertices,
            sample_ball(case.mc_samples - len(vertices), len(basis), center, generator),
        )
    )
    lower = center.new_full((len(points),), math.inf)
    upper = -lower
    lower_ids = torch.zeros(len(points), device=center.device, dtype=torch.long)
    upper_ids = lower_ids.clone()

    def search():
        for start in range(0, len(coefficients), case.mc_batch_size):
            part = coefficients[start : start + case.mc_batch_size]
            inputs = center[None] + (part @ basis.flatten(1)).reshape(
                len(part), *center.shape
            )
            outputs = native(graph, inputs)[:, 0].flatten(1)
            # Native-grid aliases can differ by roundoff. Prefer the earliest
            # signed Fourier vertex in a numerical tie, rather than replacing
            # a low-frequency witness with an aliased high-frequency one.
            tolerance = 1e-12 * (1 + outputs.abs().amax(0))
            low, high = outputs.amin(0), outputs.amax(0)
            ids_low = (outputs <= low + tolerance).to(torch.int64).argmax(0)
            ids_high = (outputs >= high - tolerance).to(torch.int64).argmax(0)
            columns = torch.arange(outputs.shape[1], device=outputs.device)
            low, high = outputs[ids_low, columns], outputs[ids_high, columns]
            mask = low < lower - tolerance
            lower[mask], lower_ids[mask] = low[mask], start + ids_low[mask]
            mask = high > upper + tolerance
            upper[mask], upper_ids[mask] = high[mask], start + ids_high[mask]

    _, seconds, peak = measure(search, center.device)
    index = int((upper - lower).argmax())
    return dict(
        seconds=seconds,
        peak_gib=peak / 2**30,
        lower=lower.cpu(),
        upper=upper.cpu(),
        global_width=float((upper - lower)[index]),
        global_index=index,
        global_coef=coefficients[
            torch.stack((upper_ids[index], lower_ids[index]))
        ].cpu(),
        point=points[index].cpu(),
        tie_break="earliest candidate within 1e-12*(1+output magnitude); signed vertices first",
        samples=case.mc_samples,
        basis_size=len(basis),
        seed=case.mc_seed,
        grid_resolution=list(center.shape[1:]),
        dtype="float64",
    )

@torch.no_grad()
def evaluate_witness(case, model, mc):
    if case.spatial_dimension == 1 and not case.time_samples:
        grid = FourierGrid((8 * case.full_resolution,), (case.length,), torch.float64)
        method = Method("crown", "gaussian")
    else:
        shape = (case.full_resolution,) * 2
        lengths = (case.length, case.time_period if case.time_samples else case.length)
        grid = FourierGrid(shape, lengths, torch.float64)
        method = linearized_method
    device = next(model.parameters()).device
    points = mc["point"].to(device=device, dtype=torch.float64)[None]
    if case.time_samples:
        x = torch.arange(64, device=device, dtype=torch.float64) * case.length / 64
        t = torch.linspace(0, 1, 65, device=device, dtype=torch.float64)
        points = torch.cartesian_prod(x, t)
        if not (points == mc["point"].to(points)).all(-1).any():
            raise ValueError("continuum query grid must include the native winner")
    evidence, seconds, peak = measure(
        lambda: witness(
            model,
            grid,
            tuple(witness_input(case, c) for c in mc["global_coef"]),
            points,
            method,
            chunk_size=1,
        ),
        next(model.parameters()).device,
    )
    return dict(
        grid=dict(width=mc["global_width"], lower_bound=mc["global_width"]),
        continuum=dict(
            width=evidence["estimate"],
            lower_bound=evidence["lower_bound"],
            resolution=list(grid.resolution),
            location=evidence["point"],
            estimate_point=evidence["estimate_point"],
            query_points=len(points),
            query_grid=[64, 65] if case.time_samples else None,
            outputs=evidence["outputs"],
            point=evidence["point"],
            intervals=evidence["intervals"],
        ),
        search=dict(
            width=mc["global_width"],
            seconds=mc["seconds"],
            samples=mc["samples"],
            basis_size=mc["basis_size"],
            seed=mc["seed"],
            coefficients=mc["global_coef"].tolist(),
            point=mc["point"].tolist(),
            grid_resolution=mc["grid_resolution"],
            dtype="float64",
        ),
        seconds=dict(grid=mc["seconds"], continuum=seconds),
        continuum_peak_gib=peak / 2**30,
    )
