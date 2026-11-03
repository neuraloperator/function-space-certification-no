import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

from lean_no.adapters.sfno import SFNOInput, adapt_backbone, native_parity
from lean_no.certify import certify
from lean_no.evaluation import sample_temperature
from lean_no.methods import Method
from lean_no.paths import repo_path
from lean_no.propagation.spherical import linear_field
from lean_no.quantities.spherical import regional_maximum
from lean_no.runtime import Timer, target_hashes, write_json
from lean_no.spectral.spherical import (
    basis_cache_capacity,
    harmonic_eval,
    sfno_grid,
)

summary_path = repo_path("results/sfno_temperature/summary.json")
data_dir = repo_path("data/era5_20230715T18")
initial_time = datetime(2023, 7, 15, 18, tzinfo=timezone.utc)
latitude_count, longitude_count = 262144, 196608
region_box = (32.5, 42, -124.5, -114)
target_identity = dict(
    parameters="dc523ef21c58e66ff005f892474bcdfaaad06d01dfcb6dada4087d2fc9ff034b",
    input="a14a499a707cbb86a70c03a17409d48007fa7008d4b444526cf40a0b92fc9977",
)
strip_radii = (
    (0.0005, 0.001, 0.004),
    (0.00075, 0.001, 0.002, 0.004),
)

def load_model():
    from earth2studio.models.px import SFNO

    forecast = (
        SFNO.load_model(SFNO.load_default_package(), device="cuda").to("cuda").eval()
    )
    backbone = next(
        m
        for m in forecast.modules()
        if all(hasattr(m, k) for k in ("encoder", "decoder", "blocks"))
    )
    model = adapt_backbone(backbone, tuple(forecast.model.params.channel_names))
    return forecast, backbone, model

def regional_samples():
    latitude = torch.linspace(90, -90, 721)
    longitude = torch.arange(1440) * 0.25
    rows = (latitude >= region_box[0]) & (latitude <= region_box[1])
    columns = (longitude >= region_box[2] + 360) & (longitude <= region_box[3] + 360)
    points = torch.stack(
        torch.meshgrid(
            (90 - latitude[rows]).deg2rad(), longitude[columns].deg2rad(), indexing="ij"
        ),
        -1,
    ).reshape(-1, 2)
    return rows, columns, points

def sampled_rmse(values, reference):
    return float((values.double() - reference.double()).square().mean().sqrt())

def download(channels):
    from earth2studio.data import ARCO_ERA5

    data_dir.mkdir(parents=True, exist_ok=True)
    metadata = data_dir / "metadata.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if datetime.fromisoformat(saved["initial_time_utc"]).replace(
            tzinfo=timezone.utc
        ) != initial_time or saved["channels"] != list(channels):
            raise ValueError("ERA5 cache timestamp/channel order differs")
    source = ARCO_ERA5(verbose=True)
    for name, timestamp, variables in (
        ("initial.npy", initial_time, list(channels)),
        ("verification_t2m.npy", initial_time + timedelta(hours=6), ["t2m"]),
    ):
        if not (data_dir / name).exists():
            data = source(timestamp.replace(tzinfo=None), variables)
            values = data.transpose("time", "variable", "lat", "lon").values.astype(
                np.float32
            )
            if (
                values.shape != (1, len(variables), 721, 1440)
                or not np.isfinite(values).all()
            ):
                raise ValueError("unexpected ERA5 snapshot")
            np.save(data_dir / name, values[0])
    metadata.write_text(
        json.dumps(
            dict(initial_time_utc=initial_time.isoformat(), channels=list(channels))
        )
    )

@torch.no_grad()
def prepare():
    from torch_harmonics import InverseRealSHT

    forecast, backbone, model = load_model()
    download(model.channels)
    raw = torch.from_numpy(np.load(data_dir / "initial.npy")).cuda()
    captured = []
    hook = backbone.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    try:
        forecast.model(raw[None], [initial_time], normalized_data=False)
    finally:
        hook.remove()
    values = captured.pop()[0]
    if values.shape != (77, 721, 1440):
        raise ValueError("unexpected native backbone input")
    mean, scale = forecast.model.in_bias.flatten(), forecast.model.in_scale.flatten()
    torch.testing.assert_close(
        values[:73], (raw - mean[:, None, None]) / scale[:, None, None]
    )
    coefficients = project_samples(values, 240)
    inputs = SFNOInput(
        coefficients, model.channels + ("xzen", "xoro", "xlsml", "xlsms")
    )
    if target_hashes(backbone, inputs) != target_identity:
        raise ValueError("fixed unsmoothed input or checkpoint changed")
    inverse = InverseRealSHT(
        721, 1440, lmax=240, mmax=240, grid="equiangular", norm="ortho", csphase=True
    ).to(device=raw.device, dtype=torch.float32)
    channel = model.channels.index("t2m")
    reconstructed = (
        inverse(coefficients[channel : channel + 1])[0] * scale[channel] + mean[channel]
    )
    rows, columns, _ = regional_samples()
    input_rmse = sampled_rmse(
        reconstructed.cpu()[rows][:, columns], raw[channel].cpu()[rows][:, columns]
    )
    native, _ = native_parity(
        backbone, model, prepared_values(inputs, model.input_grid, 48)
    )
    config = dict(
        channels=model.channels,
        mean=mean,
        scale=scale,
        target_hashes=target_identity,
        input_rmse_K=input_rmse,
    )
    native_temperature = (
        native[0, channel].cpu() * scale[channel].cpu() + mean[channel].cpu()
    )
    return model, config, inputs, native_temperature

@torch.no_grad()
def report(config, base, native_temperature):
    field = base.enclosure
    if field.error.dtype != torch.float32:
        raise ValueError("float32 forecast required; start a fresh reproduction")
    channel = config["channels"].index("t2m")
    weight = field.error.new_zeros((1, len(field.error)))
    weight[0, channel] = config["scale"][channel]
    temperature = linear_field(
        field, weight, field.error.new_tensor([config["mean"][channel] - 273.15])
    )
    with Timer(field.error.device, reset_peak=False) as query_timer:
        region, _ = regional_maximum(temperature, region_box)
    region.update(
        target_gap=1.0,
        target_met=region["gap"] <= 1.0,
        regional_query_seconds=query_timer.seconds,
    )
    if not region["target_met"]:
        raise ArithmeticError("regional maximum did not reach the 1 K target")
    rows, columns, points = regional_samples()
    points = points.to(field.error)
    values = sample_temperature(temperature, points=points)
    truth = (
        torch.from_numpy(np.load(data_dir / "verification_t2m.npy"))[0][rows][
            :, columns
        ].flatten()
        - 273.15
    )
    native = native_temperature[rows][:, columns].flatten() - 273.15
    summary = dict(
        region,
        pointwise_width_K=2 * float(temperature.error[0]),
        input_rmse_K=config["input_rmse_K"],
        forecast_rmse_K=sampled_rmse(values, truth),
        native_rmse_K=sampled_rmse(values, native),
        era5_sampled_maximum_C=float(truth.max()),
        forecast_sampled_maximum_C=float(values.max()),
        era5_minus_forecast_maximum_K=[
            float(truth.max()) - region["maximum_upper"],
            float(truth.max()) - region["maximum_lower"],
        ],
        box=region_box,
        maximum_degree=239,
        smoothing_scale=None,
        epsilon=0,
        arithmetic="float32/complex64",
        roundoff_enclosed=False,
        target_hashes=config["target_hashes"],
    )
    return summary

@torch.no_grad()
def prepared_values(inputs, grid, chunk_size):
    result = torch.empty(
        (1, len(inputs.channels), *grid.shape),
        device=grid.latitude.device,
        dtype=grid.latitude.dtype,
    )
    for start in range(0, grid.shape[0], chunk_size):
        latitude = grid.latitude[start : start + chunk_size]
        values = harmonic_eval(inputs.coefficients, latitude, grid.longitude, True)
        result[0, :, start : start + len(latitude)] = values.permute(2, 0, 1)
    return result

def project_samples(values, degrees):
    from torch_harmonics import RealSHT

    transform = RealSHT(
        *values.shape[-2:],
        lmax=degrees,
        mmax=degrees,
        grid="equiangular",
        norm="ortho",
        csphase=True,
    ).to(values)
    coefficients = transform(values)
    l = torch.arange(degrees, device=values.device)[:, None]
    m = torch.arange(degrees, device=values.device)[None, :]
    coefficients[..., m > l] = 0
    coefficients[..., 0] = coefficients[..., 0].real.clone()
    return coefficients

def run_experiment():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("SFNO: prepare", flush=True)
    model, config, inputs, native = prepare()
    torch.cuda.empty_cache()
    print("SFNO: certify", flush=True)
    grid = sfno_grid(
        latitude_count,
        "cuda",
        torch.float32,
        rule="clenshaw-curtis",
        nlon=longitude_count,
    )

    # Retain every latitude basis within the 128 GiB batch allocation.
    largest = max(
        op.data["degrees"] * op.data["orders"]
        for op in model.graph.ops
        if op.kind == "spectral"
    )
    entries = min(
        math.ceil(latitude_count / 32) + 512, (80 * 1024**3) // (4 * 32 * largest)
    )
    with basis_cache_capacity(entries):
        with Timer(grid.latitude.device, reset_peak=False) as timing:
            base = certify(
                model=model,
                grid=grid,
                inputs=inputs,
                chunk_size=32,
                strip_radii=strip_radii,
                strip_shape=(2048, 1024),
                progress=lambda row: print(
                    row["name"], float(row["radius"].max()), flush=True
                ),
                method=Method("ibp", "clenshaw_curtis"),
            )
    del inputs
    torch.cuda.empty_cache()
    print("SFNO: report", flush=True)
    summary = report(config, base, native)
    summary["base_seconds"] = timing.seconds
    return summary

def main():
    with Timer("cuda") as timing:
        summary = run_experiment()
    summary.update(
        full_run_seconds=timing.seconds,
        full_run_peak_cuda_allocated_bytes=timing.peak_bytes,
        device=torch.cuda.get_device_name(),
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary

if __name__ == "__main__":
    main()
