import math

import torch

from lean_no.enclosures.spherical_taylor import spherical_taylor_cells
from lean_no.propagation.spherical import evaluate

@torch.no_grad()
def regional_maximum(field, box, cells=128, batch_size=32):
    south, north, west, east = box
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("expected a nonempty geographic rectangle without wrapping")
    if (
        type(cells) is not int
        or cells < 1
        or type(batch_size) is not int
        or batch_size < 1
        or len(field.error) != 1
    ):
        raise ValueError(
            "positive cell and batch counts and a scalar field are required"
        )
    radius, lipschitz = float(field.error[0]), float(field.sizes[1, 0])
    if not math.isfinite(radius + lipschitz):
        raise ValueError(
            "regional maximization needs finite radius and Lipschitz bound"
        )
    a, b, c, d = map(math.radians, (90 - north, 90 - south, west + 360, east + 360))
    ht, hx = (b - a) / (2 * cells), (d - c) / (2 * cells)
    corners = field.error.new_tensor([(a, c), (a, d), (b, c), (b, d)])
    samples = [corners.cpu()]
    values = [evaluate(field, corners[:, 0], corners[:, 1], False, {})[:, 0].cpu()]
    upper = -math.inf
    for start in range(0, cells**2, batch_size):
        ids = torch.arange(
            start, min(start + batch_size, cells**2), device=field.error.device
        )
        centers = torch.stack(
            (
                a + (ids // cells + 0.5) * (b - a) / cells,
                c + (ids % cells + 0.5) * (d - c) / cells,
            ),
            -1,
        ).to(field.error)
        half_widths = field.error.new_tensor([ht, hx]).expand(len(ids), -1)
        nominal, local = spherical_taylor_cells(field, centers, half_widths, degree=8)
        local = torch.where(torch.isfinite(local) & (local >= 0), local, math.inf)
        caps = nominal + radius + local.clamp_max(lipschitz * (ht + hx))
        upper = max(upper, float(caps.max()))
        samples.append(centers.cpu())
        values.append(nominal.cpu())
    points, nominal = torch.cat(samples).double(), torch.cat(values).double()
    if not torch.isfinite(nominal).all():
        raise FloatingPointError("nonfinite nominal temperature")
    index = int(nominal.argmax())
    lower = float(nominal[index]) - radius
    return (
        dict(
            maximum_lower=lower,
            maximum_upper=upper,
            gap=upper - lower,
            grid=[cells, cells],
            degree=8,
            complete_uniform_cover=True,
            evaluated_cells=cells**2,
            witness_latitude=90 - math.degrees(float(points[index, 0])),
            witness_longitude=math.degrees(float(points[index, 1])) - 360,
            pointwise_radius=radius,
            lipschitz_per_radian=lipschitz,
        ),
        dict(
            colatitude_longitude_radians=points,
            nominal=nominal,
            lower=nominal - radius,
            upper=nominal + radius,
        ),
    )
