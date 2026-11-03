import torch

def spherical_integral(source, basis, domain, method, options=None):
    from lean_no.function_space.spherical import SphereGrid, sphere_grid
    from lean_no.integration.clenshaw_curtis import (
        integration_error,
        spherical_moments,
        strip_integration_error,
    )
    from lean_no.spectral.spherical import sfno_grid

    if not isinstance(domain, SphereGrid):
        raise ValueError("spherical integration requires a spherical grid")
    if method in ("gaussian", "arb"):
        grid = sphere_grid(
            domain.shape,
            kind="legendre-gauss",
            dtype=domain.latitude.dtype,
            device=domain.latitude.device,
        )
    elif method == "rectangle":
        n = len(domain.latitude)
        mu = (
            1
            - 2
            * torch.arange(
                n, device=domain.latitude.device, dtype=domain.latitude.dtype
            )
            / n
        )
        grid = SphereGrid(
            mu.acos(), torch.full_like(mu, 2 / n), domain.nlon, "rectangle"
        )
        grid.validate()
    elif method == "clenshaw_curtis":
        grid = (
            domain
            if domain.kind == "clenshaw-curtis"
            else sfno_grid(
                len(domain.latitude),
                domain.latitude.device,
                domain.latitude.dtype,
                rule="clenshaw-curtis",
                nlon=domain.nlon,
            )
        )
    else:
        raise ValueError("unknown spherical rule")
    degrees, orders = basis
    cache = options if isinstance(options, dict) else None
    moments, exact = spherical_moments(
        source, grid, degrees, orders, moment_cache=cache
    )
    remainder = None
    if not exact:
        remainder = integration_error(source.sizes, grid, degrees)
        if method == "clenshaw_curtis":
            strip = strip_integration_error(source, grid, degrees)
            if strip is not None:
                remainder = torch.minimum(remainder, strip)
    return moments, exact, remainder
