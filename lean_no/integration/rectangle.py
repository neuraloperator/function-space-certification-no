from lean_no.integration.spatial import spatial

def rectangle(source, basis=None, domain=None, options=None):
    from lean_no.integration.cell_kernel import CellKernel, cell_kernel_integral

    if isinstance(source, CellKernel):
        return cell_kernel_integral(source, "rectangle", options)
    from lean_no.integration.kernel_norm import LinearizedKernel, kernel_norm

    if isinstance(source, LinearizedKernel):
        return kernel_norm(source, "rectangle", options)
    if hasattr(source, "sizes"):
        from lean_no.integration.sphere import spherical_integral

        return spherical_integral(source, basis, domain, "rectangle", options)
    from lean_no.integration.gaussian import AnalyticSource, moment_integral

    if isinstance(source, AnalyticSource):
        return moment_integral(source, basis, domain, options, method="rectangle")
    return spatial(source, basis, domain, "rectangle", options)
