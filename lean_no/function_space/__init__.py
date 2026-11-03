from importlib import import_module

exports = {
    "DiscreteGrid": ("lean_no.domain", "DiscreteGrid"),
    "GridBounds": ("lean_no.enclosures.grid", "GridBounds"),
    "Certificate": ("lean_no.certificate", "Certificate"),
    "FourierGrid": ("lean_no.domain", "FourierGrid"),
    "certify": ("lean_no.certify", "certify"),
    "quadrature_crown": ("lean_no.function_space.core", "quadrature_crown"),
    "quadrature_ibp": ("lean_no.function_space.core", "quadrature_ibp"),
    "perturbation_bounds": ("lean_no.propagation.perturbation", "perturbation_bounds"),
    "residual_bound": ("lean_no.quantities.residual", "residual_bound"),
    "spherical_bounds": (
        "lean_no.propagation.spherical",
        "spherical_bounds",
    ),
    "grid_crown": ("lean_no.function_space.grid.crown", "grid_crown"),
    "grid_ibp_conv": ("lean_no.function_space.grid.ibp", "grid_ibp_conv"),
    "grid_ibp": ("lean_no.function_space.grid.ibp", "grid_ibp"),
    "Expr": ("lean_no.function_space.input", "Expr"),
    "FourierInput": ("lean_no.function_space.input", "FourierInput"),
    "InputBounds": ("lean_no.function_space.input", "InputBounds"),
    "acos": ("lean_no.function_space.input", "acos"),
    "acosh": ("lean_no.function_space.input", "acosh"),
    "asin": ("lean_no.function_space.input", "asin"),
    "asinh": ("lean_no.function_space.input", "asinh"),
    "atan": ("lean_no.function_space.input", "atan"),
    "atanh": ("lean_no.function_space.input", "atanh"),
    "coord": ("lean_no.function_space.input", "coord"),
    "cos": ("lean_no.function_space.input", "cos"),
    "cos_pi": ("lean_no.function_space.input", "cos_pi"),
    "cosh": ("lean_no.function_space.input", "cosh"),
    "exp": ("lean_no.function_space.input", "exp"),
    "log": ("lean_no.function_space.input", "log"),
    "max_expr": ("lean_no.function_space.input", "max_expr"),
    "min_expr": ("lean_no.function_space.input", "min_expr"),
    "pi": ("lean_no.function_space.input", "pi"),
    "sin": ("lean_no.function_space.input", "sin"),
    "sin_pi": ("lean_no.function_space.input", "sin_pi"),
    "sinh": ("lean_no.function_space.input", "sinh"),
    "sqrt": ("lean_no.function_space.input", "sqrt"),
    "tan": ("lean_no.function_space.input", "tan"),
    "tanh": ("lean_no.function_space.input", "tanh"),
    "to_expr": ("lean_no.function_space.input", "to_expr"),
    "ResidualBounds": ("lean_no.quantities.norms", "ResidualBounds"),
    "sfno_grid": ("lean_no.spectral.spherical", "sfno_grid"),
    "SFNOInput": ("lean_no.adapters.sfno", "SFNOInput"),
    "SFNOModel": ("lean_no.adapters.sfno", "SFNOModel"),
    "adapt_backbone": ("lean_no.adapters.sfno", "adapt_backbone"),
    "SphereGrid": ("lean_no.function_space.spherical", "SphereGrid"),
    "sphere_grid": ("lean_no.function_space.spherical", "sphere_grid"),
}
__all__ = list(exports)

def __getattr__(name):
    if name not in exports:
        raise AttributeError(name)
    module, attribute = exports[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
