from dataclasses import dataclass
from importlib import import_module

backends = ("rectangle", "taylor_quadrature", "gaussian", "clenshaw_curtis", "arb")

@dataclass(frozen=True)
class AdjointKernel:
    module: object
    cache: dict | None = None

@dataclass(frozen=True)
class Integration:
    method: str = "gaussian"
    options: object = None

    def __post_init__(self):
        if isinstance(self.options, dict):
            object.__setattr__(self, "options", dict(self.options))

    def __call__(self, source, basis=None, domain=None, options=None):
        if self.options is not None:
            options = (
                {**self.options, **(options or {})}
                if isinstance(options, (dict, type(None)))
                else options
            )
        return integrate(source, basis, domain, self.method, options)

def integrate(source, basis=None, domain=None, method="gaussian", options=None):
    if method not in backends:
        raise ValueError(f"unknown integration method: {method}")
    backend = getattr(import_module(f"{__name__}.{method}"), method)
    return backend(source, basis, domain, options)
