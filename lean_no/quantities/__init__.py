from dataclasses import dataclass

@dataclass(frozen=True)
class Residual:
    pde: str
    coefficient: float | None = None

    def required_derivatives(self, grid=None):
        if self.pde == "ks":
            return ((0, 0), (1, 0), (0, 1), (0, 2), (0, 3), (0, 4))
        if self.pde == "burgers":
            return ((0, 0), (1, 0), (0, 1), (2, 0))
        if self.pde in ("poisson", "helmholtz"):
            dimension = len(grid.resolution) if grid is not None else 1
            if self.pde == "helmholtz" and dimension != 1:
                raise ValueError("Helmholtz residual requires one spatial dimension")
            return (tuple(0 for _ in range(dimension)),) + tuple(
                tuple(order if i == axis else 0 for i in range(dimension))
                for order in (1, 2)
                for axis in range(dimension)
            )
        raise ValueError(f"unsupported PDE: {self.pde}")

    def evaluate(self, parts, forcing=None):
        dimension = len(next(iter(parts)))
        value = parts[(0,) * dimension]
        if self.pde in ("poisson", "helmholtz"):
            second = sum(
                parts[tuple(2 if i == axis else 0 for i in range(dimension))]
                for axis in range(dimension)
            )
            if self.pde == "poisson":
                return second * -1 + forcing * -1
            if self.coefficient is None:
                raise ValueError("Helmholtz residual requires kappa")
            return second * -1 + value * (-(self.coefficient**2)) + forcing * -1
        coefficient = 0.01 if self.coefficient is None else self.coefficient
        if self.pde == "burgers":
            return (
                parts[(0, 1)] + value * parts[(1, 0)] + parts[(2, 0)] * (-coefficient)
            )
        if self.pde == "ks":
            return (
                parts[(1, 0)]
                + value * parts[(0, 1)]
                + parts[(0, 2)]
                + coefficient * parts[(0, 4)]
            )
        raise ValueError(f"unsupported PDE: {self.pde}")
