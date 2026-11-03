from dataclasses import dataclass

rules = ("rectangle", "gaussian", "clenshaw_curtis", "arb")

@dataclass(frozen=True)
class Method:
    propagation: str
    integration: str | None
    strategy: str = "general"

    def validate(self):
        if self.propagation not in ("ibp", "crown"):
            raise ValueError("propagation must be ibp or crown")
        if self.strategy not in ("general", "taylor_cell", "linearized"):
            raise ValueError("strategy must be general, taylor_cell or linearized")
        if self.propagation == "crown" and self.strategy != "general":
            raise ValueError("forward strategies require propagation='ibp'")
        allowed = (*rules, None) if self.strategy != "taylor_cell" else rules
        if self.strategy == "general" and self.propagation == "ibp":
            allowed = (*allowed, "taylor_quadrature")
        if self.integration not in allowed:
            raise ValueError(
                f"unsupported integration for {self.propagation}/{self.strategy}: {self.integration}"
            )
        return self

    @property
    def label(self):
        if self.strategy == "linearized":
            return (
                "Linearized perturbation bounds"
                if self.integration is None
                else f"{self.integration.replace('_', ' ').title()} L1 linearized perturbation bounds"
            )
        backend = {
            None: "Grid",
            "arb": "Arb",
            "taylor_quadrature": "Taylor quadrature",
            "rectangle": "Rectangle",
            "gaussian": "Gaussian",
            "clenshaw_curtis": "Clenshaw-Curtis",
        }[self.integration]
        name = (
            "first-order Taylor cell bounds"
            if self.strategy == "taylor_cell"
            else self.propagation.upper()
        )
        return f"{backend} {name}"

linearized_method = Method("ibp", None, "linearized")
grid_ibp_method = Method("ibp", None)
grid_crown_method = Method("crown", None)
