import torch

from lean_no.propagation.rollout import initial_state_handoff, rollout
from lean_no.quantities import Residual
from lean_no.quantities.ks import geometry
from lean_no.quantities.residual import residual_bound
from lean_no.spectral.strip import StripGauss

ref = torch.zeros(1, dtype=torch.float64)
state = {"fno_blocks.convs.bias": ref.new_zeros(1, 1, 1, 1)}
for prefix, weight in (
    ("lifting.fcs.0", [[0.05, 0.1, 0.0]]),
    ("lifting.fcs.1", [[0.5]]),
    ("projection.fcs.0", [[0.2]]),
    ("projection.fcs.1", [[0.2]]),
    ("fno_blocks.fno_skips.0", [[1.0]]),
    ("fno_blocks.mlp.0.fcs.0", [[0.0]]),
    ("fno_blocks.mlp.0.fcs.1", [[0.0]]),
):
    state[prefix + ".weight"] = ref.new_tensor(weight)
    state[prefix + ".bias"] = ref.new_zeros(1)
state["fno_blocks.convs.weight.0.tensor"] = ref.new_zeros(1, 1, 16, 33, 2)
state["fno_blocks.mlp_skips.0.weight"] = ref.new_ones(1)
for index in (0, 1):
    state[f"fno_blocks.norm.{index}.weight"] = ref.new_full((1,), 0.001)
    state[f"fno_blocks.norm.{index}.bias"] = ref.new_zeros(1)

def initial(x):
    return 0.2

profile = initial
for window in range(1, 3):
    integration = StripGauss(
        order=12, cells=((8, 16),), batch=8, progress=lambda *args, **kwargs: None
    )
    field, _, _ = rollout(
        state=state, profile=profile, integrate=integration, **geometry
    )
    certificate = residual_bound(
        field, Residual(pde="ks", coefficient=0.01), index=window
    )
    report = certificate.diagnostics
    print("Window", window, "residual Linf upper:", report["linf_upper"])
    profile = initial_state_handoff(output=field, retain_derivatives=True)
