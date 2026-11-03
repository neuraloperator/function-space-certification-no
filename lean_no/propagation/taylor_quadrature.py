import math
from dataclasses import replace

import torch

from lean_no.adapters.fno import build_graph
from lean_no.bounds.operators import is_identity, is_tanh
from lean_no.integration import integrate
from lean_no.propagation import linear as ibp

def pointwise(operation, bounds, chunk=1048576):
    count = bounds.resolution[-1]
    chunk = max(1, chunk // math.prod(bounds.resolution[:-1]))
    if count <= chunk:
        return operation(bounds)
    names = (
        "node_lower",
        "node_upper",
        "partial_lower",
        "partial_upper",
        "partial_node_lower",
        "partial_node_upper",
        "second_lower",
        "second_upper",
    )
    output = {}
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        values = {
            name: (
                None
                if getattr(bounds, name) is None
                else getattr(bounds, name)[..., start:stop]
            )
            for name in names
        }
        # Pointwise rules use only cell spacing, never global integrals or queries.
        part = replace(
            bounds,
            axes=(*bounds.axes[:-1], bounds.axes[-1][start:stop]),
            lengths=(*bounds.lengths[:-1], bounds.steps[-1] * (stop - start)),
            **values,
        )
        part = operation(part)
        for name in names:
            value = getattr(part, name)
            if name not in output:
                output[name] = (
                    None
                    if value is None
                    else value.new_empty((*value.shape[:-1], count))
                )
            if value is not None:
                output[name][..., start:stop] = value
        del part
    return replace(bounds, **output)

@torch.no_grad()
def graph(model, function, resolution, lengths, dtype=None):
    ibp.validate(model, resolution, lengths)
    function.validate()
    if not function.is_exact:
        raise ValueError("Taylor quadrature requires a fixed exact input")
    if function.curvature is None:
        raise ValueError("Taylor quadrature requires input curvature bounds")
    network = build_graph(model, dtype)
    activations = (
        network.lifting.activation,
        network.projection.activation,
        *(block.activation for block in network.blocks if block.activation is not None),
    )
    if any(not is_tanh(item) and not is_identity(item) for item in activations):
        raise NotImplementedError(
            "Taylor quadrature requires smooth tanh or identity activations"
        )
    field = ibp.sample_input(network, resolution, lengths, function, second_order=True)
    return propagate(network, field)

@torch.no_grad()
def propagate(network, field):
    from lean_no.graph import execute

    def apply(op, parents):
        source = parents[0]
        if op.kind == "affine":
            return pointwise(lambda part: ibp.affine(op, part), source)
        if op.kind == "spectral":
            return integrate(source, op, source.lengths, "taylor_quadrature")
        if op.kind == "add":
            return ibp.add(*parents)
        return pointwise(
            lambda part: ibp.activation(op.data["activation"], part), source
        )

    output = execute(network, field, apply)
    return replace(output, stats={"implementation": "taylor_ibp_v2"})
