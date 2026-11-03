from dataclasses import replace

import torch

from lean_no.graph import Operator

def compose_affine(graph):
    ops, cache = [], {}
    reference = graph.reference

    def append(kind, parents, data):
        ops.append(Operator(f"fused.{len(ops)}", kind, tuple(parents), data))
        return len(ops)

    def visit(index, weight=None, bias=None):
        if index == 0:
            if weight is None:
                return 0
            return append("affine", (0,), dict(weight=weight, bias=bias))
        op = graph.ops[index - 1]
        if op.kind == "affine":
            matrix = op.weight if weight is None else weight @ op.weight
            offset = op.bias if weight is None else weight @ op.bias + bias
            return visit(op.parents[0], matrix, offset)
        if op.kind == "add" and weight is not None:
            parents = [
                visit(parent, weight, bias if j == 0 else torch.zeros_like(bias))
                for j, parent in enumerate(op.parents)
            ]
            return append("add", parents, {})
        if weight is not None and op.kind not in ("spectral", "affine", "add"):
            return append("affine", (visit(index),), dict(weight=weight, bias=bias))
        if weight is None and index in cache:
            return cache[index]
        data = dict(op.data)
        if op.kind == "spectral":
            parent = op.parents[0]
            matrix = torch.eye(
                op.in_channels, device=reference.device, dtype=reference.dtype
            )
            offset = reference.new_zeros(op.in_channels)
            while parent and graph.ops[parent - 1].kind == "affine":
                affine = graph.ops[parent - 1]
                offset = matrix @ affine.bias + offset
                matrix = matrix @ affine.weight
                parent = affine.parents[0]
            kernel = torch.einsum("oi...,ij->oj...", op.weight, matrix.to(op.weight))
            zero = tuple(int((axis == 0).nonzero()[0]) for axis in data["modes"])
            value = op.bias + op.weight[(slice(None), slice(None), *zero)].real @ offset
            if weight is not None:
                kernel = torch.einsum("po,oi...->pi...", weight.to(kernel), kernel)
                value = weight @ value + bias
            result = append(
                "spectral", (visit(parent),), {**data, "weight": kernel, "bias": value}
            )
        else:
            parents = tuple(visit(parent) for parent in op.parents)
            if op.kind == "activation" and len(parents) == 1 and parents[0]:
                previous = ops[parents[0] - 1]
                if (
                    previous.kind == "activation"
                    and previous.data["activation"] == data["activation"]
                ):
                    if data["activation"] == "identity":
                        cache[index] = parents[0]
                        return parents[0]
                    data["repetitions"] = previous.data.get("repetitions", 1) + 1
                    parents = previous.parents
            result = append(op.kind, parents, data)
            if weight is not None:
                return append("affine", (result,), dict(weight=weight, bias=bias))
        if weight is None:
            cache[index] = result
        return result

    target = visit(len(graph.ops))
    used = set()

    def collect(index):
        if not index or index in used:
            return
        used.add(index)
        for parent in ops[index - 1].parents:
            collect(parent)

    collect(target)
    indices = {old: new for new, old in enumerate(sorted(used), 1)}
    return replace(
        graph,
        ops=tuple(
            replace(
                ops[i - 1], parents=tuple(indices.get(p, 0) for p in ops[i - 1].parents)
            )
            for i in sorted(used)
        ),
    )
