import torch
from torch import nn

from lean_no.bounds.activations import is_gelu, is_identity, is_relu, is_tanh
from lean_no.bounds.normalization import normalization_parameters
from lean_no.bounds.operators import affine_bias, affine_weight, validate_fno
from lean_no.graph import Graph, Operator
from lean_no.kernel import mode_weights

def model_real_dtype(model):
    return model_reference(model).dtype

def model_reference(model):
    return (
        model.reference if isinstance(model, Graph) else next(model.parameters()).real
    )

def cast_module(module, dtype):
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return module._apply(
        lambda value: (
            value.to(complex_dtype if value.is_complex() else dtype)
            if value.is_floating_point() or value.is_complex()
            else value
        )
    )

def activation_name(function):
    for name, predicate in (
        ("identity", is_identity),
        ("relu", is_relu),
        ("tanh", is_tanh),
        ("gelu", is_gelu),
    ):
        if predicate(function):
            if name == "gelu" and getattr(function, "approximate", "none") != "none":
                raise NotImplementedError("only exact GELU is supported")
            return name
    raise NotImplementedError(f"unsupported activation: {function}")

def build_graph(model, dtype=None, normalization=False):
    if isinstance(model, Graph):
        model.validate()
        if dtype is None or dtype == model.dtype:
            return model
        from dataclasses import replace

        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        return replace(
            model,
            ops=tuple(
                replace(
                    op,
                    data={
                        key: value.to(complex_dtype if value.is_complex() else dtype)
                        if isinstance(value, torch.Tensor)
                        else tuple(k.to(dtype) for k in value)
                        if key == "modes"
                        else value
                        for key, value in op.data.items()
                    },
                )
                for op in model.ops
            ),
        )
    validate_fno(model, normalization=normalization)
    if model.positional_embedding is not None:
        from neuralop.layers.embeddings import GridEmbeddingND

        embedding = model.positional_embedding
        if not isinstance(embedding, GridEmbeddingND) or any(
            tuple(bounds) != (0.0, 1.0) for bounds in embedding.grid_boundaries
        ):
            raise NotImplementedError(
                "only normalized grid coordinate embeddings are supported"
            )
    dtype = model_real_dtype(model) if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    ops, groups = [], []
    reference = next(model.parameters()).real.to(dtype=dtype)

    def append(name, kind, parents, **data):
        parents = parents if isinstance(parents, tuple) else (parents,)
        ops.append(Operator(name, kind, parents, data))
        return len(ops)

    def affine(source, module, name, channels):
        weight = (
            torch.eye(channels, device=reference.device, dtype=dtype)
            if isinstance(module, nn.Identity)
            else affine_weight(module).to(reference)
        )
        bias = affine_bias(module, reference)
        return append(
            name,
            "affine",
            source,
            weight=weight.detach().clone(),
            bias=reference.new_zeros(len(weight)) if bias is None else bias.clone(),
        )

    def activate(source, function, name):
        return append(name, "activation", source, activation=activation_name(function))

    def mlp(source, module, name):
        start = len(ops)
        for index, layer in enumerate(module.fcs):
            source = affine(source, layer, f"{name}.{index}", 0)
            if index + 1 < len(module.fcs):
                source = activate(
                    source, module.non_linearity, f"{name}.{index}.activation"
                )
        return source, tuple(range(start + 1, len(ops) + 1))

    def normalize(source, module, name, channels):
        weight, bias, epsilon, count = normalization_parameters(
            module, channels, reference
        )
        return append(
            name,
            "normalize",
            source,
            weight=weight.detach().clone(),
            bias=bias.detach().clone(),
            epsilon=epsilon,
            groups=count,
        )

    source, lifting = mlp(0, model.lifting, "lifting")
    blocks = model.fno_blocks
    for index, conv in enumerate(blocks.convs):
        previous = source
        weight, modes = mode_weights(conv, dtype)
        bias = affine_bias(conv, reference)
        source = append(
            f"blocks.{index}.spectral",
            "spectral",
            previous,
            weight=weight.clone(),
            modes=tuple(k.clone() for k in modes),
            bias=reference.new_zeros(len(weight)) if bias is None else bias.clone(),
            basis="fourier",
        )
        group = dict(spectral=source, activations=[])
        if blocks.norm is not None:
            source = normalize(
                source,
                blocks.norm[blocks.n_norms * index],
                f"blocks.{index}.norm0",
                conv.out_channels,
            )
            group["norm0"] = source
        if blocks.fno_skips is not None:
            skip = affine(
                previous,
                blocks.fno_skips[index],
                f"blocks.{index}.skip",
                conv.in_channels,
            )
            group["skip"] = skip
            source = append(f"blocks.{index}.add", "add", (source, skip))
        if index + 1 < model.n_layers:
            source = activate(
                source, blocks.non_linearity, f"blocks.{index}.activation.0"
            )
            group["activations"].append(source)
            if not blocks.use_channel_mlp and blocks.norm is None:
                source = activate(
                    source, blocks.non_linearity, f"blocks.{index}.activation.1"
                )
                group["activations"].append(source)
        if blocks.use_channel_mlp:
            source, nodes = mlp(
                source, blocks.channel_mlp[index], f"blocks.{index}.mlp"
            )
            group["channel_mlp"] = nodes
            if blocks.channel_mlp_skips is not None:
                skip = affine(
                    previous,
                    blocks.channel_mlp_skips[index],
                    f"blocks.{index}.mlp_skip",
                    conv.in_channels,
                )
                source = append(f"blocks.{index}.mlp_add", "add", (source, skip))
        if blocks.norm is not None:
            source = normalize(
                source,
                blocks.norm[blocks.n_norms * index + 1],
                f"blocks.{index}.norm1",
                conv.out_channels,
            )
            group["norm1"] = source
        if (
            blocks.use_channel_mlp or blocks.norm is not None
        ) and index + 1 < model.n_layers:
            source = activate(
                source, blocks.non_linearity, f"blocks.{index}.output_activation"
            )
            group["activations"].append(source)
        groups.append(group)
    source, projection = mlp(source, model.projection, "projection")
    graph = Graph(
        tuple(ops),
        dict(
            basis="fourier",
            dimension=model.n_dim,
            coordinate_channels=model.positional_embedding is not None,
            input_channels=model.in_channels,
            output_channels=model.out_channels,
            lifting=lifting,
            projection=projection,
            blocks=tuple(groups),
        ),
    )
    graph.validate()
    return graph

@torch.no_grad()
def native(graph, values):
    from lean_no.bounds.operators import convolution
    from lean_no.graph import activation_value, execute
    from lean_no.kernel import grid_kernel

    values = graph.append_coordinates(values)

    def apply(op, parents):
        source = parents[0]
        if op.kind == "affine":
            shape = (1, -1, *([1] * (source.ndim - 2)))
            return torch.einsum("oi,bi...->bo...", op.weight, source) + op.bias.reshape(
                shape
            )
        if op.kind == "spectral":
            shape = (1, -1, *([1] * (source.ndim - 2)))
            return convolution(
                grid_kernel(op, tuple(source.shape[2:])), source
            ) + op.bias.reshape(shape)
        if op.kind == "activation":
            return activation_value(op.data["activation"], source)
        if op.kind == "normalize":
            return torch.nn.functional.group_norm(
                source, op.data["groups"], op.weight, op.bias, op.data["epsilon"]
            )
        if op.kind == "add":
            return sum(parents)
        raise NotImplementedError(op.kind)

    return execute(graph, values, apply)
