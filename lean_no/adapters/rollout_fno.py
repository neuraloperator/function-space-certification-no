import torch

from lean_no.graph import Graph, Operator

def build_graph(state, lengths, split, native_shape, orders=((0, 0),)):
    if len(lengths) != 2 or any(length <= 0 for length in lengths):
        raise ValueError("two positive physical periods required")
    if not 0 < split <= 1 or len(native_shape) != 2 or any(n < 1 for n in native_shape):
        raise ValueError("valid split and two positive native grid sizes required")
    ops = []
    layers = state["fno_blocks.convs.bias"].shape[0]

    def append(name, kind, parent, **data):
        parents = parent if isinstance(parent, tuple) else (parent,)
        data = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }
        ops.append(Operator(name, kind, parents, data))
        return len(ops)

    def linear(source, prefix):
        weight = state[prefix + ".weight"].flatten(1)
        bias = state.get(prefix + ".bias", weight.new_zeros(len(weight)))
        return append(prefix, "affine", source, weight=weight, bias=bias.flatten())

    def mlp(source, prefix):
        source = linear(source, prefix + ".fcs.0")
        source = append(prefix + ".gelu", "activation", source, activation="gelu")
        return linear(source, prefix + ".fcs.1")

    def norm(source, index):
        return append(
            f"norm.{index}",
            "normalize",
            source,
            weight=state[f"fno_blocks.norm.{index}.weight"],
            bias=state[f"fno_blocks.norm.{index}.bias"],
            epsilon=1e-5,
            native_shape=native_shape,
            groups=1,
            averaging="native_arithmetic",
            index=index,
        )

    source = mlp(0, "lifting")
    source = append("padding", "pad", source, split=split)
    for layer in range(layers):
        previous = source
        spectral = append(
            f"blocks.{layer}.spectral",
            "spectral",
            previous,
            layer=layer,
            weight=torch.view_as_complex(
                state[f"fno_blocks.convs.weight.{layer}.tensor"].contiguous()
            )
            .transpose(0, 1)
            .contiguous(),
            bias=state["fno_blocks.convs.bias"][layer, :, 0, 0],
        )
        spectral = norm(spectral, 2 * layer)
        skip = linear(previous, f"fno_blocks.fno_skips.{layer}")
        hidden = append(f"blocks.{layer}.add", "add", (spectral, skip))
        hidden = append(f"blocks.{layer}.gelu", "activation", hidden, activation="gelu")
        gate = state[f"fno_blocks.mlp_skips.{layer}.weight"].flatten()
        skip = append(
            f"blocks.{layer}.gate",
            "affine",
            previous,
            weight=torch.diag(gate),
            bias=torch.zeros_like(gate),
        )
        hidden = mlp(hidden, f"fno_blocks.mlp.{layer}")
        hidden = append(f"blocks.{layer}.mlp_add", "add", (hidden, skip))
        source = norm(hidden, 2 * layer + 1)
        if layer < layers - 1:
            source = append(
                f"blocks.{layer}.output_gelu", "activation", source, activation="gelu"
            )
    source = mlp(source, "projection")
    if bool(state.get("_hard_initial_condition", False)):
        ref = state["fno_blocks.convs.bias"]
        initial = append(
            "initial_trace",
            "affine",
            0,
            weight=ref.new_tensor([[1.0, 0.0, 0.0]]),
            bias=ref.new_zeros(1),
        )
        append("hard_initial_blend", "initial_blend", (initial, source))
    from dataclasses import replace

    for index, op in enumerate(ops):
        if op.kind == "spectral":
            weight = op.weight
            modes = (
                torch.arange(
                    -(weight.shape[-2] // 2),
                    weight.shape[-2] - weight.shape[-2] // 2,
                    device=weight.device,
                    dtype=weight.real.dtype,
                ),
                torch.arange(
                    weight.shape[-1], device=weight.device, dtype=weight.real.dtype
                ),
            )
            ops[index] = replace(
                op, data={**op.data, "modes": modes, "basis": "fourier"}
            )
    return Graph(
        tuple(ops),
        {
            "basis": "fourier",
            "axes": ("t", "x"),
            "periodic": (False, True),
            "native_shape": native_shape,
            "lengths": tuple(lengths),
            "split": split,
            "orders": tuple(orders),
            "modes": tuple(
                op.data["weight"].shape[-2:] for op in ops if op.kind == "spectral"
            ),
        },
    )
