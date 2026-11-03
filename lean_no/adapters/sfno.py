import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional

from lean_no.function_space.spherical import sphere_grid
from lean_no.graph import Graph, Operator
from lean_no.spectral.spherical import harmonic_eval, harmonic_sizes

@dataclass(frozen=True)
class SFNOInput:

    coefficients: torch.Tensor | None
    channels: tuple[str, ...]
    epsilon: float = 0.0
    lipschitz: torch.Tensor | float | None = None
    reference: Callable | None = None
    sizes: torch.Tensor | None = None

    def __post_init__(self):
        value = self.coefficients
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("SFNO certification requires finite epsilon>=0")
        if value is None:
            sizes = self.sizes
            if (
                not callable(self.reference)
                or not isinstance(sizes, torch.Tensor)
                or sizes.shape != (2, len(self.channels))
                or sizes.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(sizes).all()
                or (sizes < 0).any()
                or not self.channels
            ):
                raise ValueError(
                    "callable inputs need certified supremum/Lipschitz sizes"
                )
        else:
            self.validate_harmonics()
            sizes = harmonic_sizes(value)
            if self.reference is not None or (
                self.sizes is not None and not torch.equal(self.sizes, sizes)
            ):
                raise ValueError("supply either harmonics or a callable reference")
        budget = self.lipschitz
        if budget is None:
            if self.epsilon:
                raise ValueError(
                    "nonzero epsilon requires a full-input Lipschitz budget"
                )
            budget = sizes[1]
        budget = torch.as_tensor(budget, device=sizes.device, dtype=sizes.dtype)
        if budget.ndim == 0:
            budget = budget.expand(len(self.channels))
        if (
            budget.shape != (len(self.channels),)
            or not torch.isfinite(budget).all()
            or (budget < sizes[1]).any()
        ):
            raise ValueError(
                "full-input Lipschitz budgets must cover the reference bounds"
            )
        object.__setattr__(self, "sizes", sizes.detach().clone())
        object.__setattr__(self, "lipschitz", budget.detach().clone())

    def validate_harmonics(self):
        value = self.coefficients
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or min(value.shape) < 1
            or value.dtype not in (torch.complex64, torch.complex128)
            or len(self.channels) != value.shape[0]
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                "SFNO certification requires finite harmonic coefficients and finite epsilon>=0"
            )
        degree = torch.arange(value.shape[1], device=value.device)[:, None]
        order = torch.arange(value.shape[2], device=value.device)[None, :]
        if (value[:, order > degree] != 0).any() or (value[:, :, 0].imag != 0).any():
            raise ValueError(
                "real harmonic inputs require m<=l and real m=0 coefficients"
            )

    def evaluate(self, points):
        if self.coefficients is None:
            value = self.reference(points)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (len(points), len(self.channels))
                or value.dtype != self.sizes.dtype
                or value.device != self.sizes.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    "reference must return finite [points, channels] values"
                )
            return value
        return harmonic_eval(self.coefficients, points[:, 0], points[:, 1])

    def envelope(self, points):
        values = self.evaluate(points)
        return values - self.epsilon, values + self.epsilon

    def admit(self, deviation, lipschitz):
        for value, limit in ((deviation, self.epsilon), (lipschitz, self.lipschitz)):
            value = torch.as_tensor(
                value, dtype=self.sizes.dtype, device=self.sizes.device
            )
            if (
                value.shape not in ((), (len(self.channels),))
                or not torch.isfinite(value).all()
                or (value < 0).any()
                or (value > limit).any()
            ):
                raise ValueError("field budgets exceed the continuous input set")

@dataclass
class SFNOModel:
    graph: Graph
    input_grid: object
    output_grid: object
    channels: tuple[str, ...]
    transforms: dict | None = None

    @torch.no_grad()
    def native(self, values, callback=None):
        from lean_no.graph import execute

        def apply(op, args):
            data, x = op.data, args[0]
            if op.kind == "affine":
                result = functional.conv2d(
                    x, data["weight"].to(x)[:, :, None, None], data["bias"].to(x)
                )
            elif op.kind == "activation":
                result = functional.gelu(x)
            elif op.kind == "normalize":
                result = functional.instance_norm(
                    x,
                    weight=data["weight"].to(x),
                    bias=data["bias"].to(x),
                    eps=data["epsilon"],
                )
            elif op.kind == "spectral":
                sht, isht = self.transforms[op.name]
                modes = sht(x.float()).contiguous()
                if not op.data.get("projection", False):
                    modes = torch.einsum(
                        "bilm,oil->bolm", modes, data["weight"].to(modes)
                    )
                result = isht(modes.contiguous()).to(x)
            elif op.kind == "add":
                result = args[0] + args[1]
            elif op.kind == "concat":
                result = torch.cat(args, dim=1)
            else:
                raise NotImplementedError(op.kind)
            if callback is not None:
                callback(op.name, result)
            return result

        return execute(self.graph, values, apply)

def check_children(module, allowed):
    unknown = set(dict(module.named_children())) - set(allowed)
    if unknown or module.training:
        raise NotImplementedError(
            f"unsupported or training module {type(module).__name__}: {unknown}"
        )

def adapt_backbone(backbone, channels):
    check_children(
        backbone,
        (
            "encoder",
            "decoder",
            "blocks",
            "pos_drop",
            "trans",
            "itrans",
            "trans_down",
            "itrans_up",
            "residual_transform",
        ),
    )
    if hasattr(backbone, "pos_embed"):
        raise NotImplementedError("positional embeddings are unsupported")
    ops = []

    def append(name, kind, parents, **data):
        data = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }
        ops.append(Operator(name, kind, tuple(parents), data))
        return len(ops)

    def pointwise(module, parent, name, grid):
        if isinstance(module, nn.Sequential):
            for key, child in module.named_children():
                parent = pointwise(child, parent, f"{name}.{key}", grid)
            return parent
        if hasattr(module, "fwd"):
            check_children(module, ("fwd",))
            return pointwise(module.fwd, parent, name, grid)
        if type(module) in (nn.Identity, nn.Dropout, nn.Dropout2d):
            if module.training:
                raise ValueError("dropout must be disabled")
            return parent
        if type(module) is nn.Conv2d:
            if (
                module.kernel_size != (1, 1)
                or module.stride != (1, 1)
                or module.padding != (0, 0)
                or module.groups != 1
            ):
                raise NotImplementedError(
                    "only pointwise channel convolutions are supported"
                )
            weight = module.weight.detach()[:, :, 0, 0]
            bias = (
                weight.new_zeros(weight.shape[0])
                if module.bias is None
                else module.bias.detach()
            )
            return append(name, "affine", (parent,), weight=weight, bias=bias)
        if type(module) is nn.GELU and module.approximate == "none":
            return append(name, "activation", (parent,), activation="gelu")
        if (
            type(module) is nn.InstanceNorm2d
            and not module.track_running_stats
            and module.eps > 0
        ):
            weight = (
                module.weight.detach()
                if module.affine
                else grid.latitude.new_ones(module.num_features)
            )
            bias = module.bias.detach() if module.affine else torch.zeros_like(weight)
            return append(
                name,
                "normalize",
                (parent,),
                weight=weight,
                bias=bias,
                epsilon=module.eps,
                grid=grid,
            )
        raise NotImplementedError(
            f"unsupported pointwise operation: {type(module).__name__}"
        )

    def transform(sht, isht):
        if (
            type(sht).__name__ != "RealSHT"
            or type(isht).__name__ != "InverseRealSHT"
            or sht.norm != "ortho"
            or isht.norm != "ortho"
            or not sht.csphase
            or not isht.csphase
            or (sht.lmax, sht.mmax) != (isht.lmax, isht.mmax)
            or min(sht.lmax, sht.mmax) > min(sht.nlon, isht.nlon) // 2
        ):
            raise NotImplementedError(
                "expected orthonormal, CS-phase, unaliased real SHTs"
            )
        parameter = next(backbone.parameters())
        grid = sphere_grid(
            (isht.nlat, isht.nlon),
            isht.grid,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        return dict(
            sht=sht,
            isht=isht,
            degrees=sht.lmax,
            orders=sht.mmax,
            grid=grid,
            source_shape=(sht.nlat, sht.nlon),
            source_kind=sht.grid,
            target_kind=isht.grid,
        )

    first = backbone.blocks[0].filter.filter
    initial = sphere_grid(
        (first.forward_transform.nlat, first.forward_transform.nlon),
        first.forward_transform.grid,
        dtype=next(backbone.parameters()).dtype,
        device=next(backbone.parameters()).device,
    )
    grid = initial
    parent = pointwise(backbone.encoder, 0, "encoder", grid)
    parent = pointwise(backbone.pos_drop, parent, "pos_drop", grid)
    for index, block in enumerate(backbone.blocks):
        name = f"blocks.{index}"
        check_children(
            block,
            (
                "filter",
                "norm0",
                "inner_skip",
                "act_layer0",
                "mlp",
                "norm1",
                "drop_path",
                "outer_skip",
                "act_layer1",
            ),
        )
        check_children(block.filter, ("filter",))
        spectral = block.filter.filter
        check_children(spectral, ("forward_transform", "inverse_transform"))
        if (
            spectral.operator_type != "dhconv"
            or spectral.separable
            or spectral.num_groups != 1
            or spectral.weight.ndim != 4
            or spectral.weight.shape[0] != 1
            or not spectral.weight.is_complex()
        ):
            raise NotImplementedError(
                "expected complex, nonseparable, single-group dhconv weights"
            )
        data = transform(spectral.forward_transform, spectral.inverse_transform)
        if spectral.weight.shape[-1] != data["degrees"]:
            raise ValueError("spectral weights disagree with retained degrees")
        if grid.shape != data["source_shape"]:
            raise ValueError("inconsistent native grid sequence")
        grid = data["grid"]
        residual = parent
        if spectral.scale_residual:
            residual = append(
                name + ".resample", "spectral", (parent,), projection=True, **data
            )
        parent = append(
            name + ".spectral",
            "spectral",
            (parent,),
            weight=spectral.weight.detach()[0].permute(1, 0, 2),
            **data,
        )
        if hasattr(spectral, "bias"):
            bias = spectral.bias.detach().flatten()
            parent = append(
                name + ".bias",
                "affine",
                (parent,),
                weight=torch.eye(len(bias), device=bias.device, dtype=bias.dtype),
                bias=bias,
            )
        parent = pointwise(block.norm0, parent, name + ".norm0", grid)
        if hasattr(block, "inner_skip"):
            skip = pointwise(block.inner_skip, residual, name + ".inner_skip", grid)
            parent = append(name + ".inner_sum", "add", (parent, skip))
        if hasattr(block, "act_layer0"):
            parent = pointwise(block.act_layer0, parent, name + ".act_layer0", grid)
        if hasattr(block, "mlp"):
            parent = pointwise(block.mlp, parent, name + ".mlp", grid)
        parent = pointwise(block.norm1, parent, name + ".norm1", grid)
        parent = pointwise(block.drop_path, parent, name + ".drop_path", grid)
        if hasattr(block, "outer_skip"):
            skip = pointwise(block.outer_skip, residual, name + ".outer_skip", grid)
            parent = append(name + ".outer_sum", "add", (parent, skip))
        if hasattr(block, "act_layer1"):
            parent = pointwise(block.act_layer1, parent, name + ".act_layer1", grid)
        # A stable name for block-by-block parity, independent of optional skips.
        ops[parent - 1] = replace(ops[parent - 1], name=name)
    parent = pointwise(backbone.decoder, parent, "decoder", grid)
    if backbone.big_skip:
        residual = 0
        if initial.shape != grid.shape:
            residual = append(
                "resample",
                "spectral",
                (0,),
                projection=True,
                **transform(backbone.trans_down, backbone.itrans_up),
            )
        skip = pointwise(
            backbone.residual_transform, residual, "residual_transform", grid
        )
        parent = append("output", "add", (parent, skip))
    if parent != len(ops) or len(channels) != backbone.out_chans:
        raise ValueError("invalid output interface")
    transforms = {
        op.name: (op.data["sht"], op.data["isht"])
        for op in ops
        if op.kind == "spectral"
    }
    ops = tuple(
        replace(
            op,
            data={
                key: value
                for key, value in op.data.items()
                if key not in ("sht", "isht")
            },
        )
        for op in ops
    )
    return SFNOModel(
        Graph(ops, {"basis": "spherical"}), initial, grid, tuple(channels), transforms
    )

@torch.no_grad()
def native_parity(backbone, model, values):
    (recorded, report) = ({}, {})
    handles = [
        block.register_forward_hook(
            lambda module, args, output, name=f"blocks.{index}": recorded.update(
                {name: output}
            )
        )
        for (index, block) in enumerate(backbone.blocks)
    ]
    try:
        expected = backbone(values)
    finally:
        for handle in handles:
            handle.remove()

    def compare(name, output):
        if name in recorded:
            target = recorded.pop(name)
            torch.testing.assert_close(
                output,
                target,
                rtol=0.0001,
                atol=1e-05,
                msg=f"native adapter mismatch at {name}",
            )
            report[name] = float((output - target).abs().max())

    actual = model.native(values, compare)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0001,
        atol=1e-05,
        msg="native adapter final output mismatch",
    )
    report["output"] = float((actual - expected).abs().max())
    return (expected, report)
