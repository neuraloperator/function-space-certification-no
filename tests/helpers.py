import torch
from neuralop.models import FNO
from torch import nn

from lean_no.adapters.fno import cast_module

def fno(
    dimension=1, layers=1, activation=None, inputs=1, outputs=1, channels=2, norm=None
):
    return cast_module(
        FNO(
            n_modes=(2,) * dimension,
            in_channels=inputs,
            out_channels=outputs,
            hidden_channels=channels,
            n_layers=layers,
            lifting_channel_ratio=1,
            projection_channel_ratio=1,
            non_linearity=activation or torch.nn.Identity(),
            use_channel_mlp=False,
            positional_embedding=None,
            fno_skip="linear",
            factorization=None,
            norm=norm,
        ).eval(),
        torch.float64,
    )

def averaging_model(dimension=1):
    model = fno(dimension, channels=1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.zero_()
            if ("lifting" in name or "projection" in name) and "weight" in name:
                p.fill_(1)
        zero = (0, 0, 0) if dimension == 1 else (0, 0, 1, 0)
        model.fno_blocks.convs[0].weight.tensor[zero] = 1
    return model

def burgers_model():
    model = FNO(
        n_modes=(4, 4),
        hidden_channels=1,
        n_layers=1,
        in_channels=2,
        out_channels=1,
        non_linearity=nn.Identity(),
        positional_embedding=None,
        use_channel_mlp=False,
        lifting_channel_ratio=0,
        projection_channel_ratio=1,
    )
    model = cast_module(model.eval(), torch.float64)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.lifting.fcs[0].weight[0, :, 0] = torch.tensor(
            [0.3, 0.2], dtype=torch.float64
        )
        model.fno_blocks.fno_skips[0].conv.weight.fill_(1)
        for layer in model.projection.fcs:
            layer.weight.fill_(1)
    return model

def burgers_identity():
    # Identity pointwise network: v(x,t)=a(x)+t. All Fourier weights zero.
    model = (
        FNO(
            n_modes=(2, 2),
            hidden_channels=2,
            n_layers=1,
            in_channels=2,
            out_channels=1,
            non_linearity=nn.Identity(),
            lifting_channel_ratio=0,
            projection_channel_ratio=1,
            positional_embedding=None,
            use_channel_mlp=False,
            fno_skip="linear",
            factorization=None,
        )
        .double()
        .eval()
    )
    model.projection.fcs = torch.nn.ModuleList([torch.nn.Conv1d(2, 1, 1).double()])
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.lifting.fcs[0].weight[:, :, 0].copy_(torch.eye(2))
        model.fno_blocks.fno_skips[0].conv.weight[:, :, 0].copy_(torch.eye(2))
        model.projection.fcs[0].weight[0, :, 0].fill_(1)
    return cast_module(model, torch.float64)
