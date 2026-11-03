import torch
from neuralop.models import FNO

from lean_no.adapters.fno import cast_module
from lean_no.bounds.operators import affine_weight

def make_model(dimension, channels=1):
    model = cast_module(
        FNO(
            n_modes=(2,) * dimension,
            in_channels=channels,
            out_channels=1,
            hidden_channels=2,
            n_layers=2,
            lifting_channel_ratio=1,
            projection_channel_ratio=1,
            positional_embedding=None,
            non_linearity=torch.nn.functional.tanh,
            use_channel_mlp=False,
            fno_skip="linear",
        ),
        torch.float64,
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for layer in (
            *model.lifting.fcs,
            *model.fno_blocks.fno_skips,
            *model.projection.fcs,
        ):
            affine_weight(layer).fill_(0.1)
    return model
