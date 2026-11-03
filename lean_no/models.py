import torch
from neuralop.models import FNO
from torch import nn

from lean_no.adapters.fno import cast_module
from lean_no.paths import repo_path
from lean_no.spectral import SpectralConv

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def make_fno(config: dict) -> FNO:
    config = dict(config)
    # Older Poisson 2D checkpoints used the name "activation".
    if "activation" in config:
        activation = config.pop("activation")
        if config.get("non_linearity", activation) != activation:
            raise ValueError("conflicting activation and non_linearity settings")
        config["non_linearity"] = activation
    defaults = dict(
        in_channels=1,
        out_channels=1,
        n_layers=1,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        non_linearity="relu",
        use_channel_mlp=False,
        positional_embedding="grid",
        domain_padding=None,
        fno_skip="linear",
        factorization=None,
    )
    unknown = config.keys() - defaults.keys() - {"n_modes", "hidden_channels"}
    if unknown:
        raise ValueError(f"unsupported FNO settings: {', '.join(sorted(unknown))}")
    settings = defaults | config
    settings["n_modes"] = tuple(settings["n_modes"])
    activation = settings.pop("non_linearity")
    activations = dict(relu=nn.ReLU, tanh=nn.Tanh, gelu=nn.GELU, identity=nn.Identity)
    if activation not in activations:
        raise ValueError(f"unsupported FNO activation: {activation!r}")
    return FNO(
        conv_module=SpectralConv,
        non_linearity=activations[activation](),
        **settings,
    )

def restore_fno(checkpoint, *, device=None, dtype=None):
    model = make_fno(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    if dtype is not None:
        model = cast_module(model, dtype)
    return model.to(device or pick_device()).eval()

def load_fno(path, *, device=None, dtype=None, legacy_config=None):
    checkpoint = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        if legacy_config is None:
            raise ValueError("raw state dictionaries require legacy_config")
        checkpoint = dict(model_config=legacy_config, model_state_dict=checkpoint)
    return restore_fno(checkpoint, device=device, dtype=dtype)
