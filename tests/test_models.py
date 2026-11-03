import pytest
import torch

from lean_no.models import load_fno, make_fno
from lean_no.paths import repo_path

@pytest.mark.parametrize(
    "problem", ["poisson_1d", "poisson_2d", "helmholtz_1d", "burgers_1d"]
)
@pytest.mark.parametrize("property_name", ["robustness", "solution"])
def test_paper_checkpoint(problem, property_name, tmp_path, monkeypatch):
    path = f"models/{problem}/{property_name}.pt"
    checkpoint = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    raw = "model_state_dict" not in checkpoint
    legacy = {"n_modes": [32], "hidden_channels": 4, "positional_embedding": None}
    monkeypatch.chdir(tmp_path)
    if raw:
        with pytest.raises(ValueError, match="legacy_config"):
            load_fno(path, device="cpu")
    model = load_fno(path, device="cpu", dtype=torch.float64, legacy_config=legacy)
    state = checkpoint if raw else checkpoint["model_state_dict"]
    for key, original in state.items():
        if isinstance(original, torch.Tensor):
            actual = model.state_dict()[key]
            dtype = torch.complex128 if original.is_complex() else torch.float64
            assert actual.dtype == dtype
            torch.testing.assert_close(actual, original.to(dtype), rtol=0, atol=0)
    assert not model.training
    resolution = (32, 32) if problem in ("poisson_2d", "burgers_1d") else (64,)
    with torch.no_grad():
        output = model(
            torch.randn(1, model.in_channels, *resolution, dtype=torch.float64)
        )
    assert output.shape == (1, 1, *resolution)
    assert torch.isfinite(output).all()

def test_conflicting_checkpoint_activation_is_rejected():
    with pytest.raises(ValueError, match="conflicting"):
        make_fno(
            dict(
                n_modes=[4], hidden_channels=2, activation="relu", non_linearity="tanh"
            )
        )

@pytest.mark.parametrize("setting", ["preactivation", "norm", "n_layer", "conv_module"])
def test_unsupported_checkpoint_settings_are_rejected(setting):
    with pytest.raises(ValueError, match=setting):
        make_fno({"n_modes": [4], "hidden_channels": 2, setting: True})
