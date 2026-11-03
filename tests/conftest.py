import pytest
import torch

@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)

@pytest.fixture(autouse=True)
def fixed_random_seed():
    previous = torch.random.get_rng_state()
    torch.manual_seed(0)
    yield
    torch.random.set_rng_state(previous)
