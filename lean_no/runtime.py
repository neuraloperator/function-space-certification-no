import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def sync(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

def measure(operation, device):
    with Timer(device) as timer:
        result = operation()
    return result, timer.seconds, timer.peak_bytes

@dataclass
class Timer:

    device: object
    reset_peak: bool = True
    seconds: float = 0.0
    peak_bytes: int = 0

    def __enter__(self):
        self.device = torch.device(self.device)
        sync(self.device)
        if self.device.type == "cuda" and self.reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.seconds = time.perf_counter() - self.started
        if self.device.type == "cuda":
            self.peak_bytes = torch.cuda.max_memory_allocated(self.device)

def json_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.is_complex():
            return {"real": json_value(value.real), "imag": json_value(value.imag)}
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value

def target_hashes(backbone, inputs):
    parameters, coefficients = hashlib.sha256(), hashlib.sha256()
    for name, tensor in backbone.named_parameters():
        parameters.update(name.encode())
        parameters.update(str(tuple(tensor.shape)).encode())
        parameters.update(str(tensor.dtype).encode())
        parameters.update(tensor.detach().cpu().numpy().tobytes())
    for value in (
        str(inputs.channels).encode(),
        str(inputs.epsilon).encode(),
        str(inputs.coefficients.dtype).encode(),
        inputs.coefficients.cpu().numpy().tobytes(),
    ):
        coefficients.update(value)
    if inputs.epsilon and hasattr(inputs, "lipschitz"):
        coefficients.update(inputs.lipschitz.detach().cpu().numpy().tobytes())
    return dict(parameters=parameters.hexdigest(), input=coefficients.hexdigest())

def write_json(path, record, strict=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record if strict else json_value(record), indent=2, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)
