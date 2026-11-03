from __future__ import annotations

import math

import torch

@torch.no_grad()
def grid_kernel(
    conv: torch.nn.Module,
    resolution: tuple[int, ...],
) -> torch.Tensor:
    from lean_no.graph import Operator

    if isinstance(conv, Operator):
        coordinates = tuple(
            torch.arange(n, device=conv.weight.device, dtype=conv.weight.real.dtype) / n
            for n in resolution
        )
        return kernel(
            conv, torch.meshgrid(*coordinates, indexing="ij"), (1.0,) * len(resolution)
        )
    weight = conv.weight.to_tensor()
    dtype = weight.real.dtype if weight.is_complex() else weight.dtype
    point_count = math.prod(resolution)
    values = torch.empty(
        conv.out_channels,
        conv.in_channels,
        *resolution,
        device=weight.device,
        dtype=dtype,
    )
    shape = (1, conv.out_channels, *((1,) * len(resolution)))
    bias = getattr(conv, "bias", None)
    bias = None if bias is None else bias.detach().to(values).reshape(shape)
    index = (0,) * len(resolution)
    for channel in range(conv.in_channels):
        impulse = torch.zeros(
            1,
            conv.in_channels,
            *resolution,
            device=weight.device,
            dtype=dtype,
        )
        impulse[(0, channel, *index)] = point_count
        output = conv(impulse)
        if bias is not None:
            output = output - bias
        values[:, channel] = output[0]
    return values

def mode_weights(
    conv: torch.nn.Module,
    real_dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    from lean_no.graph import Operator

    if isinstance(conv, Operator):
        return conv.weight.to(
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        ), conv.data["modes"]
    tensor = conv.weight.to_tensor().detach()
    if getattr(conv, "separable", False):
        channels = tensor.shape[0]
        diagonal = torch.eye(channels, device=tensor.device, dtype=tensor.dtype)
        tensor = (
            diagonal.reshape(channels, channels, *([1] * (tensor.ndim - 1)))
            * tensor[:, None]
        )
    dimension = tensor.ndim - 2
    if dimension not in (1, 2, 3):
        raise NotImplementedError(
            "Fourier kernels support one, two, or three dimensions"
        )
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    active = tuple(int(value) for value in conv.n_modes)
    slices = (slice(None), slice(None)) + tuple(
        slice((size - count) // 2, (size - count) // 2 + count)
        if axis < dimension - 1
        else slice(0, count)
        for axis, (size, count) in enumerate(zip(tensor.shape[2:], active))
    )
    tensor = tensor[slices]
    order = (1, 0, *range(2, tensor.ndim))
    weight = tensor.permute(order).to(dtype=complex_dtype)
    shape = weight.shape[2:]
    modes = []
    for axis, count in enumerate(shape):
        if axis < dimension - 1:
            values = torch.arange(
                -(count // 2),
                count - count // 2,
                device=weight.device,
                dtype=real_dtype,
            )
        else:
            values = torch.arange(
                count,
                device=weight.device,
                dtype=real_dtype,
            )
        modes.append(values)
    return weight, tuple(modes)

def kernel_series_impl(
    conv: torch.nn.Module,
    coordinates: tuple[torch.Tensor, ...],
    L: tuple[float, ...],
    orders: tuple[int, ...],
    cell_widths: tuple[float, ...] | None,
) -> torch.Tensor:
    dimension = len(coordinates)
    if len(L) != dimension or len(orders) != dimension:
        raise ValueError("kernel geometry lengths must match")
    if cell_widths is not None and len(cell_widths) != dimension:
        raise ValueError("kernel cell widths must match the dimension")
    if any(value.shape != coordinates[0].shape for value in coordinates):
        raise ValueError("kernel coordinates must have matching shapes")

    reference = coordinates[0]
    weight, modes = mode_weights(conv, reference.dtype)
    weight = weight.to(device=reference.device)
    frequencies = tuple(
        2 * math.pi * mode.to(reference) / float(length)
        for mode, length in zip(modes, L)
    )
    weighted = weight
    for axis, (frequency, order) in enumerate(zip(frequencies, orders)):
        shape = [1] * dimension
        shape[axis] = frequency.numel()
        weighted = weighted * (1j * frequency).pow(order).reshape(
            1,
            1,
            *shape,
        )
        if cell_widths is not None:
            half_phase = 0.5 * float(cell_widths[axis]) * frequency
            average = torch.exp(-1j * half_phase) * torch.sinc(half_phase / math.pi)
            weighted = weighted * average.reshape(1, 1, *shape)

    point_count = reference.numel()
    symmetry = torch.ones_like(modes[-1])
    symmetry[1:] = 2
    shape = [1] * dimension
    shape[-1] = symmetry.numel()
    weighted = weighted * symmetry.reshape(1, 1, *shape)
    channels = weight.shape[0] * weight.shape[1]
    # Bound the temporary point-by-mode array independently of grid resolution.
    result = torch.empty(
        channels, point_count, device=reference.device, dtype=reference.dtype
    )
    flat_coordinates = tuple(value.reshape(-1) for value in coordinates)
    for begin in range(0, point_count, 16384):
        end = min(begin + 16384, point_count)
        phase = torch.ones(
            end - begin,
            *(mode.numel() for mode in modes),
            device=reference.device,
            dtype=weight.dtype,
        )
        for axis, (coordinate, frequency) in enumerate(
            zip(flat_coordinates, frequencies)
        ):
            shape = [1] * dimension
            shape[axis] = frequency.numel()
            phase = phase * torch.exp(
                1j
                * coordinate[begin:end].reshape(-1, *([1] * dimension))
                * frequency.reshape(1, *shape)
            )
        values = phase.reshape(end - begin, -1) @ weighted.reshape(channels, -1).mT
        result[:, begin:end] = values.real.mT
    return result.reshape(
        weight.shape[0],
        weight.shape[1],
        *reference.shape,
    )

def kernel_series(
    conv: torch.nn.Module,
    coordinates: tuple[torch.Tensor, ...],
    L: tuple[float, ...],
    orders: tuple[int, ...],
) -> torch.Tensor:
    return kernel_series_impl(conv, coordinates, L, orders, None)

def kernel(
    conv: torch.nn.Module,
    coordinates: torch.Tensor | tuple[torch.Tensor, ...],
    L: float | tuple[float, ...],
) -> torch.Tensor:
    if isinstance(coordinates, torch.Tensor):
        coordinates = (coordinates,)
    if not isinstance(L, tuple):
        L = (float(L),)
    return kernel_series(conv, coordinates, L, (0,) * len(coordinates))

def kernel_with_partials(
    conv: torch.nn.Module,
    axes: tuple[torch.Tensor, ...],
    L: tuple[float, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    coordinates = torch.meshgrid(*axes, indexing="ij")
    values = kernel(conv, coordinates, L)
    partials = tuple(
        kernel_partial(conv, coordinates, L, axis) for axis in range(len(axes))
    )
    return values, partials

def kernel_partial(
    conv: torch.nn.Module,
    coordinates: torch.Tensor | tuple[torch.Tensor, ...],
    L: float | tuple[float, ...],
    axis: int,
) -> torch.Tensor:
    if isinstance(coordinates, torch.Tensor):
        coordinates = (coordinates,)
    if not 0 <= axis < len(coordinates):
        raise ValueError("kernel axis is outside the spatial dimension")
    if not isinstance(L, tuple):
        L = (float(L),)
    orders = [0] * len(coordinates)
    orders[axis] = 1
    return kernel_series(conv, coordinates, L, tuple(orders))

def kernel_gain(
    conv: torch.nn.Module,
    L: float | tuple[float, ...],
    orders: tuple[int, ...],
    real_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    from lean_no.graph import Operator

    tensor = (
        conv.weight if isinstance(conv, Operator) else conv.weight.to_tensor().detach()
    )
    if real_dtype is None:
        real_dtype = tensor.real.dtype if tensor.is_complex() else tensor.dtype
    if not isinstance(L, tuple):
        L = (float(L),)
    weight, modes = mode_weights(conv, real_dtype)
    if len(L) != len(modes) or len(orders) != len(modes):
        raise ValueError("kernel geometry lengths must match")
    multiplier = torch.ones(
        *(mode.numel() for mode in modes),
        device=weight.device,
        dtype=real_dtype,
    )
    for axis, (mode, length, order) in enumerate(zip(modes, L, orders)):
        frequency = (2 * math.pi * mode / float(length)).abs().pow(order)
        shape = [1] * len(modes)
        shape[axis] = frequency.numel()
        multiplier = multiplier * frequency.reshape(shape)
    symmetry = torch.ones_like(modes[-1])
    symmetry[1:] = 2
    shape = [1] * len(modes)
    shape[-1] = symmetry.numel()
    # Real hypot avoids a runtime-compiled complex abs kernel on newer GPUs.
    magnitude = torch.hypot(weight.real, weight.imag)
    gain = (magnitude * multiplier[None, None] * symmetry.reshape(1, 1, *shape)).sum(
        dim=tuple(range(2, weight.ndim))
    )
    return gain
