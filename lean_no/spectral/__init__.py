import torch
from neuralop.layers.spectral_convolution import SpectralConv as BaseSpectralConv

def real_inverse(spectrum, shape, norm="backward"):
    shape = tuple(shape)
    if not spectrum.is_complex() or not shape or len(shape) > spectrum.ndim:
        raise ValueError("expected a complex half spectrum and output shape")
    if any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("output sizes must be positive integers")
    target = (*shape[:-1], shape[-1] // 2 + 1)
    current = spectrum.shape[-len(shape) :]
    if current == target:
        output = spectrum.clone()
    else:
        output = spectrum.new_zeros(*spectrum.shape[: -len(shape)], *target)
        slices = (Ellipsis, *(slice(0, min(a, b)) for a, b in zip(current, target)))
        output[slices] = spectrum[slices]
    indices = [0]
    if shape[-1] % 2 == 0:
        indices.append(shape[-1] // 2)
    for index in indices:
        plane = output[..., index]
        reflected = plane
        for axis, size in enumerate(shape[:-1]):
            pairs = (-torch.arange(size, device=output.device)) % size
            reflected = reflected.index_select(axis - len(shape) + 1, pairs)
        output[..., index] = (plane + reflected.conj()) / 2
    dimensions = tuple(range(-len(shape), 0))
    return torch.fft.irfftn(output, s=shape, dim=dimensions, norm=norm)

class SpectralConv(BaseSpectralConv):
    def forward(self, x, output_shape=None):
        if self.complex_data:
            return super().forward(x, output_shape)
        shape = x.shape[-self.order :]
        dimensions = tuple(range(-self.order, 0))
        if self.fno_block_precision == "half":
            x = x.half()
        spectrum = torch.fft.rfftn(x, dim=dimensions, norm=self.fft_norm)
        if self.fno_block_precision == "mixed":
            spectrum = spectrum.chalf()
        if self.order > 1:
            spectrum = torch.fft.fftshift(spectrum, dim=dimensions[:-1])
        sizes = spectrum.shape[-self.order :]
        active = tuple(
            min(size, mode, maximum)
            for size, mode, maximum in zip(sizes, self.n_modes, self.max_n_modes)
        )
        starts = tuple(size - count for size, count in zip(self.max_n_modes, active))
        slices = [slice(None)] * (1 if self.separable else 2)
        slices.extend(
            slice(start // 2, -(start - start // 2)) if start else slice(None)
            for start in starts[:-1]
        )
        slices.append(slice(0, active[-1]))
        weight = self.weight[tuple(slices)]
        counts = weight.shape[1 if self.separable else 2 :]
        selection = (
            slice(None),
            slice(None),
            *(
                slice(size // 2 - count // 2, size // 2 + count // 2 + count % 2)
                for size, count in zip(sizes[:-1], counts[:-1])
            ),
            slice(0, counts[-1]),
        )
        output = spectrum.new_zeros(x.shape[0], self.out_channels, *sizes)
        output[selection] = self._contract(
            spectrum[selection], weight, separable=self.separable
        )
        if self.order > 1:
            output = torch.fft.ifftshift(output, dim=dimensions[:-1])
        if output_shape is not None:
            shape = tuple(output_shape)
        elif self.resolution_scaling_factor is not None:
            shape = tuple(
                round(size * scale)
                for size, scale in zip(shape, self.resolution_scaling_factor)
            )
        result = real_inverse(output, shape, self.fft_norm)
        return result if self.bias is None else result + self.bias
