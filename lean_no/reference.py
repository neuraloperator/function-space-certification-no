import numpy as np
from scipy import fft

def cole_hopf_reference(initial, times, resolution, viscosity=0.01):
    k = np.arange(resolution // 2 + 1, dtype=np.longdouble)
    pi = np.arccos(np.longdouble(-1))
    omega = 2 * pi * k
    interpolant = np.zeros(len(k), dtype=np.clongdouble)
    for frequency, value in initial.terms:
        mode = frequency[0]
        if mode >= 0:
            if mode != int(mode) or mode >= len(k):
                raise ValueError("reference requires resolved integer Fourier modes")
            interpolant[int(mode)] += value
    mean = interpolant[0].real
    primitive = interpolant.copy()
    primitive[0] = 0
    primitive[1:] /= 1j * omega[1:]
    exponent = -fft.irfft(primitive, n=resolution, norm="forward") / (2 * viscosity)
    heat = fft.rfft(np.exp(exponent - exponent.max()), norm="forward")
    result = []
    for t in times:
        if t == 0:
            result.append(fft.irfft(interpolant, n=resolution, norm="forward"))
            continue
        spectrum = heat * np.exp(-viscosity * omega**2 * t - 1j * omega * mean * t)
        phi = fft.irfft(spectrum, n=resolution, norm="forward")
        if np.min(phi) <= 0:
            raise FloatingPointError("nonpositive numerical Cole-Hopf potential")
        derivative = fft.irfft(1j * omega * spectrum, n=resolution, norm="forward")
        result.append(mean - 2 * viscosity * derivative / phi)
    return np.asarray(result, dtype=np.float64)
