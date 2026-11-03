def spectral_kernel_l2(weight):
    symmetry = weight.real.new_full((weight.shape[-1],), 2)
    symmetry[0] = 1
    return (weight.abs().square() * symmetry).sum((-1, -2)).sqrt()
