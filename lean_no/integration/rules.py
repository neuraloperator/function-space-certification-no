import math
from functools import lru_cache

import numpy as np

@lru_cache(maxsize=32)
def interval_rule(method, order=2):
    if method == "rectangle":
        nodes, weights = np.array([0.0]), np.array([1.0])
    elif method == "gaussian":
        from lean_no.integration.gaussian import gaussian_rule

        if type(order) is not int or order < 2:
            raise ValueError("Gaussian order must be at least two")
        nodes, weights = gaussian_rule(order)
        nodes, weights = (nodes + 1) / 2, weights / 2
    elif method == "arb":
        nodes, weights = np.array([0.5]), np.array([1.0])
    elif method == "clenshaw_curtis":
        if type(order) is not int or order < 2:
            raise ValueError("CC order must be at least two")
        n = order - 1
        theta = np.arange(order) * math.pi / n
        nodes = (1 - np.cos(theta)) / 2
        weights = np.zeros(order)
        v = np.ones(n - 1)
        if n % 2 == 0:
            weights[0] = weights[-1] = 1 / (n * n - 1)
            for k in range(1, n // 2):
                v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k * k - 1)
            v -= np.cos(n * theta[1:-1]) / (n * n - 1)
        else:
            weights[0] = weights[-1] = 1 / (n * n)
            for k in range(1, (n + 1) // 2):
                v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k * k - 1)
        weights[1:-1] = 2 * v / n
        weights /= 2
    else:
        raise ValueError(f"No positive interval rule for {method}")
    edges = np.concatenate(([0.0], weights.cumsum()))
    a, b = edges[:-1] - nodes, edges[1:] - nodes
    distance = float(((b * abs(b) - a * abs(a)) / 2).sum())
    return nodes, weights, distance
