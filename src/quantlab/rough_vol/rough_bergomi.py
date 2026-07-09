from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoughBergomiParams:
    hurst: float
    eta: float
    rho: float
    xi0: float

    def validate(self) -> None:
        if not 0 < self.hurst < 0.5:
            raise ValueError("hurst must be in (0, 0.5) for rough volatility")
        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.xi0 <= 0:
            raise ValueError("xi0 must be positive")


def simulate_rough_bergomi(
    spot: float,
    maturity: float,
    steps: int,
    paths: int,
    params: RoughBergomiParams,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate rough Bergomi spot and variance paths with a direct Volterra scheme."""
    params.validate()
    if spot <= 0 or maturity <= 0:
        raise ValueError("spot and maturity must be positive")
    if steps < 2 or paths < 1:
        raise ValueError("steps must be >= 2 and paths must be >= 1")

    rng = np.random.default_rng(seed)
    dt = maturity / steps
    times = np.linspace(0.0, maturity, steps + 1)
    dw1 = rng.normal(0.0, np.sqrt(dt), size=(paths, steps))
    dw_perp = rng.normal(0.0, np.sqrt(dt), size=(paths, steps))
    dw2 = params.rho * dw1 + np.sqrt(1.0 - params.rho**2) * dw_perp

    variance = np.full((paths, steps + 1), params.xi0)
    spot_paths = np.full((paths, steps + 1), spot, dtype=float)
    kernel_scale = np.sqrt(2.0 * params.hurst)

    for i in range(1, steps + 1):
        lags = times[i] - times[:i]
        kernel = np.maximum(lags, 1e-12) ** (params.hurst - 0.5)
        volterra = kernel_scale * (dw1[:, :i] @ kernel)
        variance[:, i] = params.xi0 * np.exp(params.eta * volterra - 0.5 * params.eta**2 * times[i] ** (2.0 * params.hurst))
        prev_var = np.maximum(variance[:, i - 1], 0.0)
        spot_paths[:, i] = spot_paths[:, i - 1] * np.exp(-0.5 * prev_var * dt + np.sqrt(prev_var) * dw2[:, i - 1])

    return spot_paths, variance
