from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quantlab.options.surface_arbitrage import ArbitrageViolation, detect_surface_arbitrage


@dataclass(frozen=True)
class SurfaceRepairResult:
    repaired_prices: np.ndarray
    objective_value: float
    success: bool
    message: str
    violations_before: list[ArbitrageViolation]
    violations_after: list[ArbitrageViolation]


def repair_call_price_surface(
    maturities: np.ndarray,
    strikes: np.ndarray,
    call_prices: np.ndarray,
    tolerance: float = 1e-8,
) -> SurfaceRepairResult:
    """Project call prices onto basic calendar/monotonic/convex no-arbitrage constraints."""
    maturities = np.asarray(maturities, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(call_prices, dtype=float)
    if maturities.ndim != 1 or strikes.ndim != 1:
        raise ValueError("maturities and strikes must be one-dimensional")
    if prices.shape != (len(maturities), len(strikes)):
        raise ValueError("call_prices shape must be (len(maturities), len(strikes))")
    if np.any(np.diff(maturities) <= 0) or np.any(np.diff(strikes) <= 0):
        raise ValueError("maturities and strikes must be strictly increasing")

    original = prices.ravel()
    n_maturities, n_strikes = prices.shape

    def unpack(x: np.ndarray) -> np.ndarray:
        return x.reshape(n_maturities, n_strikes)

    constraints = []
    for t_idx in range(n_maturities - 1):
        for k_idx in range(n_strikes):
            constraints.append(
                {"type": "ineq", "fun": lambda x, t=t_idx, k=k_idx: unpack(x)[t + 1, k] - unpack(x)[t, k]}
            )

    for t_idx in range(n_maturities):
        for k_idx in range(n_strikes - 1):
            constraints.append(
                {"type": "ineq", "fun": lambda x, t=t_idx, k=k_idx: unpack(x)[t, k] - unpack(x)[t, k + 1]}
            )

    for t_idx in range(n_maturities):
        for k_idx in range(1, n_strikes - 1):
            left_width = strikes[k_idx] - strikes[k_idx - 1]
            right_width = strikes[k_idx + 1] - strikes[k_idx]
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, t=t_idx, k=k_idx, lw=left_width, rw=right_width: (
                        (unpack(x)[t, k + 1] - unpack(x)[t, k]) / rw - (unpack(x)[t, k] - unpack(x)[t, k - 1]) / lw
                    ),
                }
            )

    result = minimize(
        lambda x: float(np.sum((x - original) ** 2)),
        np.maximum(original, 0.0),
        method="SLSQP",
        bounds=[(0.0, None)] * len(original),
        constraints=constraints,
        options={"ftol": tolerance, "maxiter": 1_000},
    )
    repaired = unpack(result.x)
    return SurfaceRepairResult(
        repaired_prices=repaired,
        objective_value=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        violations_before=detect_surface_arbitrage(maturities, strikes, prices, tolerance=tolerance),
        violations_after=detect_surface_arbitrage(maturities, strikes, repaired, tolerance=max(tolerance * 10.0, 1e-7)),
    )
